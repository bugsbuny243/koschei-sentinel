from __future__ import annotations

import time
from functools import partial
from pathlib import Path
from typing import Literal
from pydantic import Field
from koschei_sentinel.models import StrictModel
from koschei_sentinel.production_foundation_catalog import FoundationCatalogCursor, load_foundation_catalog
from koschei_sentinel.production_foundation_prefetch import FoundationDoubleBuffer
from koschei_sentinel.production_foundation_worker_prefetch import FoundationWorkerPrefetch
from koschei_sentinel.production_mcore_activation_probe import MCoreActivationProbe
from koschei_sentinel.production_mcore_async_checkpoint import AsyncCheckpointConfig
from koschei_sentinel.production_mcore_ddp import wrap_runtime_with_megatron_ddp
from koschei_sentinel.production_mcore_distributed_commit import save_and_commit_state
from koschei_sentinel.production_mcore_moe_metrics_probe import clear_native_moe_metrics, read_native_moe_losses
from koschei_sentinel.production_mcore_native_checkpoint import initialize_mcore_native_parameters
from koschei_sentinel.production_mcore_optimizer_smoke import build_mcore_distributed_optimizer
from koschei_sentinel.production_mcore_recovery import RecoveryLedger, RecoveryPolicy, build_recovery_decision, distributed_recovery_consensus, fingerprint_batch
from koschei_sentinel.production_mcore_recovery_checkpoint_manager import RecoveryCheckpointManager
from koschei_sentinel.production_mcore_resume_bundle import attach_catalog_resume_state, restore_catalog_resume_bundle
from koschei_sentinel.production_mcore_router_probe import MCoreRouterProbe
from koschei_sentinel.production_mcore_stability_telemetry import MCoreStabilityStepTelemetry, StabilityThresholds, collect_mcore_stability_telemetry
from koschei_sentinel.production_mcore_training_checkpoint import MCoreTrainingState, load_mcore_training_checkpoint, save_mcore_training_checkpoint
from koschei_sentinel.production_mcore_training_trends import TrainingTrendSnapshot, TrainingTrendThresholds, TrainingTrendTracker

class CatalogTrainingConfig(StrictModel):
    schema_version: Literal["sentinel.catalog-training-config.v1"]="sentinel.catalog-training-config.v1"; max_steps:int=Field(gt=0); microbatches_per_step:int=Field(gt=0); learning_rate:float=Field(gt=0); weight_decay:float=Field(ge=0); clip_grad:float=Field(gt=0); global_seed:int=Field(ge=0,lt=2**63); checkpoint_every_steps:int=Field(gt=0); prefetch_workers:int=Field(gt=0); prefetch_depth:int=Field(ge=2)
    stability_thresholds:StabilityThresholds=Field(default_factory=lambda:StabilityThresholds(max_grad_norm=100,max_cuda_allocated_fraction=.95,min_expert_utilization_fraction=.5,max_expert_load_cv=2)); trend_thresholds:TrainingTrendThresholds=Field(default_factory=lambda:TrainingTrendThresholds(history_window=32,min_history_for_spike=3,max_loss_ratio_to_median=2.5,max_grad_norm_ratio_to_median=4,max_activation_rms_ratio_to_median=4,max_activation_abs=1e4,max_router_utilization_drop=.35)); recovery_policy:RecoveryPolicy=Field(default_factory=lambda:RecoveryPolicy(max_retries_per_batch=2,learning_rate_backoff=.5,min_learning_rate=1e-7,quarantine_after_failures=3,skip_quarantined_batch=True)); async_recovery_checkpoints:bool=True; async_checkpoint_strategy:Literal["nvrx","mcore"]="nvrx"; max_unfinalized_checkpoints:int=Field(default=1,ge=1,le=4); keep_recovery_slots:int=Field(default=2,ge=2,le=16)
class CatalogTrainingStepMetrics(StrictModel):
    global_step:int=Field(gt=0); local_tokens:int=Field(gt=0); global_tokens:int=Field(gt=0); elapsed_seconds:float=Field(gt=0); local_tokens_per_second:float=Field(gt=0); global_tokens_per_second:float=Field(gt=0); lm_loss:float|None=Field(default=None,ge=0); stability:MCoreStabilityStepTelemetry; trend:TrainingTrendSnapshot; recovery_attempts_before_success:int=Field(ge=0); effective_learning_rate:float=Field(gt=0); worker_prefetch:Literal[True]=True; pinned_memory:Literal[True]=True; async_h2d:Literal[True]=True; double_buffered:Literal[True]=True
class CatalogQuarantineEvent(StrictModel):
    batch_fingerprint:str=Field(pattern=r"^[a-f0-9]{64}$"); global_sequence_offset:int=Field(ge=0); global_window:int=Field(gt=0); failure_count:int=Field(gt=0); skipped:bool; reason_codes:list[str]=Field(default_factory=list)
class CatalogTrainingResult(StrictModel):
    schema_version:Literal["sentinel.catalog-training-result.v1"]="sentinel.catalog-training-result.v1"; status:Literal["catalog_training_completed"]="catalog_training_completed"; start_step:int=Field(ge=0); final_step:int=Field(gt=0); resumed_from_checkpoint:bool; catalog_sha256:str=Field(pattern=r"^[a-f0-9]{64}$"); step_metrics:list[CatalogTrainingStepMetrics]=Field(default_factory=list); quarantine_events:list[CatalogQuarantineEvent]=Field(default_factory=list); total_retries:int=Field(ge=0); final_learning_rate:float=Field(gt=0); final_checkpoint_dir:str|None=None; all_steps_stable:Literal[True]=True; distributed_recovery_consensus:Literal[True]=True; execution_authorized:Literal[False]=False

def _run_prefetched_batches(runtime,spec,batches):
    import torch
    from megatron.core import parallel_state
    from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
    from megatron.core.utils import get_batch_on_this_cp_rank
    cp=spec.topology.context_parallel_size; prepared=[]; losses=[]
    for b in batches:
        x={"tokens":b.tokens,"labels":b.labels,"loss_mask":b.loss_mask,"position_ids":b.position_ids,"attention_mask":None,"cu_seqlens":None}; prepared.append(get_batch_on_this_cp_rank(x,is_hybrid_cp=False,cp_group=parallel_state.get_context_parallel_group()) if cp>1 else x)
    if not prepared: raise RuntimeError("catalog prefetch produced no batches")
    def loss_func(mask,out):
        values=out.float().view(-1); mask=mask.float().view(-1)
        if values.numel()!=mask.numel(): raise RuntimeError("catalog loss and mask shape mismatch")
        denom=mask.sum()
        if denom.item()<=0: raise RuntimeError("catalog loss mask is empty")
        loss=(values*mask).sum()/denom; losses.append(float(loss.detach().item())); return loss,{"catalog_lm_loss":loss.detach()}
    def forward_step(it,model,checkpoint_activations_microbatch=None):
        del checkpoint_activations_microbatch; x=next(it); out=model(input_ids=x["tokens"],position_ids=x["position_ids"],attention_mask=x["attention_mask"],labels=x["labels"],loss_mask=x["loss_mask"]); return out,partial(loss_func,x["loss_mask"])
    get_forward_backward_func()(forward_step_func=forward_step,data_iterator=iter(prepared),model=runtime.model,num_microbatches=len(prepared),seq_length=prepared[0]["tokens"].shape[1]*cp,micro_batch_size=1,forward_only=False,collect_non_loss_data=False)
    payload=torch.tensor([sum(losses),len(losses)],device=prepared[0]["tokens"].device,dtype=torch.float64); torch.distributed.all_reduce(payload); return float(payload[0].item()/payload[1].item()) if payload[1].item()>0 else None

def _optimizer_step_result(v):
    ok=bool(v[0]) if isinstance(v,tuple) else (bool(v) if v is not None else True); raw=v[1] if isinstance(v,tuple) and len(v)>1 else 0
    try:g=float(raw.item())
    except AttributeError:g=float(raw or 0)
    return ok,g

def _restore_last_good(model,optimizer,checkpoint_dir,seed):
    if checkpoint_dir is None:return None
    return load_mcore_training_checkpoint(model,optimizer,checkpoint_dir=checkpoint_dir,expected_global_seed=seed)[0]
def _set_optimizer_lr(optimizer,lr):
    groups=getattr(optimizer,"param_groups",None)
    if groups is None:
        inner=getattr(optimizer,"optimizer",None); groups=getattr(inner,"param_groups",None)
    if groups is None: raise RuntimeError("Megatron optimizer exposes no mutable param_groups")
    for group in groups: group["lr"]=lr
def _attach(state,catalog,lr,ledger,trend,retries,quarantines): return attach_catalog_resume_state(state.model_copy(update={"learning_rate":lr}),catalog_sha256=catalog,current_learning_rate=lr,ledger=ledger,trend_tracker=trend,total_retries=retries,quarantine_events=[q.model_dump(mode="json") for q in quarantines])
def _sync_save(model,optimizer,state,path,catalog,lr,ledger,trend,retries,quarantines,*,device):
    proposed=_attach(state,catalog,lr,ledger,trend,retries,quarantines)
    def _save():
        save_mcore_training_checkpoint(model,optimizer,proposed,checkpoint_dir=path)
        return proposed
    saved,committed=save_and_commit_state(_save,checkpoint_dir=path,device=device)
    return saved,committed.checkpoint_dir

def run_catalog_training(runtime,spec,config,*,catalog_path,checkpoint_root,resume_checkpoint=None):
    import torch
    from megatron.core import parallel_state
    catalog=load_foundation_catalog(catalog_path)
    if catalog.sequence_length>spec.max_position_embeddings: raise ValueError("catalog sequence length exceeds model context window")
    if catalog.sequence_length%(2*spec.topology.context_parallel_size): raise ValueError("catalog sequence length must be divisible by 2*context_parallel_size")
    if config.prefetch_depth<config.prefetch_workers: raise ValueError("prefetch_depth must be >= prefetch_workers")
    if resume_checkpoint is None: initialize_mcore_native_parameters(runtime.model,global_seed=config.global_seed)
    ddp=wrap_runtime_with_megatron_ddp(runtime); model=ddp.model; optimizer=build_mcore_distributed_optimizer(ddp,learning_rate=config.learning_rate,weight_decay=config.weight_decay,clip_grad=config.clip_grad); router=MCoreRouterProbe(); activation=MCoreActivationProbe(); router.attach(model); activation.attach(model); ledger=RecoveryLedger.empty(); trend_tracker=TrainingTrendTracker(config.trend_thresholds); quarantines=[]; retries=0; cursor=None; resumed=resume_checkpoint is not None
    if resumed:
        state=load_mcore_training_checkpoint(model,optimizer,checkpoint_dir=resume_checkpoint,expected_global_seed=config.global_seed)[0]; bundle=restore_catalog_resume_bundle(state,expected_catalog_sha256=catalog.catalog_sha256,trend_thresholds=config.trend_thresholds); cursor=bundle.cursor; ledger=bundle.ledger; trend_tracker=bundle.trend_tracker; lr=bundle.current_learning_rate; retries=bundle.total_retries; quarantines=[CatalogQuarantineEvent.model_validate(x) for x in bundle.quarantine_events]; _set_optimizer_lr(optimizer,lr)
    else: lr=config.learning_rate; state=MCoreTrainingState(global_step=0,consumed_microbatches=0,learning_rate=lr,global_seed=config.global_seed,data_state=None)
    start=state.global_step
    if start>=config.max_steps: raise ValueError("resume checkpoint is already at or beyond max_steps")
    dp_rank=parallel_state.get_data_parallel_rank(with_context_parallel=False); dp_size=parallel_state.get_data_parallel_world_size(with_context_parallel=False)
    if cursor and cursor.data_parallel_size!=dp_size: raise ValueError("resume checkpoint data parallel size mismatch")
    root=Path(checkpoint_root); root.mkdir(parents=True,exist_ok=True); manager=RecoveryCheckpointManager(root,AsyncCheckpointConfig(enabled=True,strategy=config.async_checkpoint_strategy,max_unfinalized=config.max_unfinalized_checkpoints),config.keep_recovery_slots) if config.async_recovery_checkpoints else None; final=str(resume_checkpoint) if resumed else None
    device=torch.device("cuda",runtime.local_rank)
    if final is None:
        state,final=_sync_save(model,optimizer,state,root/"recovery-base-step-00000000",catalog.catalog_sha256,lr,ledger,trend_tracker,retries,quarantines,device=device)
        if manager: manager.register_committed(state,checkpoint_dir=final,kind="base",durable=True)
    offset=cursor.global_sequence_offset if cursor else 0; completed=start; metrics=[]
    try:
        while completed<config.max_steps:
            window=config.microbatches_per_step*dp_size; fp=fingerprint_batch(catalog_sha256=catalog.catalog_sha256,global_sequence_offset=offset,global_window=window)
            if fp in ledger.quarantined:
                proposed_offset=offset+window; proposed_cursor=FoundationCatalogCursor(global_sequence_offset=proposed_offset,catalog_sha256=catalog.catalog_sha256,data_parallel_size=dp_size); proposed_state=state.model_copy(update={"data_state":proposed_cursor.model_dump(mode="json")}); saved,committed=_sync_save(model,optimizer,proposed_state,root/f"recovery-skip-{proposed_offset:016d}",catalog.catalog_sha256,lr,ledger,trend_tracker,retries,quarantines,device=device);
                if manager: manager.register_committed(saved,checkpoint_dir=committed,kind="skip",durable=True)
                offset=proposed_offset; cursor=proposed_cursor; state=saved; final=committed; continue
            attempts=0
            while True:
                indices=[i for i in range(offset,offset+window) if (i-offset)%dp_size==dp_rank]; sequences=[x.sequence for x in FoundationWorkerPrefetch(catalog_path,indices,workers=config.prefetch_workers,prefetch_depth=config.prefetch_depth)]
                if len(sequences)!=config.microbatches_per_step: raise RuntimeError("catalog worker prefetch returned wrong local batch count")
                optimizer.zero_grad(set_to_none=True); model.zero_grad_buffer(); router.reset(); activation.reset(); clear_native_moe_metrics(); torch.cuda.reset_peak_memory_stats(runtime.local_rank); buffer=FoundationDoubleBuffer(sequences,device=device,depth=2); torch.distributed.barrier(); torch.cuda.synchronize(runtime.local_rank); started=time.perf_counter(); blockers=[]; stability=trend=None; lm_loss=None; grad_norm=0.; local_exception=None
                try:
                    lm_loss=_run_prefetched_batches(ddp,spec,buffer); model.finish_grad_sync(); native=read_native_moe_losses(); ok,grad_norm=_optimizer_step_result(optimizer.step());
                    if not ok:blockers.append("optimizer_step_failed")
                    torch.cuda.synchronize(runtime.local_rank); elapsed=max(time.perf_counter()-started,1e-9); rs=router.snapshot(); act=activation.snapshot(); local_tokens=catalog.sequence_length*config.microbatches_per_step; global_tokens=local_tokens*dp_size; stability=collect_mcore_stability_telemetry(model=model,global_step=completed+1,local_tokens=local_tokens,global_tokens=global_tokens,elapsed_seconds=elapsed,grad_norm=grad_norm,thresholds=config.stability_thresholds,local_rank=runtime.local_rank,aux_loss=native.aux_loss,z_loss=native.z_loss,router_layers=[(x.module_name,x.tokens_per_expert) for x in rs.layers]); trend=trend_tracker.preview(global_step=completed+1,lm_loss=lm_loss,grad_norm=grad_norm,max_activation_rms=act.max_rms,max_activation_abs=act.max_abs,worst_router_utilization_fraction=stability.worst_router_utilization_fraction); blockers+=stability.blockers+trend.blockers
                except Exception as exc: local_exception=exc; blockers.append(f"rank_exception:{type(exc).__name__}")
                unstable,reasons=distributed_recovery_consensus(blockers,device=device)
                if unstable:
                    decision=build_recovery_decision(globally_unstable=True,batch_fingerprint=fp,current_learning_rate=lr,policy=config.recovery_policy,ledger=ledger,reason_codes=reasons)
                    if manager:
                        manager.poll(); final=manager.committed_last_good_dir or final
                    restored=_restore_last_good(model,optimizer,final,config.global_seed)
                    if restored is None: raise RuntimeError("distributed recovery required but no committed last-good checkpoint exists") from local_exception
                    state=restored; lr=decision.next_learning_rate; _set_optimizer_lr(optimizer,lr)
                    if decision.retry_allowed: attempts+=1; retries=ledger.total_retries; continue
                    if decision.quarantine_batch and decision.skip_batch:
                        event=CatalogQuarantineEvent(batch_fingerprint=fp,global_sequence_offset=offset,global_window=window,failure_count=decision.failure_count,skipped=True,reason_codes=reasons); proposed_quarantines=[*quarantines,event]; proposed_offset=offset+window; proposed_cursor=FoundationCatalogCursor(global_sequence_offset=proposed_offset,catalog_sha256=catalog.catalog_sha256,data_parallel_size=dp_size); proposed_state=state.model_copy(update={"data_state":proposed_cursor.model_dump(mode="json"),"learning_rate":lr}); saved,committed=_sync_save(model,optimizer,proposed_state,root/f"recovery-quarantine-{proposed_offset:016d}",catalog.catalog_sha256,lr,ledger,trend_tracker,retries,proposed_quarantines,device=device);
                        if manager: manager.register_committed(saved,checkpoint_dir=committed,kind="quarantine",durable=True)
                        quarantines=proposed_quarantines; offset=proposed_offset; cursor=proposed_cursor; state=saved; final=committed; break
                    raise RuntimeError(f"distributed recovery exhausted for batch {fp}: {reasons}") from local_exception
                if stability is None or trend is None: raise RuntimeError("stable consensus reached without local telemetry")
                trend_tracker.commit_values(trend,grad_norm=grad_norm); elapsed=max(time.perf_counter()-started,1e-9); local_tokens=catalog.sequence_length*config.microbatches_per_step; global_tokens=local_tokens*dp_size; offset+=window; completed+=1; cursor=FoundationCatalogCursor(global_sequence_offset=offset,catalog_sha256=catalog.catalog_sha256,data_parallel_size=dp_size); metrics.append(CatalogTrainingStepMetrics(global_step=completed,local_tokens=local_tokens,global_tokens=global_tokens,elapsed_seconds=elapsed,local_tokens_per_second=local_tokens/elapsed,global_tokens_per_second=global_tokens/elapsed,lm_loss=lm_loss,stability=stability,trend=trend,recovery_attempts_before_success=attempts,effective_learning_rate=lr)); state=MCoreTrainingState(global_step=completed,consumed_microbatches=state.consumed_microbatches+config.microbatches_per_step,learning_rate=lr,global_seed=config.global_seed,data_state=cursor.model_dump(mode="json")); regular=completed%config.checkpoint_every_steps==0 or completed==config.max_steps; path=root/(f"step-{completed:08d}" if regular else f"recovery-step-{completed:08d}"); state=_attach(state,catalog.catalog_sha256,lr,ledger,trend_tracker,retries,quarantines)
                if manager:
                    manager.save_async(model,optimizer,state,checkpoint_dir=path,kind="periodic" if regular else "recovery",durable=regular); manager.poll(); final=manager.committed_last_good_dir or final
                else: save_mcore_training_checkpoint(model,optimizer,state,checkpoint_dir=path); final=path.as_posix()
                break
        if manager:
            manager.wait(); final=manager.committed_last_good_dir or final
            if final is None: raise RuntimeError("async recovery manager completed without a committed checkpoint")
    finally:
        if manager: manager.close(abort=False)
        activation.detach(); router.detach()
    return CatalogTrainingResult(start_step=start,final_step=config.max_steps,resumed_from_checkpoint=resumed,catalog_sha256=catalog.catalog_sha256,step_metrics=metrics,quarantine_events=quarantines,total_retries=retries,final_learning_rate=lr,final_checkpoint_dir=final)