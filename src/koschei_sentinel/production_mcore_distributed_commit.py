from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TypeVar
T=TypeVar("T")
@dataclass(frozen=True)
class DistributedCommitResult:
    checkpoint_dir:str
    all_ranks_ready:bool

def distributed_all_ready(local_ready:bool,*,device:Any)->bool:
    try:
        import torch
        import torch.distributed as dist
    except ImportError as exc: raise RuntimeError("PyTorch distributed is required for durable commit consensus") from exc
    if not dist.is_initialized(): raise RuntimeError("torch.distributed must be initialized before durable commit consensus")
    flag=torch.tensor([1 if local_ready else 0],device=device,dtype=torch.int32); dist.all_reduce(flag,op=dist.ReduceOp.MIN); return bool(int(flag.item()))

def commit_checkpoint_dir(checkpoint_dir:str|Path,*,device:Any)->DistributedCommitResult:
    path=Path(checkpoint_dir).resolve(); local_ready=path.exists() and path.is_dir() and any(path.iterdir()); ready=distributed_all_ready(local_ready,device=device)
    if not ready: raise RuntimeError(f"distributed checkpoint commit rejected: {path}")
    return DistributedCommitResult(checkpoint_dir=path.as_posix(),all_ranks_ready=True)

def save_and_commit_state(save_fn:Callable[[],T],*,checkpoint_dir:str|Path,device:Any)->tuple[T,DistributedCommitResult]:
    """Write first, then require every rank to observe the checkpoint before logical commit."""
    saved=save_fn(); committed=commit_checkpoint_dir(checkpoint_dir,device=device); return saved,committed
