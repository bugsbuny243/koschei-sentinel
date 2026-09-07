# Qwen3.5-397B-A17B Megatron-SWIFT external bootstrap runbook

This runbook prepares and launches the only active **external bootstrap/trainer-proof** large-MoE training target currently checked into the repository. It does not define Koschei Sentinel's product identity or native architecture target. The Sentinel-native target is approximately 397B total / 35B active parameters and is defined in `configs/model/sentinel-moe-397b-a35b.target.json`.

## 1. Build or mount the corpus

For a deterministic smoke-only source:

```bash
sentinel-cyber-seed-curriculum --output-root build/cyber-training
```

For a promotion-eligible run, mount a valid Gold release and use
`configs/training/cyber-sft.qwen3.5-397b-a17b.gold.example.json` instead. Promotion-eligible
TRAIN/VALIDATION inputs are accepted only when the sibling Gold release, including HOLDOUT,
passes the repository's Gold audit.

## 2. Render and seal the ms-swift dataset

```bash
sentinel-cyber-megatron-sft \
  --config configs/training/cyber-sft.qwen3.5-397b-a17b.megatron.json \
  --materialize-dataset \
  --plan-output build/cyber-training/megatron/397b-plan.json
```

The renderer writes `train.jsonl`, `validation.jsonl`, and `manifest.json`. The manifest binds the
source corpus hashes, split seed, model ID, model revision, row counts, both rendered JSONL hashes,
the Gold audit digest when applicable, and the largest observed tokenized sequence. Every example
is rendered with the pinned model tokenizer before materialization; an overlength example blocks
the run. Planning fails when any bound byte changes.

## 3. Validate the candidate cluster

The checked-in candidate topology is 4 nodes × 8 GPUs with TP=4, PP=1, CP=1, EP=8, and DP=8.
EP must divide DP; the config validator rejects incoherent partitions. Before spending budget,
verify at minimum:

- every node exposes eight distinct GPUs with at least the configured 80 GiB each;
- the full model snapshot and checkpoints fit shared storage;
- all ranks can read the sealed dataset and write the shared output directory;
- NCCL all-reduce and all-to-all tests pass across nodes;
- checkpoint save and restore throughput meets the recovery objective; and
- the approved cost cap covers the measured wall-time estimate.

Do not infer feasibility from the 17B active-parameter count. The checkpoint contains 397B total
parameters. These figures describe the external bootstrap model, not the Sentinel-native 35B-active architecture target.

## 4. Review the generated command

Without `--execute`, the CLI only prints a plan. The plan includes config, source, manifest,
rendered-dataset, and run-identity digests. Confirm that it contains the exact model and revision
plus:

```text
--tuner_type lora
--language_model_only true
--freeze_vit true
--freeze_aligner true
--moe_aux_loss_coeff 1e-06
--add_version true
--no_save_optim false
--no_save_rng false
```

## 5. Launch on every node

The scheduler must provide a consistent distributed environment. Generate one unique launch
session value per scheduler job and export the same value on every node. For the checked-in
topology:

```bash
export NNODES=4
export NPROC_PER_NODE=8
export NODE_RANK=<0..3>
export MASTER_ADDR=<rank-0-host>
export MASTER_PORT=<approved-port>
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export KOSCHEI_397B_LAUNCH_APPROVED=sentinel-cyber-sft-qwen3p5-397b-a17b-001
export KOSCHEI_397B_LAUNCH_SESSION=<unique-16-to-128-character-job-id>

sentinel-cyber-megatron-sft \
  --config configs/training/cyber-sft.qwen3.5-397b-a17b.megatron.json \
  --execute
```

The approval value must exactly equal the config `run_id`. Node 0 exclusively creates and binds
the output directory to the run/config/dataset digests. Other nodes wait for that exact identity
and launch session instead of racing to create the directory. After the final plan, every node
publishes a readiness receipt bound to both the plan digest and a canonical hash of
`MASTER_ADDR:MASTER_PORT`; node 0 opens the barrier only after all configured nodes match. The
shared launch state carries the same rendezvous hash, so late or resumed nodes cannot accept a
barrier created for another endpoint. A missing, stale, or mismatched value blocks Megatron before
training starts.

## 6. Resume a failed run

Optimizer and RNG state are retained. Resume requires both MCore paths and the identity manifest
written into the existing output directory. All checkpoint paths must remain inside that bound
output directory.

```bash
sentinel-cyber-megatron-sft \
  --config configs/training/cyber-sft.qwen3.5-397b-a17b.megatron.json \
  --resume-mcore-model <output-dir>/<checkpoint>/model \
  --resume-mcore-adapter <output-dir>/<checkpoint>/adapter \
  --resume-binding-manifest <output-dir>/koschei-run-identity.json \
  --execute
```

Resume is rejected if the self-hashed identity does not match the current run ID, config digest,
dataset manifest, source digests, and rendered train/validation digests. An accepted resume sets
`--finetune false`, `--no_load_optim false`, and `--no_load_rng false` so iteration, optimizer,
RNG, and dataset position are restored rather than silently restarting.

## 7. Promotion

No checkpoint is promoted from loss alone. First verify the training artifact, then run the Gold
HOLDOUT and Cyber Range gates defined by the repository. The model remains advisory behind
deterministic Sentinel authority.

## Upstream references

- [Qwen3.5-397B-A17B model card](https://huggingface.co/Qwen/Qwen3.5-397B-A17B)
- [Qwen3.5 Megatron-SWIFT best practices](https://swift.readthedocs.io/en/latest/BestPractices/Qwen3_8-Best-Practice.html)
- [Megatron-SWIFT command-line parameters](https://swift.readthedocs.io/en/latest/Megatron-SWIFT/Command-line-parameters.html)
