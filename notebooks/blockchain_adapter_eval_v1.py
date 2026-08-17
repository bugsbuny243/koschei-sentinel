# KOSCHEI SENTINEL — BLOCKCHAIN ADAPTER EVAL V1
#
# Kaggle-ready offline model comparison for koschei-sentinel-blockchain-run-001.
# This script does NOT train. It reconstructs the exact deterministic 5% held-out
# split used by the training notebook and compares Qwen2.5-Coder-7B base loss
# against the saved LoRA adapter on the same unseen documents.
#
# Output:
#   /kaggle/working/koschei-sentinel-blockchain-run-001/
#       blockchain-adapter-eval-v1.json

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
import subprocess
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

REQUIRED = {
    "transformers": "transformers>=4.45",
    "peft": "peft>=0.12",
    "accelerate": "accelerate>=0.34",
    "bitsandbytes": "bitsandbytes>=0.43",
    "gdown": "gdown",
    "safetensors": "safetensors",
}
missing = [pkg for module, pkg in REQUIRED.items() if importlib.util.find_spec(module) is None]
if missing:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=True)

import gdown
import torch
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "Qwen/Qwen2.5-Coder-7B"
MODEL_REVISION = "0396a76181e127dfc13e5c5ec48a8cee09938b02"
CORPUS_DRIVE_ID = "1eG_-CF_SNdaejkyf0KbGnhCfoAzrEOzZ"
EXPECTED_CORPUS_SHA256 = "2685a8d2d9b0effb16787f88610a8b57a42bf0eb267c29ffb5e588e73e5a1731"
ADAPTER_WEIGHTS_DRIVE_ID = "1GDPjeaSDIB_QwQ6Xdt5YX4wOJ0Zh22KN"
SEQ_LEN = 1024
MIN_TAIL_TOKENS = 32

WORK = Path("/kaggle/working")
RUN_DIR = WORK / "koschei-sentinel-blockchain-run-001"
RUN_DIR.mkdir(parents=True, exist_ok=True)
CORPUS = WORK / "batch-0001.approved-document-corpus-v2.jsonl"
ADAPTER_DIR = RUN_DIR / "final-adapter"
ADAPTER_WEIGHTS = ADAPTER_DIR / "adapter_model.safetensors"
REPORT = RUN_DIR / "blockchain-adapter-eval-v1.json"
HF_CACHE = WORK / "hf-cache"
HF_CACHE.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_file(path: Path, drive_id: str) -> None:
    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    result = gdown.download(id=drive_id, output=str(path), quiet=False)
    if not result or not path.is_file():
        raise RuntimeError(f"Drive artifact could not be downloaded: {path.name}")


def ensure_adapter_config() -> None:
    config_path = ADAPTER_DIR / "adapter_config.json"
    if config_path.is_file():
        return
    # Reconstruct exactly the LoRA topology used by the sealed training notebook.
    config = LoraConfig(
        task_type="CAUSAL_LM",
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    config.save_pretrained(ADAPTER_DIR)


def load_heldout_rows() -> list[dict]:
    rows: list[dict] = []
    with CORPUS.open("r", encoding="utf-8", newline="") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if int(row["content_sha256"][:8], 16) % 20 == 0:
                rows.append(row)
    if len(rows) != 55:
        raise RuntimeError(f"Expected 55 held-out documents, got {len(rows)}")
    return rows


def exact_packed_blocks(rows: list[dict], tokenizer) -> list[list[int]]:
    eos = tokenizer.eos_token_id
    if eos is None:
        raise RuntimeError("Tokenizer has no EOS token")
    buffer: list[int] = []
    blocks: list[list[int]] = []
    for row in rows:
        ids = tokenizer(
            row["text"].strip(),
            add_special_tokens=False,
            return_attention_mask=False,
        )["input_ids"]
        ids.append(eos)
        buffer.extend(ids)
        while len(buffer) >= SEQ_LEN:
            blocks.append(buffer[:SEQ_LEN])
            del buffer[:SEQ_LEN]
    if len(blocks) != 96:
        raise RuntimeError(f"Expected 96 exact eval blocks, got {len(blocks)}")
    return blocks


def document_chunks(row: dict, tokenizer) -> list[list[int]]:
    eos = tokenizer.eos_token_id
    ids = tokenizer(
        row["text"].strip(),
        add_special_tokens=False,
        return_attention_mask=False,
    )["input_ids"]
    ids.append(eos)
    chunks = [ids[i : i + SEQ_LEN] for i in range(0, len(ids), SEQ_LEN)]
    if chunks and len(chunks[-1]) < MIN_TAIL_TOKENS:
        chunks = chunks[:-1]
    return [chunk for chunk in chunks if len(chunk) >= 2]


@torch.inference_mode()
def nll_for_chunks(model, chunks: list[list[int]]) -> tuple[float, int]:
    total_nll = 0.0
    total_predicted_tokens = 0
    for ids in chunks:
        tensor = torch.tensor(ids, dtype=torch.long, device=model.device).unsqueeze(0)
        out = model(input_ids=tensor, labels=tensor, use_cache=False)
        predicted_tokens = tensor.shape[1] - 1
        total_nll += float(out.loss.detach().float().cpu()) * predicted_tokens
        total_predicted_tokens += predicted_tokens
    return total_nll, total_predicted_tokens


def safe_perplexity(loss: float) -> float:
    return math.exp(min(loss, 20.0))


def aggregate_document_metrics(rows: list[dict], tokenizer, model, adapter_enabled: bool) -> dict:
    total_nll = 0.0
    total_tokens = 0
    by_chain: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    by_domain: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])

    context = torch.enable_grad()  # placeholder replaced below
    if adapter_enabled:
        context = torch.inference_mode()
    else:
        context = model.disable_adapter()

    # nll_for_chunks already enters inference_mode. The outer context only toggles LoRA.
    with context:
        for row in rows:
            chunks = document_chunks(row, tokenizer)
            if not chunks:
                continue
            nll, tokens = nll_for_chunks(model, chunks)
            total_nll += nll
            total_tokens += tokens
            chain = row["chain_family"]
            by_chain[chain][0] += nll
            by_chain[chain][1] += tokens
            for domain in row.get("threat_domains", []):
                by_domain[domain][0] += nll
                by_domain[domain][1] += tokens

    overall_loss = total_nll / total_tokens
    return {
        "loss": overall_loss,
        "perplexity": safe_perplexity(overall_loss),
        "predicted_tokens": total_tokens,
        "by_chain": {
            key: {
                "loss": nll / tokens,
                "perplexity": safe_perplexity(nll / tokens),
                "predicted_tokens": int(tokens),
            }
            for key, (nll, tokens) in sorted(by_chain.items())
            if tokens
        },
        "by_threat_domain": {
            key: {
                "loss": nll / tokens,
                "perplexity": safe_perplexity(nll / tokens),
                "predicted_tokens": int(tokens),
            }
            for key, (nll, tokens) in sorted(by_domain.items())
            if tokens
        },
    }


def compare_breakdown(base: dict, adapter: dict, section: str) -> dict:
    keys = sorted(set(base[section]) | set(adapter[section]))
    result = {}
    for key in keys:
        if key not in base[section] or key not in adapter[section]:
            continue
        b = base[section][key]["loss"]
        a = adapter[section][key]["loss"]
        result[key] = {
            "base_loss": b,
            "adapter_loss": a,
            "absolute_loss_delta": a - b,
            "relative_loss_change_pct": ((a - b) / b) * 100.0,
            "adapter_improved": a < b,
        }
    return result


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required for this evaluation")

    print("KOSCHEI SENTINEL — BLOCKCHAIN ADAPTER EVAL V1")
    print("GPU:", torch.cuda.get_device_name(0))

    ensure_file(CORPUS, CORPUS_DRIVE_ID)
    corpus_sha = sha256_file(CORPUS)
    if corpus_sha != EXPECTED_CORPUS_SHA256:
        raise RuntimeError(
            "CORPUS SHA256 MISMATCH\n"
            f"expected={EXPECTED_CORPUS_SHA256}\nactual={corpus_sha}"
        )

    ensure_adapter_config()
    ensure_file(ADAPTER_WEIGHTS, ADAPTER_WEIGHTS_DRIVE_ID)
    adapter_sha = sha256_file(ADAPTER_WEIGHTS)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir=str(HF_CACHE),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    heldout = load_heldout_rows()
    exact_blocks = exact_packed_blocks(heldout, tokenizer)

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir=str(HF_CACHE),
        quantization_config=bnb,
        device_map={"": 0},
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    base.config.use_cache = False
    model = PeftModel.from_pretrained(base, ADAPTER_DIR, is_trainable=False)
    model.eval()

    print("Evaluating exact packed holdout with adapter disabled...")
    with model.disable_adapter():
        base_nll, base_tokens = nll_for_chunks(model, exact_blocks)
    base_exact_loss = base_nll / base_tokens

    print("Evaluating exact packed holdout with adapter enabled...")
    adapter_nll, adapter_tokens = nll_for_chunks(model, exact_blocks)
    adapter_exact_loss = adapter_nll / adapter_tokens

    if base_tokens != adapter_tokens:
        raise RuntimeError("Base/adapter token counts differ")

    print("Evaluating per-document breakdown...")
    base_docs = aggregate_document_metrics(heldout, tokenizer, model, adapter_enabled=False)
    adapter_docs = aggregate_document_metrics(heldout, tokenizer, model, adapter_enabled=True)

    chain_counts = Counter(row["chain_family"] for row in heldout)
    domain_counts = Counter(
        domain for row in heldout for domain in row.get("threat_domains", [])
    )

    report = {
        "schema_version": "sentinel.blockchain-adapter-eval.v1",
        "run": "koschei-sentinel-blockchain-run-001",
        "base_model": MODEL_ID,
        "base_revision": MODEL_REVISION,
        "corpus_sha256": corpus_sha,
        "adapter_weights_sha256": adapter_sha,
        "heldout_rule": "int(content_sha256[:8], 16) % 20 == 0",
        "heldout_documents": len(heldout),
        "exact_eval_blocks": len(exact_blocks),
        "sequence_length": SEQ_LEN,
        "coverage": {
            "chain_document_counts": dict(sorted(chain_counts.items())),
            "threat_domain_document_counts": dict(sorted(domain_counts.items())),
            "known_blind_spots": [
                chain for chain in ("BITCOIN", "TRON") if chain_counts.get(chain, 0) == 0
            ],
        },
        "exact_packed": {
            "predicted_tokens": base_tokens,
            "base_loss": base_exact_loss,
            "base_perplexity": safe_perplexity(base_exact_loss),
            "adapter_loss": adapter_exact_loss,
            "adapter_perplexity": safe_perplexity(adapter_exact_loss),
            "absolute_loss_delta": adapter_exact_loss - base_exact_loss,
            "relative_loss_change_pct": ((adapter_exact_loss - base_exact_loss) / base_exact_loss) * 100.0,
            "adapter_improved": adapter_exact_loss < base_exact_loss,
        },
        "document_level": {
            "base": base_docs,
            "adapter": adapter_docs,
            "chain_comparison": compare_breakdown(base_docs, adapter_docs, "by_chain"),
            "threat_domain_comparison": compare_breakdown(
                base_docs, adapter_docs, "by_threat_domain"
            ),
        },
        "promotion": {
            "automatic_promotion": False,
            "reason": (
                "This report measures held-out domain-modeling gain only. "
                "The strict Sentinel authority/grounding/abstention/privacy gate must run separately."
            ),
        },
    }

    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("\n========================================")
    print("HELD-OUT COMPARISON COMPLETE")
    print("========================================")
    print(f"base_loss:    {base_exact_loss:.6f}")
    print(f"adapter_loss: {adapter_exact_loss:.6f}")
    print(
        "relative_change_pct:",
        f"{report['exact_packed']['relative_loss_change_pct']:.3f}%",
    )
    print("adapter_improved:", report["exact_packed"]["adapter_improved"])
    print("report:", REPORT)
    print("========================================")


if __name__ == "__main__":
    main()
