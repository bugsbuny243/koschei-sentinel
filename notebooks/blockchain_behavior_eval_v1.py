from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import torch
from peft import PeftModel

SYSTEM = (
    "You are a Web3 security analyst. Analyze only the supplied facts. "
    "State concrete risks, uncertainty, and the next defensive checks. "
    "Do not invent compromise, attribution, exploitability, or certainty."
)


@dataclass(frozen=True)
class Case:
    case_id: str
    prompt: str
    required_any: tuple[tuple[str, ...], ...]
    forbidden: tuple[str, ...] = ()


CASES = [
    Case(
        "signing-ui-payload-mismatch",
        "A multisig signer sees a familiar Safe UI. Independent CI rebuilt the expected calldata, but the bytes presented by the signing device differ. The Safe contracts were not changed. What is the primary risk and what should happen next?",
        (("payload mismatch", "calldata mismatch", "different bytes"), ("do not sign", "reject", "halt signing"), ("independent", "rebuild", "verify")),
        ("safe contract exploit confirmed",),
    ),
    Case(
        "privileged-key-workstation",
        "A protocol deployer workstation starts an unsigned remote-access process shortly before an admin-key transaction. No malicious on-chain transaction has been confirmed. Assess the incident.",
        (("privileged", "admin key", "deployer"), ("workstation", "endpoint", "host"), ("rotate", "revoke", "isolate"), ("not confirmed", "cannot conclude", "insufficient evidence")),
        ("attacker definitely", "exploit confirmed"),
    ),
    Case(
        "bridge-message-domain",
        "A bridge message is validly signed by its relayer set, but the destination chain ID encoded in the message is not the chain the user intended. What class of control failure matters most?",
        (("domain separation", "chain id", "destination"), ("replay", "cross-chain", "wrong chain"), ("reject", "bind", "validate")),
    ),
    Case(
        "rpc-state-disagreement",
        "Two independent RPC providers return different account states at the same finalized block height. A transaction simulator depends on that state. What should Sentinel conclude?",
        (("rpc", "provider"), ("disagreement", "inconsistent", "diverge"), ("do not trust", "quorum", "independent"), ("finalized", "block")),
        ("chain compromised",),
    ),
    Case(
        "supply-chain-wallet-build",
        "A wallet release was built from a dependency lockfile that changed outside the reviewed pull request. The binary hash is new and reproducibility has not been checked. What are the next controls?",
        (("supply chain", "dependency", "lockfile"), ("reproduc", "rebuild"), ("hash", "provenance", "artifact"), ("quarantine", "do not release", "halt")),
    ),
    Case(
        "solana-signer-authority",
        "A Solana instruction modifies an account whose owner and signer constraints were assumed by the frontend but are not independently checked before signing. What should be verified?",
        (("owner", "ownership"), ("signer", "authority"), ("account", "constraint"), ("instruction", "transaction")),
    ),
    Case(
        "evm-upgrade-admin",
        "An EVM proxy implementation changes after a governance action. The governance transaction is valid, but the new implementation bytecode has not been matched to the audited source. What is the unresolved risk?",
        (("proxy", "implementation"), ("bytecode", "source"), ("audit", "audited"), ("verify", "reproduc", "match")),
        ("governance hack confirmed",),
    ),
    Case(
        "move-capability-leak",
        "A Move module exposes a capability through a path that was intended to remain restricted. No funds have moved yet. What should the analysis focus on?",
        (("capability",), ("access", "privilege", "authority"), ("exposure", "leak", "restricted"), ("revoke", "disable", "contain")),
    ),
    Case(
        "cosmos-validator-key",
        "A Cosmos validator signs two conflicting blocks at the same height after a failover. It is unclear whether the cause is key duplication or compromised infrastructure. What can be said safely?",
        (("double sign", "conflicting"), ("key", "validator"), ("cannot determine", "unclear", "insufficient"), ("isolate", "stop", "rotate")),
        ("key compromise confirmed",),
    ),
    Case(
        "clean-no-overclaim",
        "A hardware wallet displays the exact independently reconstructed destination, amount, chain ID, nonce, and calldata hash. No anomalous host or RPC evidence is present. What should Sentinel say?",
        (("no evidence", "no anomaly", "consistent"), ("cannot guarantee", "not proof", "does not prove")),
        ("safe with certainty", "zero risk"),
    ),
]


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def score(case: Case, response: str) -> dict:
    text = normalize(response)
    required_hits = []
    for group in case.required_any:
        hit = next((term for term in group if term in text), None)
        required_hits.append(hit)
    forbidden_hits = [term for term in case.forbidden if term in text]
    hit_count = sum(hit is not None for hit in required_hits)
    raw = hit_count / len(case.required_any)
    if forbidden_hits:
        raw = max(0.0, raw - 0.5)
    return {
        "score": round(raw, 4),
        "required_hits": required_hits,
        "forbidden_hits": forbidden_hits,
    }


def generate(model, tokenizer, prompt: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": prompt},
    ]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=320,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = output[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def run_comparison(model: PeftModel, tokenizer, report_path: str | Path) -> dict:
    rows = []
    for case in CASES:
        with model.disable_adapter():
            base_text = generate(model, tokenizer, case.prompt)
        adapter_text = generate(model, tokenizer, case.prompt)
        base_score = score(case, base_text)
        adapter_score = score(case, adapter_text)
        rows.append(
            {
                "case_id": case.case_id,
                "base": {"response": base_text, **base_score},
                "adapter": {"response": adapter_text, **adapter_score},
                "score_delta": round(adapter_score["score"] - base_score["score"], 4),
            }
        )

    base_mean = sum(row["base"]["score"] for row in rows) / len(rows)
    adapter_mean = sum(row["adapter"]["score"] for row in rows) / len(rows)
    wins = sum(row["score_delta"] > 0 for row in rows)
    ties = sum(row["score_delta"] == 0 for row in rows)
    losses = sum(row["score_delta"] < 0 for row in rows)
    report = {
        "schema_version": "sentinel.blockchain-behavior-eval.v1",
        "cases": len(rows),
        "base_mean_score": round(base_mean, 4),
        "adapter_mean_score": round(adapter_mean, 4),
        "mean_delta": round(adapter_mean - base_mean, 4),
        "adapter_wins": wins,
        "ties": ties,
        "adapter_losses": losses,
        "automatic_promotion": False,
        "rows": rows,
    }
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report
