from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RUFF_TARGETS = [
    "src/koschei_sentinel/blockchain_security_corpus.py",
    "src/koschei_sentinel/blockchain_security_corpus_cli.py",
    "src/koschei_sentinel/blockchain_security_ingest.py",
    "src/koschei_sentinel/blockchain_security_ingest_cli.py",
    "src/koschei_sentinel/blockchain_source_catalog.py",
    "src/koschei_sentinel/blockchain_source_catalog_cli.py",
    "src/koschei_sentinel/blockchain_security_release.py",
    "src/koschei_sentinel/blockchain_security_release_cli.py",
    "src/koschei_sentinel/blockchain_training.py",
    "src/koschei_sentinel/blockchain_training_cli.py",
    "src/koschei_sentinel/blockchain_security_eval.py",
    "src/koschei_sentinel/blockchain_security_eval_cli.py",
    "src/koschei_sentinel/blockchain_model_tournament.py",
    "src/koschei_sentinel/blockchain_model_tournament_cli.py",
    "src/koschei_sentinel/blockchain_base_candidates.py",
    "src/koschei_sentinel/blockchain_base_candidates_cli.py",
    "src/koschei_sentinel/blockchain_runtime_preflight.py",
    "src/koschei_sentinel/blockchain_runtime_preflight_cli.py",
    "scripts/run_blockchain_t4_preflight.py",
    "tests/test_blockchain_security_corpus.py",
    "tests/test_blockchain_security_ingest.py",
    "tests/test_blockchain_source_catalog.py",
    "tests/test_blockchain_security_release.py",
    "tests/test_blockchain_training.py",
    "tests/test_blockchain_security_eval.py",
    "tests/test_blockchain_model_tournament.py",
    "tests/test_blockchain_base_candidates.py",
    "tests/test_blockchain_runtime_preflight.py",
    "tests/test_blockchain_t4_candidate.py",
]

PYTEST_TARGETS = [
    "tests/test_blockchain_security_corpus.py",
    "tests/test_blockchain_security_ingest.py",
    "tests/test_blockchain_source_catalog.py",
    "tests/test_blockchain_security_release.py",
    "tests/test_blockchain_training.py",
    "tests/test_blockchain_security_eval.py",
    "tests/test_blockchain_model_tournament.py",
    "tests/test_blockchain_base_candidates.py",
    "tests/test_blockchain_runtime_preflight.py",
    "tests/test_blockchain_t4_candidate.py",
]


def run(command: list[str]) -> None:
    print("$", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> int:
    run(["ruff", "check", *RUFF_TARGETS])
    run([sys.executable, "-m", "pytest", "-q", *PYTEST_TARGETS])
    print("BLOCKCHAIN T4 FEASIBILITY STACK LOCAL GATE: PASSED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
