from __future__ import annotations

import argparse
import json

from koschei_sentinel.blockchain_training import (
    BlockchainTrainingPlan,
    execute_blockchain_training,
    load_blockchain_training_config,
    plan_blockchain_training,
    write_blockchain_training_plan,
)
from koschei_sentinel.blockchain_training_authorization import (
    load_training_authorization_approval,
    load_training_authorization_policy,
    load_training_authorization_proposal,
    verify_training_authorization_bundle,
)
from koschei_sentinel.promotion import load_owner_public_key

_AUTHORIZATION_ARGUMENTS = (
    "authorization_proposal",
    "authorization_approval",
    "authorization_policy",
    "owner_public_key",
    "preflight_seal",
    "preflight_run",
    "registry",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plan or explicitly execute Koschei Sentinel blockchain-security "
            "continued pretraining"
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--root", default=".")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--authorization-proposal")
    parser.add_argument("--authorization-approval")
    parser.add_argument("--authorization-policy")
    parser.add_argument("--owner-public-key")
    parser.add_argument("--preflight-seal")
    parser.add_argument("--preflight-run")
    parser.add_argument("--registry")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_blockchain_training_config(args.config)
        if args.execute:
            missing = [name for name in _AUTHORIZATION_ARGUMENTS if not getattr(args, name)]
            if missing:
                raise ValueError(
                    "execution requires signed training authorization inputs: "
                    + ", ".join(sorted(missing))
                )
            try:
                plan = BlockchainTrainingPlan.model_validate_json(
                    open(args.plan, encoding="utf-8").read()
                )
            except (OSError, ValueError) as exc:
                raise ValueError("execution requires a valid existing blockchain plan") from exc

            proposal = load_training_authorization_proposal(args.authorization_proposal)
            approval = load_training_authorization_approval(args.authorization_approval)
            policy = load_training_authorization_policy(args.authorization_policy)
            verified_approval = verify_training_authorization_bundle(
                proposal=proposal,
                approval=approval,
                owner_public_key=load_owner_public_key(args.owner_public_key),
                policy=policy,
                seal_path=args.preflight_seal,
                preflight_run_dir=args.preflight_run,
                registry_path=args.registry,
                release_dir=config.blockchain_release,
                training_config=config,
                root=args.root,
            )
            manifest = execute_blockchain_training(config, plan, root=args.root)
            print(
                json.dumps(
                    {
                        "ok": True,
                        "executed": True,
                        "run_id": manifest.run_id,
                        "base_model": manifest.base_model,
                        "base_revision": manifest.base_revision,
                        "source_corpus_digest": manifest.source_corpus_digest,
                        "benchmark_suite_digest": manifest.benchmark_suite_digest,
                        "train_documents": manifest.train_documents,
                        "validation_documents": manifest.validation_documents,
                        "held_out_test_documents": manifest.held_out_test_documents,
                        "held_out_test_consumed": manifest.held_out_test_consumed,
                        "train_chunks": manifest.train_chunks,
                        "validation_chunks": manifest.validation_chunks,
                        "adapter_digest": manifest.adapter_digest,
                        "authorization_id": verified_approval.authorization_id,
                        "authorization_approval_digest": verified_approval.approval_digest,
                        "training_authorized": True,
                        "production_authority": False,
                        "output": manifest.output_dir,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0

        plan = plan_blockchain_training(config, root=args.root)
        write_blockchain_training_plan(plan, args.plan)
        print(
            json.dumps(
                {
                    "ok": True,
                    "executed": False,
                    "run_id": plan.run_id,
                    "base_model": plan.base_model,
                    "base_revision": plan.base_revision,
                    "source_corpus_digest": plan.source_corpus_digest,
                    "benchmark_suite_digest": plan.benchmark_suite_digest,
                    "train_documents": plan.train_documents,
                    "validation_documents": plan.validation_documents,
                    "held_out_test_documents": plan.held_out_test_documents,
                    "estimated_document_steps": plan.estimated_document_steps,
                    "plan": args.plan,
                    "training_started": False,
                    "training_authorized": False,
                    "production_authority": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except (FileExistsError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"sentinel-blockchain-train: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
