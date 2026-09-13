from __future__ import annotations

import argparse
import json

from koschei_sentinel.production_megatron_gate import verify_production_megatron_gate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify the fail-closed 397B/35B production Megatron training gate"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--binding", required=True)
    parser.add_argument("--architecture-manifest", required=True)
    parser.add_argument("--router-manifest", required=True)
    parser.add_argument("--expert-topology-manifest", required=True)
    parser.add_argument("--provenance", required=True)
    parser.add_argument("--owner-public-key", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config, binding, verification, provenance = verify_production_megatron_gate(
            config_path=args.config,
            binding_path=args.binding,
            architecture_manifest_path=args.architecture_manifest,
            router_manifest_path=args.router_manifest,
            expert_topology_manifest_path=args.expert_topology_manifest,
            provenance_path=args.provenance,
            owner_public_key_path=args.owner_public_key,
        )
        payload = {
            "schema_version": "sentinel.production-megatron-gate-result.v1",
            "ready": True,
            "run_id": config.run_id,
            "lane": config.lane,
            "model_ref": binding.model_ref,
            "model_revision": binding.model_revision,
            "total_parameters": binding.total_parameters,
            "active_parameters": binding.active_parameters,
            "production_training_allowed": binding.production_training_allowed,
            "topology_verification_sha256": verification.verification_sha256,
            "provenance_sha256": provenance.provenance_sha256,
            "owner_key_fingerprint": provenance.owner_key_fingerprint,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    except (OSError, TypeError, ValueError) as exc:
        print(f"sentinel-production-megatron-gate: {exc}")
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
