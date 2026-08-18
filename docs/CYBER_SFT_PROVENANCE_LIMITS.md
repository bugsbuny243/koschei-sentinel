# Cyber SFT Provenance — Current Guarantees and Limits

Koschei Sentinel Cyber SFT now records a pre-training source binding and re-validates that binding during completed-run reuse, run attestation, and portable export verification.

## What the current software chain proves

Under the repository launcher workflow, the source binding records:

- repository commit,
- training config digest,
- training plan digest,
- exact model/revision,
- corpus examples digest, and
- corpus manifest digest.

The launchers create this binding before invoking the real training executor. A completed run can be reused only when its verified adapter/receipt/runtime identity still matches the same source binding and current repository/config/plan identity.

Run attestation hashes the source-binding file, and portable export verification independently rebuilds and validates its semantics.

This provides deterministic software provenance, protects against accidental stale-run reuse, and prevents the normal launcher path from silently relabeling an old adapter as a new repository/config/corpus run.

## What it does not yet prove

The current source binding is not hardware-backed remote attestation.

It does not by itself prove to an external verifier that:

- the source-binding file physically existed before the first GPU instruction,
- the host kernel/hypervisor was uncompromised,
- the Python process was running only the attested repository code,
- the GPU execution environment was measured by a trusted hardware root, or
- an operator with full filesystem control could not fabricate an internally consistent software evidence set outside the approved launcher workflow.

Therefore the current chain must not be described as TPM/HSM/TEE-grade provenance.

## Planned Trust Plane upgrade

A production-grade Trust Plane should eventually bind training authorization to a signed lease containing at least:

- repository/tree digest,
- training config and plan digests,
- dataset/corpus digests,
- exact base model revision,
- authorized accelerator/runtime identity,
- training-source digest,
- execution start nonce and expiry,
- operator/service identity, and
- policy revision.

The training executor should consume that lease before optimizer execution and include the signed lease digest in the immutable training receipt. Where available, hardware-backed confidential-compute or accelerator attestation can further bind the receipt to the measured execution environment.

Until that Trust Plane exists, the current source-binding/reuse/attestation chain should be treated as strong deterministic software provenance and quota-safety infrastructure, not as hardware-rooted proof of execution.
