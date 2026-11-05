# Checkpoint metadata deserialization boundary

`verify_rank_checkpoint_load` inspects `sentinel.rank-weight-checkpoint.v1`
metadata using PyTorch's restricted `torch.load(weights_only=True)` path.
The verifier maps tensors to the `meta` device and validates the expected
schema, manifest digest, rank identity, tensor names, shapes and dtypes.

## Unauthenticated metadata inspection

The existing `verify_rank_checkpoint_load` call remains supported. Its result
means only **metadata matches the supplied initialization manifest**. It
does not attest to the source, integrity or training history of the file.
Optional `expected_file_sha256` and `expected_file_size_bytes` arguments
must be supplied together; the verifier hashes the opened file before
deserializing it, reads the same file descriptor and checks the hash again
after deserialization. These optional values alone do **not** establish
where an approval came from.

## Signed intake for approved model identities

For a model-identity-bound check, use
`production_checkpoint_trust.verify_approved_rank_checkpoint_load`.
Its required inputs are:

- the checkpoint and its initialization manifest;
- an `ApprovedRankCheckpointReceipt` signed by the target owner;
- a verified `ProductionTargetBinding` and owner-signed
  `ProductionTargetProvenance`;
- the owner's **independently trusted, pinned Ed25519 public key**.

The receipt binds model reference/revision, verified target binding,
source provenance digest, rank/manifest identity and checkpoint file
SHA-256 and size. The trust verifier checks target status, provenance,
the domain-separated receipt signature and identity matches **before**
opening the file for tensor inspection. The byte digest and size are
checked before restricted deserialization; the same file descriptor is
used throughout. Symlinks and non-regular checkpoint entries are rejected.

**Trust-root requirement:** The caller must obtain and authenticate the
owner public key out of band. Passing an attacker-controlled receipt,
binding, provenance and public key together does **not** create trusted
approval. A signed receipt is not proof of a successful training run,
independent model evaluation, permitted deployment or market release.

## Threat model and limitations

A checkpoint is serialized input, not trusted program code. Neither path
may fall back to unrestricted Python object loading when the restricted
loader rejects an unsupported object.

`weights_only=True` limits Python unpickling risk but does not make
malformed checkpoint data safe from all denial-of-service or library-level
bugs. Mapping tensors to `meta` does not authenticate the checkpoint.
SHA-256 protects against accidental/malicious changes **relative to a
trusted digest**; it is not a substitute for a pinned trust root or a
production rollout authorization. In-place concurrent modifications are
rechecked after inspection, but external filesystem isolation is still
required for a robust intake environment.

The standalone result's `metadata_verified` status and the signed-intake
result's `approved_metadata_verified` status both keep
`execution_authorized=False`.

## Regression coverage

The PyTorch CPU suite verifies a valid tensor state dictionary, rejects a
tampered manifest, blocks unallowlisted Python objects, validates a
properly signed model/receipt pair and rejects wrong trust roots, changed
artifact bytes, tampered model identity, invalid signatures, unverified
target bindings, symlinks and incomplete hash/size arguments.
This does not test GPU training, full-model inference or rollout safety.
