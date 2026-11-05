# Checkpoint metadata deserialization boundary

`verify_rank_checkpoint_load` inspects `sentinel.rank-weight-checkpoint.v1`
metadata using PyTorch's restricted `torch.load(weights_only=True)` path.
The verifier maps tensors to the `meta` device and validates the expected
schema, manifest digest, rank identity, tensor names, shapes and dtypes.

## Threat model

A checkpoint is serialized input, not trusted program code. The verifier
must not fall back to unrestricted Python object loading when the restricted
loader rejects a checkpoint. Unsupported Python object types fail closed.

`weights_only=True` substantially reduces Python unpickling risk, but does
not make arbitrary checkpoint data safe against denial of service, malformed
tensor edge cases or all implementation-level vulnerabilities. Mapping tensors
to `meta` does not authenticate the checkpoint.

The manifest digest checked here binds expected metadata, **not** an
independently authenticated source or the complete serialized file bytes.
Production consumers must separately establish source provenance and verify
the approved artifact digest or signature before deserialization, within
their own trusted intake boundary.

## Coverage and limitations

The PyTorch CPU regression suite tests a valid tensor state dictionary,
manifest mismatch rejection and rejection of a disallowed Python object.
The result's `metadata_verified` status is not evidence that a checkpoint
was trained, that forward/backward passes succeeded, or that an inference
model can be promoted or deployed.
