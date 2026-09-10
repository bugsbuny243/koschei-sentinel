# Lang + Sentinel commercial bundle v1

Every package must include one `koschei-lang` and one `koschei-sentinel` component.
This is an offline artifact-integrity contract. Existing publisher-signature,
licensing, Lang release-trust and Sentinel model-promotion checks still apply.

The validator uses only the Python standard library. The implementation is mirrored
in both products to avoid adding an inter-repository runtime or package dependency.
Compatibility changes must update both validators and their contract tests together.

## Input contract

The root has exactly `schema_version`, `bundle_id`, `bundle_version`, `components`.
The schema is `koschei.commercial-bundle.v1`. Bundle IDs are lowercase ASCII
identifiers (letters, digits, dot, underscore, hyphen), at most 96 characters.
Versions use `major.minor.patch`, optionally followed by a prerelease label.

Each of the two component records has exactly:

| Field | Meaning |
| --- | --- |
| `product` | `koschei-lang` or `koschei-sentinel`, each exactly once |
| `version` | Component version |
| `artifact` | Local artifact basename, with no path or URL |
| `sha256` | Lowercase SHA-256 digest of the exact artifact bytes |

Artifact names must be distinct even under case folding. Duplicate JSON keys,
unknown fields, oversized manifests, missing components, paths, symlinks, special
files and digest mismatches are rejected. No example production model artifact is
invented; tests create explicitly synthetic temporary files.

## Result

- Metadata-only check: `status=manifest_valid`, no verified artifacts.
- Both local digests checked: `status=integrity_verified`.
- `release_approved=false` and `execution_authorized=false` in both cases.
- Exit 0 means the requested integrity/metadata check passed. Exit 2 means rejected.
- The canonical manifest digest binds both products and versions, independently
  of component ordering.

A digest comparison is not a signature or trusted publisher attestation. These
checks never download, unpack, install, execute, train or promote anything.
