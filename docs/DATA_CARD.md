# Data Card — Draft

## Intended data

Anonymized, bounded ARVIS cases containing signed deterministic verdicts, evidence rows, confidence labels, limitations, and human-approved explanatory outputs.

## Excluded data

- plaintext API keys or credentials;
- private keys or signing material;
- names, emails, phone numbers, IP addresses, or account identifiers;
- unsupported allegations about real people;
- raw production database dumps;
- model-generated examples that were not independently reviewed.

## Split policy

Cases sharing the same creator, wallet cluster, token family, or incident lineage must stay in one split to prevent leakage. Challenge and red-team sets remain frozen and are never used for training.

## Retention

Exports must be reproducible, versioned, and revocable. Source records should remain under Koschei access controls; training artifacts contain only the minimum anonymized packet required for the task.
