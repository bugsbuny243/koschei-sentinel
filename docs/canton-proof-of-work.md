# Canton Security Assurance Proof-of-Work

This branch contains a deliberately small Canton-specific vertical slice for champion/reviewer evaluation.

## What it proves

A bounded public/test fixture can be transformed into `canton.security.evidence.v1`, assigned a stable SHA-256 evidence identifier, evaluated by a deterministic assurance rule, and returned as a machine-readable finding that cites the exact evidence ID.

The proof also demonstrates the fail-closed boundary: when required provenance, integrity, or artifact-version evidence is unavailable, the result is `insufficient_evidence` rather than an invented pass/fail conclusion.

## Current rule

`CANTON-EVIDENCE-COVERAGE-001` checks whether evidence declared as expected was observed. The rule is intentionally simple: the purpose of this slice is to prove the contract and deterministic evidence-to-finding path before selecting real Canton evidence sources with Tech & Ops/champion input.

## Reproduce

```bash
pytest -q tests/test_canton_assurance.py
```

The tests verify deterministic identical output, evidence-ID citation, an explicit failure for missing expected evidence, and an `insufficient_evidence` state when provenance is unavailable.

## Boundary

This is proof-of-work, not a claim of production Canton integration. It does not connect to a validator, use private production data, custody keys, sign transactions, or alter Canton state. Selection and implementation of supported Canton interfaces/adapters remains part of the proposed Development Fund work and should be reviewed with Canton Tech & Ops.
