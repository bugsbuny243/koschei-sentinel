# Architecture

```text
Koschei Web3 Hub / ARVIS
          |
          | signed verdict + bounded evidence packet
          v
Dataset boundary -> anonymizer -> versioned case store
          |                              |
          |                              v
          |                       training / eval splits
          v                              |
Sentinel inference API <-----------------+
          |
          v
policy validator -> accepted commentary or hard rejection
```

## Trust boundary

ARVIS owns facts, rules, grades, signatures, and deterministic evidence. Sentinel owns only commentary. The inference service receives a bounded packet and returns a versioned opinion. The policy validator rejects unknown citations, signature mismatches, and attempts to produce a replacement assessment.

## Deployment shape

The intended production deployment is a separate service from Koschei Web3 Hub. Web3 Hub calls Sentinel over a private network, validates the response again, and falls back to the signed deterministic result when Sentinel is unavailable.
