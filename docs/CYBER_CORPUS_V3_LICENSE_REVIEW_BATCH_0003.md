# Cyber Corpus v3 License Review — Batch 0003

## Approved in this batch

### NIST Cybersecurity Framework (CSF) 2.0

- Source ID: `nist.csf.2.0`
- Canonical locator: `https://doi.org/10.6028/NIST.CSWP.29`
- Pin: `NIST.CSWP.29:2024-02-26:FINAL`
- Source class: `OFFICIAL_SECURITY_GUIDANCE`
- License scope: `PER_ARTIFACT`
- Decision: `ALLOW_WITH_ATTRIBUTION`

NIST Technical Series publications provide broad worldwide reprint and derivative-work permission for NIST-authored works. NIST also warns that some publications may contain third-party authored or separately protected material. For that reason this source is not treated as uniformly licensed: extraction must exclude or separately clear referenced, excerpted, or otherwise third-party material.

The CSF 2.0 publication itself states that cited, referenced, or excerpted documents are not wholly incorporated unless otherwise noted. The final publication is pinned by DOI and final publication date.

### NIST SP 800-61 Rev. 3

- Source ID: `nist.sp800-61r3`
- Canonical locator: `https://doi.org/10.6028/NIST.SP.800-61r3`
- Pin: `NIST.SP.800-61r3:2025-04-03:FINAL`
- Source class: `OFFICIAL_SECURITY_GUIDANCE`
- License scope: `PER_ARTIFACT`
- Decision: `ALLOW_WITH_ATTRIBUTION`

SP 800-61 Rev. 3 is the April 3, 2025 final revision and supersedes SP 800-61 Rev. 2. It is admitted under the same NIST Technical Series rights policy with artifact-level third-party-rights checks.

## Not blanket-approved

### OSV aggregate database

OSV is an aggregator. Its official data-source documentation lists multiple upstream licenses including CC-BY-4.0, CC0-1.0, MIT, Apache-2.0, BSD, and CC-BY-SA-4.0, and also includes converted sources such as NVD. Therefore `all.zip` is not treated as one uniformly licensed training source. Future OSV ingestion must preserve upstream source identity and resolve rights per source/artifact before training authorization.

### NVD

The NVD API Terms permit retrieval and use of NVD data subject to attribution/non-endorsement and modification rules, but the present Cyber v3 policy keeps NVD fail-closed because records can incorporate upstream CVE descriptions and provenance. NVD extraction remains context/review material until artifact-level lineage is explicit.

### CISA

No blanket CISA authorization is granted in this batch. CISA advisories and guidance remain artifact-by-artifact review candidates because individual publications can incorporate third-party material and sharing markings do not by themselves establish model-training rights.

## Boundary

Approval means a source may enter the collection pipeline under its recorded restrictions. It does not seal a corpus, waive artifact-level checks, authorize benchmark overlap, or authorize a training run.
