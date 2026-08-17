# Cyber Corpus v3 — rights review ledger

This ledger records source-family research. It is not legal advice and it does not by itself grant training authorization. Artifact-level provenance and rights must still be recorded in the source catalog.

## NIST

Official NIST copyright/licensing guidance states that works created by NIST employees generally are not subject to U.S. copyright protection under 17 U.S.C. §105, and NIST grants broad worldwide reuse/derivative rights for many NIST works while requesting acknowledgement. NIST also warns that Standard Reference Data can be separately copyrighted/licensed and that third-party or contractor-authored material can carry different rights.

References:
- https://www.nist.gov/open/copyright-fair-use-and-licensing-statements-srd-data-software-and-technical-series-publications
- https://www.nist.gov/copyrights-disclaimers

Corpus decision: **do not blanket-approve `nist.security.guidance`**. Approve individual NIST-authored technical-series artifacts only after confirming the artifact does not contain separately copyrighted third-party material or SRD restrictions. Record the DOI/revision and the applicable NIST rights statement.

## MITRE ATT&CK

MITRE's official ATT&CK Terms of Use grant a non-exclusive, royalty-free license to use ATT&CK for research, development, and commercial purposes, with a requirement to reproduce MITRE's copyright designation and license in copies.

Reference:
- https://attack.mitre.org/resources/terms-of-use/

Corpus decision: retain `REVIEW_REQUIRED` until the project records how the required copyright/license notice will be preserved in dataset lineage and distributed artifacts. Do not silently strip attribution metadata during normalization.

## OWASP

OWASP states that project source code/build artifacts use OSI-approved open-source licenses and documentation uses Creative Commons or another open license; the main OWASP website is CC BY-SA 4.0 unless otherwise specified. Individual OWASP projects can therefore have different license files.

References:
- https://owasp.org/about/
- https://owasp.org/www-policy/operational/projects

Corpus decision: **project-level review required**. Never infer the license of one OWASP project from another. Pin the repository revision and preserve the project's exact LICENSE metadata.

## CISA

CISA material can contain mixed-origin content and product-specific notices. Public analysis reports may include TLP:CLEAR/TLP:WHITE sharing language while still being subject to standard copyright rules, and some reports incorporate third-party or foreign-government material.

Corpus decision: keep `cisa.advisories` at `REVIEW_REQUIRED`; approve artifact-by-artifact after checking the page's notification, attribution, TLP marking, and third-party content. TLP sharing permission is not treated as a training license by itself.

## Fail-closed rule

A source moves to `training_authorization=true` only when all of the following are present:

1. artifact/release is pinned;
2. rights basis is explicit and recorded;
3. attribution/share-alike obligations are preserved where applicable;
4. third-party embedded material is excluded or separately cleared;
5. benchmark-overlap risk is acceptable;
6. security review marks the source `APPROVED`.
