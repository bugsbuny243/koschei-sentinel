import pytest

from koschei_sentinel.cybersecurity_core import SecurityDomain
from koschei_sentinel.security_corpus import SecurityCorpus, SecurityCorpusCase, validate_production_corpus


def c(case_id, domain, *, clean=False, adversarial=False, findings=frozenset(), containment=False):
    return SecurityCorpusCase(case_id, domain, clean, adversarial, findings, containment, f"sha256:{case_id}")


def test_multidomain_corpus_is_versioned_and_digestible():
    corpus = SecurityCorpus("sentinel-security", "1", (
        c("code-threat", SecurityDomain.CODE, adversarial=True, findings=frozenset({"unsafe-code"}), containment=True),
        c("log-clean", SecurityDomain.LOG, clean=True),
        c("network-threat", SecurityDomain.NETWORK, adversarial=True, findings=frozenset({"network-anomaly"}), containment=True),
    ))
    validate_production_corpus(corpus, (SecurityDomain.CODE, SecurityDomain.LOG, SecurityDomain.NETWORK))
    assert corpus.clean_cases == 1
    assert corpus.adversarial_cases == 2
    assert corpus.digest().startswith("sha256:")


def test_missing_domain_fails_validation():
    corpus = SecurityCorpus("sentinel-security", "1", (
        c("code-threat", SecurityDomain.CODE, adversarial=True, findings=frozenset({"unsafe-code"})),
        c("code-clean", SecurityDomain.CODE, clean=True),
    ))
    with pytest.raises(ValueError, match="missing domains"):
        validate_production_corpus(corpus, (SecurityDomain.CODE, SecurityDomain.WEB3))


def test_clean_case_cannot_expect_containment():
    with pytest.raises(ValueError, match="clean corpus case"):
        c("bad-clean", SecurityDomain.CODE, clean=True, containment=True)
