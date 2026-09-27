from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Iterable

from .cybersecurity_core import SecurityDomain


@dataclass(frozen=True)
class SecurityCorpusCase:
    case_id: str
    domain: SecurityDomain
    clean: bool
    adversarial: bool
    expected_findings: frozenset[str]
    expected_containment: bool
    fixture_digest: str

    def __post_init__(self) -> None:
        if self.clean and self.adversarial:
            raise ValueError("corpus case cannot be both clean and adversarial")
        if self.clean and self.expected_findings:
            raise ValueError("clean corpus case cannot require findings")
        if self.clean and self.expected_containment:
            raise ValueError("clean corpus case cannot require containment")


@dataclass(frozen=True)
class SecurityCorpus:
    corpus_id: str
    version: str
    cases: tuple[SecurityCorpusCase, ...]
    schema_version: str = "sentinel.security-corpus.v1"

    def __post_init__(self) -> None:
        if not self.cases:
            raise ValueError("security corpus cannot be empty")
        ids = [case.case_id for case in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("security corpus case ids must be unique")

    @property
    def clean_cases(self) -> int:
        return sum(case.clean for case in self.cases)

    @property
    def adversarial_cases(self) -> int:
        return sum(case.adversarial for case in self.cases)

    @property
    def domains(self) -> frozenset[SecurityDomain]:
        return frozenset(case.domain for case in self.cases)

    def digest(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), default=list).encode()
        return "sha256:" + hashlib.sha256(payload).hexdigest()


def validate_production_corpus(corpus: SecurityCorpus, required_domains: Iterable[SecurityDomain]) -> None:
    required = frozenset(required_domains)
    missing = required - corpus.domains
    if missing:
        raise ValueError(f"security corpus missing domains: {sorted(d.value for d in missing)!r}")
    if corpus.clean_cases == 0:
        raise ValueError("security corpus requires clean cases")
    if corpus.adversarial_cases == 0:
        raise ValueError("security corpus requires adversarial cases")
