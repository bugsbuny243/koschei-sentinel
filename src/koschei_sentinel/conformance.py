from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Mapping


@dataclass(frozen=True)
class ConformanceCaseResult:
    case_id: str
    expected_findings: frozenset[str]
    observed_findings: frozenset[str]
    expected_containment: bool
    containment_verified: bool

    @property
    def true_positive(self) -> int:
        return len(self.expected_findings & self.observed_findings)

    @property
    def false_positive(self) -> int:
        return len(self.observed_findings - self.expected_findings)

    @property
    def false_negative(self) -> int:
        return len(self.expected_findings - self.observed_findings)


@dataclass(frozen=True)
class ConformanceReport:
    corpus_id: str
    corpus_version: str
    engine_id: str
    engine_version: str
    run_date: str
    case_count: int
    true_positives: int
    false_positives: int
    false_negatives: int
    containment_expected: int
    containment_verified: int
    schema_version: str = "sentinel.conformance.v1"

    @property
    def precision(self) -> float:
        d = self.true_positives + self.false_positives
        return self.true_positives / d if d else 1.0

    @property
    def recall(self) -> float:
        d = self.true_positives + self.false_negatives
        return self.true_positives / d if d else 1.0

    @property
    def containment_success_rate(self) -> float:
        return self.containment_verified / self.containment_expected if self.containment_expected else 1.0

    def digest(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()
        return "sha256:" + hashlib.sha256(payload).hexdigest()


def build_report(*, corpus_id: str, corpus_version: str, engine_id: str,
                 engine_version: str, run_date: str,
                 results: tuple[ConformanceCaseResult, ...]) -> ConformanceReport:
    if not results:
        raise ValueError("conformance report requires cases")
    ids = [r.case_id for r in results]
    if len(ids) != len(set(ids)):
        raise ValueError("conformance case ids must be unique")
    return ConformanceReport(
        corpus_id=corpus_id, corpus_version=corpus_version,
        engine_id=engine_id, engine_version=engine_version, run_date=run_date,
        case_count=len(results),
        true_positives=sum(r.true_positive for r in results),
        false_positives=sum(r.false_positive for r in results),
        false_negatives=sum(r.false_negative for r in results),
        containment_expected=sum(1 for r in results if r.expected_containment),
        containment_verified=sum(1 for r in results if r.expected_containment and r.containment_verified),
    )
