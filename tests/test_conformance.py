import pytest

from koschei_sentinel.conformance import ConformanceCaseResult, build_report


def test_report_measures_detection_and_containment():
    results = (
        ConformanceCaseResult("threat-1", frozenset({"f1"}), frozenset({"f1"}), True, True),
        ConformanceCaseResult("clean-1", frozenset(), frozenset(), False, False),
        ConformanceCaseResult("miss-1", frozenset({"f2"}), frozenset(), True, False),
        ConformanceCaseResult("fp-1", frozenset(), frozenset({"f3"}), False, False),
    )
    report = build_report(corpus_id="sentinel-lab", corpus_version="1",
        engine_id="fixture", engine_version="1", run_date="2026-09-27", results=results)
    assert report.true_positives == 1
    assert report.false_positives == 1
    assert report.false_negatives == 1
    assert report.precision == .5
    assert report.recall == .5
    assert report.containment_success_rate == .5
    assert report.digest().startswith("sha256:")


def test_duplicate_case_ids_are_rejected():
    item = ConformanceCaseResult("same", frozenset(), frozenset(), False, False)
    with pytest.raises(ValueError, match="unique"):
        build_report(corpus_id="c", corpus_version="1", engine_id="e",
            engine_version="1", run_date="2026-09-27", results=(item, item))
