import shutil

from koschei_sentinel.defense_reflex_gold_release_audit import (
    audit_gold_defense_release,
)
from koschei_sentinel.gold_holdout_inference_verify import (
    verify_gold_holdout_inference_output,
)
from tests.test_defense_reflex_gold_release_audit import _release
from tests.test_gold_holdout_inference_verify import _fixture


def test_gold_release_audit_digest_survives_host_path_relocation(tmp_path) -> None:
    release = _release(tmp_path)
    relocated = tmp_path / "relocated" / "gold-release"
    shutil.copytree(release, relocated)

    original = audit_gold_defense_release(release)
    moved = audit_gold_defense_release(relocated)

    assert original.valid is True
    assert moved.valid is True
    assert original.release_dir != moved.release_dir
    assert original.audit_sha256 == moved.audit_sha256


def test_gold_inference_verification_digest_survives_output_relocation(
    tmp_path,
    monkeypatch,
) -> None:
    pack, output, _cases, _plan, candidate_export = _fixture(tmp_path, monkeypatch)
    relocated = tmp_path / "relocated-inference-output"
    shutil.copytree(output, relocated)

    original = verify_gold_holdout_inference_output(output, pack, candidate_export)
    moved = verify_gold_holdout_inference_output(relocated, pack, candidate_export)

    assert original.valid is True
    assert moved.valid is True
    assert original.output_dir != moved.output_dir
    assert original.verification_sha256 == moved.verification_sha256
