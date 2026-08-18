from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

from koschei_sentinel.cyber_model_access_preflight import audit_model_access
from koschei_sentinel.cyber_sft_training import CyberSFTConfig


class _FakeConfig:
    model_type = "qwen3_5"


class _FakeCausalLM:
    __name__ = "Qwen3_5ForCausalLM"


def _config() -> CyberSFTConfig:
    return CyberSFTConfig(
        run_id="model-preflight-test",
        stage="DEFENSE_REFLEX",
        base_model="Qwen/Qwen3.5-9B-Base",
        base_revision="a" * 40,
        corpus_dir="build/corpus",
        output_dir="build/run",
        minimum_cuda_memory_gb=0.0,
    )


def _install_fake_modules(monkeypatch, *, resolved=None, gated=False, private=False, mapped=None):
    resolved = resolved or ("a" * 40)
    mapped = mapped or _FakeCausalLM

    hub = ModuleType("huggingface_hub")

    def model_info(model_id, revision):
        assert model_id == "Qwen/Qwen3.5-9B-Base"
        assert revision == "a" * 40
        return SimpleNamespace(
            sha=resolved,
            gated=gated,
            private=private,
            siblings=[
                SimpleNamespace(rfilename="model-00001-of-00002.safetensors"),
                SimpleNamespace(rfilename="model-00002-of-00002.safetensors"),
                SimpleNamespace(rfilename="tokenizer.json"),
            ],
        )

    hub.model_info = model_info

    transformers = ModuleType("transformers")

    class AutoConfig:
        @staticmethod
        def from_pretrained(model_id, revision, trust_remote_code):
            assert model_id == "Qwen/Qwen3.5-9B-Base"
            assert revision == "a" * 40
            assert trust_remote_code is False
            return _FakeConfig()

    class AutoModelForCausalLM:
        _model_mapping = {_FakeConfig: mapped}

    transformers.AutoConfig = AutoConfig
    transformers.AutoModelForCausalLM = AutoModelForCausalLM

    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "transformers", transformers)


def test_exact_public_qwen35_revision_is_ready(monkeypatch) -> None:
    _install_fake_modules(monkeypatch)
    report = audit_model_access(_config())

    assert report.ready is True
    assert report.public_ungated is True
    assert report.resolved_revision == "a" * 40
    assert report.model_type == "qwen3_5"
    assert report.causal_lm_class == "Qwen3_5ForCausalLM"
    assert report.safetensors_files == 2
    assert report.tokenizer_files_present is True
    assert report.blockers == []


def test_revision_drift_is_rejected(monkeypatch) -> None:
    _install_fake_modules(monkeypatch, resolved="b" * 40)
    report = audit_model_access(_config())

    assert report.ready is False
    assert any("resolved revision differs" in row for row in report.blockers)


def test_gated_model_is_rejected(monkeypatch) -> None:
    _install_fake_modules(monkeypatch, gated=True)
    report = audit_model_access(_config())

    assert report.ready is False
    assert report.public_ungated is False
    assert any("gated" in row for row in report.blockers)


def test_wrong_causal_lm_mapping_is_rejected(monkeypatch) -> None:
    class WrongModel:
        pass

    _install_fake_modules(monkeypatch, mapped=WrongModel)
    report = audit_model_access(_config())

    assert report.ready is False
    assert report.causal_lm_class == "WrongModel"
    assert any("Qwen3_5ForCausalLM" in row for row in report.blockers)
