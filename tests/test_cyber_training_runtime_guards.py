from koschei_sentinel.cyber_sft_training import CyberSFTConfig
from koschei_sentinel.cyber_training_readiness import (
    _quantization_capability_blocker,
    _transformers_version_blocker,
)


def _config(*, bits: int = 4) -> CyberSFTConfig:
    return CyberSFTConfig(
        run_id=f"runtime-guard-{bits}bit",
        stage="DEFENSE_REFLEX",
        base_model="Qwen/Qwen3.5-9B-Base",
        base_revision="a" * 40,
        corpus_dir="build/corpus",
        output_dir=f"build/run-{bits}bit",
        minimum_cuda_memory_gb=0.0,
        quantization={
            "bits": bits,
            "quant_type": "nf4",
            "double_quant": True,
            "compute_dtype": "float16",
        },
    )


def test_transformers_511_is_blocked_for_qwen35_dtype_fix() -> None:
    blocker = _transformers_version_blocker("5.11.4")

    assert blocker is not None
    assert "transformers>=5.12" in blocker


def test_transformers_512_final_release_is_accepted() -> None:
    assert _transformers_version_blocker("5.12.0") is None
    assert _transformers_version_blocker("5.12.1") is None


def test_transformers_512_prerelease_is_blocked() -> None:
    blocker = _transformers_version_blocker("5.12.0rc1")

    assert blocker is not None
    assert "prerelease/nightly" in blocker


def test_unparseable_transformers_version_is_blocked() -> None:
    blocker = _transformers_version_blocker("not-a-version")

    assert blocker is not None
    assert "cannot parse" in blocker


def test_nf4_accepts_pascal_compute_capability_60() -> None:
    assert _quantization_capability_blocker(_config(bits=4), (6, 0)) is None


def test_nf4_rejects_gpu_below_compute_capability_60() -> None:
    blocker = _quantization_capability_blocker(_config(bits=4), (5, 2))

    assert blocker is not None
    assert "compute capability 6.0+" in blocker


def test_8bit_path_requires_compute_capability_75() -> None:
    blocker = _quantization_capability_blocker(_config(bits=8), (7, 0))

    assert blocker is not None
    assert "compute capability 7.5+" in blocker
    assert _quantization_capability_blocker(_config(bits=8), (7, 5)) is None
