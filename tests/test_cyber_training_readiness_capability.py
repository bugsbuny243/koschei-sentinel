from koschei_sentinel.cyber_sft_training import CyberSFTConfig
from koschei_sentinel.cyber_training_readiness import _quantization_capability_blocker


def _config(bits: int) -> CyberSFTConfig:
    return CyberSFTConfig(
        run_id=f"capability-{bits}",
        stage="DEFENSE_REFLEX",
        base_model="Qwen/Qwen3.5-0.8B-Base",
        base_revision="a" * 40,
        corpus_dir="build/corpus",
        output_dir=f"build/run-{bits}",
        minimum_cuda_memory_gb=0.0,
        quantization={
            "bits": bits,
            "quant_type": "nf4",
            "double_quant": True,
            "compute_dtype": "float16",
        },
    )


def test_nf4_accepts_pascal_compute_capability_60() -> None:
    assert _quantization_capability_blocker(_config(4), (6, 0)) is None


def test_nf4_rejects_compute_capability_below_60() -> None:
    blocker = _quantization_capability_blocker(_config(4), (5, 2))
    assert blocker is not None
    assert "6.0+" in blocker


def test_load_in_8bit_rejects_compute_capability_below_75() -> None:
    blocker = _quantization_capability_blocker(_config(8), (7, 0))
    assert blocker is not None
    assert "7.5+" in blocker


def test_load_in_8bit_accepts_compute_capability_75() -> None:
    assert _quantization_capability_blocker(_config(8), (7, 5)) is None
