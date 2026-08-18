from koschei_sentinel.cyber_sft_training import load_cyber_sft_config


def test_qwen35_micro_and_9b_smoke_pins_are_exact() -> None:
    micro = load_cyber_sft_config(
        "configs/training/cyber-sft.qwen3.5-0.8b.micro.json"
    )
    nine_b = load_cyber_sft_config(
        "configs/training/cyber-sft.qwen3.5-9b.smoke.json"
    )

    assert micro.base_model == "Qwen/Qwen3.5-0.8B-Base"
    assert micro.base_revision == "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"
    assert nine_b.base_model == "Qwen/Qwen3.5-9B-Base"
    assert nine_b.base_revision == "68c46c4b3498877f3ef123c856ecfde50c39f404"
    assert micro.quantization.bits == nine_b.quantization.bits == 4
    assert micro.quantization.compute_dtype == nine_b.quantization.compute_dtype == "float16"
    assert micro.minimum_cuda_memory_gb < nine_b.minimum_cuda_memory_gb
