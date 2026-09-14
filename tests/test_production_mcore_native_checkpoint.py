from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")

from koschei_sentinel.production_mcore_native_checkpoint import initialize_mcore_native_parameters


class TinyNativeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=8)
        self.input_layernorm = torch.nn.LayerNorm(8, elementwise_affine=True)
        self.linear_qkv = torch.nn.Linear(8, 24, bias=False)
        self.router = torch.nn.Linear(8, 4, bias=False)


def test_native_initialization_is_deterministic() -> None:
    left = TinyNativeModel()
    right = TinyNativeModel()

    left_result = initialize_mcore_native_parameters(left, global_seed=39735)
    right_result = initialize_mcore_native_parameters(right, global_seed=39735)

    assert left_result.parameter_count == right_result.parameter_count
    assert left_result.local_elements == right_result.local_elements
    for (left_name, left_param), (right_name, right_param) in zip(
        left.named_parameters(), right.named_parameters(), strict=True
    ):
        assert left_name == right_name
        assert torch.equal(left_param, right_param)


def test_norm_weight_is_one_and_linear_weights_are_nonconstant() -> None:
    model = TinyNativeModel()
    initialize_mcore_native_parameters(model, global_seed=39735)

    assert torch.equal(model.input_layernorm.weight, torch.ones_like(model.input_layernorm.weight))
    assert torch.count_nonzero(model.linear_qkv.weight - model.linear_qkv.weight.flatten()[0]) > 0


def test_different_seed_changes_non_norm_parameters() -> None:
    left = TinyNativeModel()
    right = TinyNativeModel()
    initialize_mcore_native_parameters(left, global_seed=39735)
    initialize_mcore_native_parameters(right, global_seed=39736)
    assert not torch.equal(left.linear_qkv.weight, right.linear_qkv.weight)
