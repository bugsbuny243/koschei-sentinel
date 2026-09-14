from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ActivationLayerSnapshot:
    module_name: str
    rms: float
    max_abs: float


@dataclass(frozen=True)
class ActivationProbeSnapshot:
    layers: tuple[ActivationLayerSnapshot, ...]

    @property
    def max_rms(self) -> float | None:
        return max((layer.rms for layer in self.layers), default=None)

    @property
    def max_abs(self) -> float | None:
        return max((layer.max_abs for layer in self.layers), default=None)


@dataclass
class MCoreActivationProbe:
    sample_elements: int = 4096
    handles: list[Any] = field(default_factory=list)
    _stats: dict[str, tuple[float, float]] = field(default_factory=dict)

    def reset(self) -> None:
        self._stats.clear()

    def _extract_tensor(self, output: Any) -> Any | None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("PyTorch is required for activation telemetry") from exc
        if isinstance(output, torch.Tensor):
            return output
        if isinstance(output, (tuple, list)):
            for value in output:
                if isinstance(value, torch.Tensor):
                    return value
        if isinstance(output, dict):
            for value in output.values():
                if isinstance(value, torch.Tensor):
                    return value
        return None

    def _hook_for(self, module_name: str):
        def hook(module: Any, inputs: Any, output: Any) -> None:
            del module, inputs
            tensor = self._extract_tensor(output)
            if tensor is None or tensor.numel() == 0:
                return
            try:
                import torch
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("PyTorch is required for activation telemetry") from exc
            sample = tensor.detach().reshape(-1)[: min(self.sample_elements, tensor.numel())].float()
            if sample.numel() == 0:
                return
            rms = float(torch.sqrt(torch.mean(sample * sample)).item())
            max_abs = float(torch.max(torch.abs(sample)).item())
            if not math.isfinite(rms) or not math.isfinite(max_abs):
                self._stats[module_name] = (rms, max_abs)
                return
            previous = self._stats.get(module_name)
            if previous is None:
                self._stats[module_name] = (rms, max_abs)
            else:
                self._stats[module_name] = (max(previous[0], rms), max(previous[1], max_abs))
        return hook

    def attach(self, model: Any) -> int:
        self.detach()
        count = 0
        for module_name, module in model.named_modules():
            class_name = module.__class__.__name__
            if class_name in {"TransformerLayer", "MoELayer"}:
                self.handles.append(module.register_forward_hook(self._hook_for(module_name or class_name)))
                count += 1
        if count == 0:
            raise RuntimeError("no transformer/MoE layers found for activation telemetry")
        return count

    def detach(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def snapshot(self) -> ActivationProbeSnapshot:
        layers = tuple(
            ActivationLayerSnapshot(module_name=name, rms=values[0], max_abs=values[1])
            for name, values in sorted(self._stats.items())
        )
        return ActivationProbeSnapshot(layers=layers)
