from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RouterLayerSnapshot:
    module_name: str
    tokens_per_expert: list[int]


@dataclass(frozen=True)
class RouterProbeSnapshot:
    layers: list[RouterLayerSnapshot]
    aux_loss: float | None = None
    z_loss: float | None = None


@dataclass
class MCoreRouterProbe:
    """Non-invasive per-layer forward-hook probe for MCore MoE routers."""

    handles: list[Any] = field(default_factory=list)
    _tokens_per_layer: dict[str, list[int]] = field(default_factory=dict)
    _aux_loss: float | None = None
    _z_loss: float | None = None

    def reset(self) -> None:
        self._tokens_per_layer.clear()
        self._aux_loss = None
        self._z_loss = None

    def set_aux_loss(self, value: float | None) -> None:
        self._aux_loss = value

    def set_z_loss(self, value: float | None) -> None:
        self._z_loss = value

    def _accumulate_routing_map(self, module_name: str, routing_map: Any) -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("PyTorch is required for router telemetry") from exc
        if not isinstance(routing_map, torch.Tensor) or routing_map.ndim != 2:
            return
        if routing_map.dtype != torch.bool:
            return
        counts = [int(value) for value in routing_map.sum(dim=0, dtype=torch.int64).detach().cpu().tolist()]
        current = self._tokens_per_layer.get(module_name)
        if current is None:
            self._tokens_per_layer[module_name] = counts
            return
        if len(current) != len(counts):
            raise RuntimeError(f"router expert-count changed within step: {module_name}")
        self._tokens_per_layer[module_name] = [a + b for a, b in zip(current, counts)]

    def _hook_for(self, module_name: str):
        def hook(module: Any, inputs: Any, output: Any) -> None:
            del module, inputs
            candidates: list[Any] = []
            if isinstance(output, dict):
                for key in ("routing_map", "routing_mask", "expert_mask"):
                    if key in output:
                        candidates.append(output[key])
            elif isinstance(output, (tuple, list)):
                candidates.extend(output)
            else:
                candidates.append(output)
            for candidate in candidates:
                self._accumulate_routing_map(module_name, candidate)
        return hook

    def attach(self, model: Any) -> int:
        self.detach()
        count = 0
        for module_name, module in model.named_modules():
            class_name = module.__class__.__name__
            module_path = module.__class__.__module__
            if class_name == "TopKRouter" or (
                "megatron.core.transformer.moe" in module_path and "Router" in class_name
            ):
                self.handles.append(module.register_forward_hook(self._hook_for(module_name)))
                count += 1
        if count == 0:
            raise RuntimeError("no Megatron-Core MoE router modules found for telemetry")
        return count

    def detach(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def snapshot(self) -> RouterProbeSnapshot:
        return RouterProbeSnapshot(
            layers=[
                RouterLayerSnapshot(module_name=name, tokens_per_expert=list(counts))
                for name, counts in sorted(self._tokens_per_layer.items())
            ],
            aux_loss=self._aux_loss,
            z_loss=self._z_loss,
        )
