from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RouterProbeSnapshot:
    tokens_per_expert: list[int] | None = None
    aux_loss: float | None = None
    z_loss: float | None = None


@dataclass
class MCoreRouterProbe:
    """Non-invasive forward-hook probe for MCore MoE routers.

    The probe never changes router outputs. It opportunistically inspects tuple/dict outputs
    for a boolean routing map and accumulates per-expert token assignments. Aux/z losses may
    be supplied by the training loop when surfaced by the installed MCore logger/API.
    """

    handles: list[Any] = field(default_factory=list)
    _tokens_per_expert: list[int] | None = None
    _aux_loss: float | None = None
    _z_loss: float | None = None

    def reset(self) -> None:
        self._tokens_per_expert = None
        self._aux_loss = None
        self._z_loss = None

    def set_aux_loss(self, value: float | None) -> None:
        self._aux_loss = value

    def set_z_loss(self, value: float | None) -> None:
        self._z_loss = value

    def _accumulate_routing_map(self, routing_map: Any) -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("PyTorch is required for router telemetry") from exc
        if not isinstance(routing_map, torch.Tensor) or routing_map.ndim != 2:
            return
        if routing_map.dtype != torch.bool:
            # MCore routing maps are masks; avoid treating arbitrary score/probability tensors
            # as assignments when API return layouts differ between releases.
            return
        counts = routing_map.sum(dim=0, dtype=torch.int64).detach().cpu().tolist()
        counts = [int(value) for value in counts]
        if self._tokens_per_expert is None:
            self._tokens_per_expert = counts
        else:
            if len(self._tokens_per_expert) != len(counts):
                raise RuntimeError("router expert-count changed within a training step")
            self._tokens_per_expert = [a + b for a, b in zip(self._tokens_per_expert, counts)]

    def _hook(self, module: Any, inputs: Any, output: Any) -> None:
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
            self._accumulate_routing_map(candidate)

    def attach(self, model: Any) -> int:
        self.detach()
        count = 0
        for module in model.modules():
            class_name = module.__class__.__name__
            module_name = module.__class__.__module__
            if class_name == "TopKRouter" or (
                "megatron.core.transformer.moe" in module_name and "Router" in class_name
            ):
                self.handles.append(module.register_forward_hook(self._hook))
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
            tokens_per_expert=None if self._tokens_per_expert is None else list(self._tokens_per_expert),
            aux_loss=self._aux_loss,
            z_loss=self._z_loss,
        )
