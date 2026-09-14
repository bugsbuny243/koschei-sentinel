from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class NativeMoELossSnapshot:
    aux_loss: float | None
    z_loss: float | None
    metric_names: tuple[str, ...]


def _metric_scalar(entry: Any) -> float | None:
    values = getattr(entry, "values", None)
    if values is None:
        return None
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required for MCore MoE metric inspection") from exc
    if not isinstance(values, torch.Tensor) or values.numel() == 0:
        return None
    finite = values.detach().float().reshape(-1)
    # Preserve NaN/Inf if present so the stability gate can reject them.
    return float(finite.mean().item())


def clear_native_moe_metrics() -> bool:
    """Clear the current MCore MoE tracker when the installed API exposes it.

    Returns False on older MCore releases that do not provide the documented tracker module.
    """
    try:
        from megatron.core.transformer.moe.moe_logging import get_moe_metrics_tracker
    except ImportError:
        return False
    tracker = get_moe_metrics_tracker()
    clear = getattr(tracker, "clear", None)
    if not callable(clear):
        raise RuntimeError("installed MCore MoE tracker lacks documented clear() API")
    clear()
    return True


def read_native_moe_losses() -> NativeMoELossSnapshot:
    """Read local, pre-report MoE losses from MCore's documented metrics property.

    This does not call report(), perform distributed reduction, or clear the tracker. Values
    are intentionally local stability signals; the training loop may still use MCore report()
    separately for canonical distributed logging.
    """
    try:
        from megatron.core.transformer.moe.moe_logging import get_moe_metrics_tracker
    except ImportError:
        return NativeMoELossSnapshot(aux_loss=None, z_loss=None, metric_names=())

    tracker = get_moe_metrics_tracker()
    metrics = getattr(tracker, "metrics", None)
    if metrics is None:
        raise RuntimeError("installed MCore MoE tracker lacks documented metrics property")
    if not isinstance(metrics, dict):
        try:
            metrics = dict(metrics)
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("MCore MoE tracker metrics are not mapping-like") from exc

    aux_values: list[float] = []
    z_values: list[float] = []
    names: list[str] = []
    for raw_name, entry in metrics.items():
        name = str(raw_name)
        names.append(name)
        scalar = _metric_scalar(entry)
        if scalar is None:
            continue
        lowered = name.lower()
        if "z_loss" in lowered or "z-loss" in lowered or lowered.endswith("z loss"):
            z_values.append(scalar)
        elif "load_balancing" in lowered or "aux_loss" in lowered or "aux-loss" in lowered:
            aux_values.append(scalar)

    aux = sum(aux_values) / len(aux_values) if aux_values else None
    z_loss = sum(z_values) / len(z_values) if z_values else None
    return NativeMoELossSnapshot(
        aux_loss=aux,
        z_loss=z_loss,
        metric_names=tuple(sorted(names)),
    )
