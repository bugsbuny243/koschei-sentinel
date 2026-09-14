from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Iterable

from koschei_sentinel.production_foundation_data import PackedFoundationSequence


@dataclass
class PreparedFoundationBatch:
    tokens: Any
    labels: Any
    loss_mask: Any
    position_ids: Any
    attention_mask: Any = None
    cu_seqlens: Any = None


def _pin_sequence(sequence: PackedFoundationSequence) -> PreparedFoundationBatch:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required for pinned foundation staging") from exc
    return PreparedFoundationBatch(
        tokens=torch.tensor([sequence.input_ids], dtype=torch.long, pin_memory=True),
        labels=torch.tensor([sequence.labels], dtype=torch.long, pin_memory=True),
        loss_mask=torch.tensor([sequence.loss_mask], dtype=torch.float32, pin_memory=True),
        position_ids=torch.tensor([sequence.position_ids], dtype=torch.long, pin_memory=True),
    )


def _async_to_device(batch: PreparedFoundationBatch, *, device: Any, stream: Any) -> PreparedFoundationBatch:
    import torch
    with torch.cuda.stream(stream):
        return PreparedFoundationBatch(
            tokens=batch.tokens.to(device=device, non_blocking=True),
            labels=batch.labels.to(device=device, non_blocking=True),
            loss_mask=batch.loss_mask.to(device=device, non_blocking=True),
            position_ids=batch.position_ids.to(device=device, non_blocking=True),
        )


class FoundationDoubleBuffer:
    """Double-buffered pinned CPU staging plus async H2D transfer.

    This class overlaps the next host-to-device copy with compute on the current batch.
    It does not claim end-to-end overlap until exercised on a real CUDA/MCore job.
    """

    def __init__(self, sequences: Iterable[PackedFoundationSequence], *, device: Any, depth: int = 2):
        if depth < 2:
            raise ValueError("prefetch depth must be at least 2")
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("PyTorch is required for CUDA prefetch") from exc
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for async foundation prefetch")
        self._torch = torch
        self._device = device
        self._source = iter(sequences)
        self._stream = torch.cuda.Stream(device=device)
        self._pending: deque[PreparedFoundationBatch] = deque()
        self._depth = depth
        self._fill()

    def _fill(self) -> None:
        while len(self._pending) < self._depth:
            try:
                sequence = next(self._source)
            except StopIteration:
                return
            staged = _pin_sequence(sequence)
            self._pending.append(_async_to_device(staged, device=self._device, stream=self._stream))

    def __iter__(self) -> "FoundationDoubleBuffer":
        return self

    def __next__(self) -> PreparedFoundationBatch:
        if not self._pending:
            raise StopIteration
        self._torch.cuda.current_stream(self._device).wait_stream(self._stream)
        batch = self._pending.popleft()
        # Record ownership on the consumer stream so pinned staging / CUDA allocations
        # are not reused before compute has consumed them.
        for tensor in (batch.tokens, batch.labels, batch.loss_mask, batch.position_ids):
            tensor.record_stream(self._torch.cuda.current_stream(self._device))
        self._fill()
        return batch
