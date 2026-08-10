"""Readable generation baseline with no scheduler or KV cache.

The implementation deliberately recomputes the full token sequence for each
new token. That makes it a poor production engine but a useful executable
specification for testing future pipeline and scheduled implementations.
"""

from __future__ import annotations

from collections.abc import Iterator
from threading import Lock
from typing import Protocol

import torch

from light_vllm.contracts import (
    ForwardBatch,
    GenerateRequest,
    GenerateResult,
    GenerationError,
    GenerationEvent,
    GenerationFinished,
    GenerationNotReadyError,
    ModelOutput,
    TokenGenerated,
)
from light_vllm.runner import ModelNotLoadedError


class _ModelExecutor(Protocol):
    """Smallest runner surface required by the reference generator."""

    @property
    def generation(self) -> int: ...

    def forward(self, batch: ForwardBatch) -> ModelOutput: ...


class GreedyGenerationService:
    """Readable baseline that emits one argmax token at a time.

    The execution lock serializes complete requests because this reference path
    has no scheduler. It is independent from ``ModelRunner``'s lifecycle lock,
    which only protects model replacement.
    """

    def __init__(self, runner: _ModelExecutor, *, device: str | torch.device = "cpu") -> None:
        self._runner = runner
        self._device = torch.device(device)
        self._execution_lock = Lock()

    @property
    def ready(self) -> bool:
        return self._runner.generation > 0

    def stream(self, request: GenerateRequest) -> Iterator[GenerationEvent]:
        """Generate domain events; this is the service's only execution path."""

        token_ids = list(request.input_ids)

        with self._execution_lock:
            try:
                for position in range(request.max_new_tokens):
                    # Rebuilding the complete input is intentional in the
                    # baseline. A KV-aware implementation will replace this
                    # strategy behind the same higher-level event semantics.
                    batch = ForwardBatch(
                        input_ids=torch.tensor([token_ids], dtype=torch.long, device=self._device)
                    )
                    output = self._runner.forward(batch)
                    if output.logits.ndim != 3 or output.logits.shape[:2] != batch.input_ids.shape:
                        raise GenerationError(
                            "model logits must have shape [batch, sequence, vocabulary]"
                        )

                    token_id = int(output.logits[0, -1].argmax().item())
                    token_ids.append(token_id)
                    yield TokenGenerated(token_id=token_id, position=position)

                    if request.eos_token_id is not None and token_id == request.eos_token_id:
                        yield GenerationFinished(finish_reason="eos")
                        return
            except ModelNotLoadedError as exc:
                raise GenerationNotReadyError("load a model before generating") from exc

            yield GenerationFinished(finish_reason="length")

    def generate(self, request: GenerateRequest) -> GenerateResult:
        """Collect ``stream`` without duplicating sampling or stop logic."""

        generated_token_ids: list[int] = []
        finish_reason = None

        for event in self.stream(request):
            if isinstance(event, TokenGenerated):
                generated_token_ids.append(event.token_id)
            elif isinstance(event, GenerationFinished):
                finish_reason = event.finish_reason

        if finish_reason is None:
            raise GenerationError("generation stream ended without a terminal event")

        return GenerateResult(
            input_ids=request.input_ids,
            generated_token_ids=tuple(generated_token_ids),
            finish_reason=finish_reason,
        )
