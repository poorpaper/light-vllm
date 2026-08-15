"""不带调度器和 KV cache 的最简单生成实现。

每生成一个 token 都会重新计算完整输入，速度不快，但逻辑清楚，
方便测试和对比后续实现。
"""

from __future__ import annotations

from collections.abc import Iterator
from threading import Lock

from light_vllm.runtime.execution.interfaces import (
    ExecutionError,
    ExecutionNotReadyError,
    TokenExecutor,
)
from light_vllm.runtime.generation.interfaces import (
    GenerateRequest,
    GenerateResult,
    GenerationError,
    GenerationEvent,
    GenerationFinished,
    GenerationNotReadyError,
    TokenGenerated,
)


class ReferenceGenerationService:
    """不带调度器的同步参考生成服务。

    token 的计算交给执行器。当前同一时间只执行一个完整请求，这里的
    生成锁和模型切换锁互不影响。
    """

    def __init__(self, executor: TokenExecutor) -> None:
        self._executor = executor
        self._execution_lock = Lock()

    @property
    def ready(self) -> bool:
        return self._executor.ready

    def stream(self, request: GenerateRequest) -> Iterator[GenerationEvent]:
        """逐个生成并返回事件。这是实际执行生成的唯一入口。"""

        token_ids = list(request.input_ids)

        with self._execution_lock:
            try:
                # 一个请求只打开一次 session，生成途中 reload 不会切换模型。
                execution = self._executor.open_session()
                for position in range(request.max_new_tokens):
                    token_id = execution.next_token(tuple(token_ids))
                    token_ids.append(token_id)
                    yield TokenGenerated(token_id=token_id, position=position)

                    if request.eos_token_id is not None and token_id == request.eos_token_id:
                        yield GenerationFinished(finish_reason="eos")
                        return
            except ExecutionNotReadyError as exc:
                raise GenerationNotReadyError("load a model before generating") from exc
            except ExecutionError as exc:
                raise GenerationError(str(exc)) from exc

            yield GenerationFinished(finish_reason="length")

    def generate(self, request: GenerateRequest) -> GenerateResult:
        """收集 ``stream`` 返回的事件，避免重复写一套生成逻辑。"""

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
