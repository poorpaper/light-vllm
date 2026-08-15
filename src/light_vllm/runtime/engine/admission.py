"""请求进入调度队列前的容量检查。"""

from light_vllm.runtime.engine.interfaces import EngineCapabilities
from light_vllm.runtime.generation.interfaces import GenerateRequest, GenerationRejectedError


class CapacityAdmission:
    """只拒绝即使独占整个引擎也装不下的请求。"""

    def validate(self, request: GenerateRequest, capabilities: EngineCapabilities) -> None:
        # 这里只检查请求本身是否过大；当前是否有空闲资源由 Scheduler 决定。
        limit = capabilities.max_request_tokens
        requested_tokens = len(request.input_ids) + request.max_new_tokens
        if limit is not None and requested_tokens > limit:
            raise GenerationRejectedError(
                f"request needs {requested_tokens} tokens, but the engine supports at most {limit}"
            )
