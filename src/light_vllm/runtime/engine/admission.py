"""生成请求的确定性容量检查。"""

from light_vllm.runtime.engine.interfaces import EngineCapabilities
from light_vllm.runtime.generation.interfaces import GenerateRequest, GenerationRejectedError


class CapacityAdmission:
    """只拒绝单个请求永远无法满足的容量条件。"""

    def validate(self, request: GenerateRequest, capabilities: EngineCapabilities) -> None:
        # 这里看空闲引擎下的单请求上限；实时竞争和等待由 Scheduler 处理。
        limit = capabilities.max_request_tokens
        requested_tokens = len(request.input_ids) + request.max_new_tokens
        if limit is not None and requested_tokens > limit:
            raise GenerationRejectedError(
                f"request needs {requested_tokens} tokens, but the engine supports at most {limit}"
            )
