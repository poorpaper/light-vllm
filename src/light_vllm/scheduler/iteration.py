"""基于等待队列和运行集合的 iteration scheduler。"""

from __future__ import annotations

from collections import deque

from light_vllm.scheduler.api import SchedulerBatch, SchedulerError


class _IterationScheduler:
    """维护 FCFS 等待队列与当前运行集合。

    ``_waiting`` 保留请求到达顺序；``_waiting_ids`` 只用于快速判重；
    ``_running`` 用有序字典同时表达成员关系和稳定的批次行顺序。
    新请求必须先进入 waiting，只有 ``schedule`` 能在迭代安全点接纳它。
    """

    def __init__(self, *, max_num_sequences: int) -> None:
        if type(max_num_sequences) is not int or max_num_sequences <= 0:
            raise ValueError("max_num_sequences must be a positive integer")
        self._max_num_sequences = max_num_sequences
        # deque 负责 FCFS 顺序，set 负责 O(1) 判重；两者必须同步更新。
        self._waiting: deque[str] = deque()
        self._waiting_ids: set[str] = set()
        # dict 保留接纳顺序；值没有额外含义。
        self._running: dict[str, None] = {}

    @property
    def has_requests(self) -> bool:
        return bool(self._waiting or self._running)

    def add(self, request_id: str) -> None:
        """把新请求放到等待队尾，不在提交路径直接改变运行批次。"""

        if not request_id:
            raise ValueError("request_id must not be empty")
        if request_id in self._waiting_ids or request_id in self._running:
            raise SchedulerError(f"request {request_id!r} is already scheduled")
        self._waiting.append(request_id)
        self._waiting_ids.add(request_id)

    def remove(self, request_id: str) -> bool:
        """幂等移除等待或运行请求，用于完成、失败和取消。"""

        # 运行请求最常在每轮 update 阶段结束，优先走 O(1) 路径。
        if request_id in self._running:
            del self._running[request_id]
            return True
        if request_id not in self._waiting_ids:
            return False

        # 取消等待请求较少发生；deque.remove 的 O(n) 换取了更简单清晰的
        # 双队列一致性，后续有真实规模数据再考虑专用队列结构。
        self._waiting_ids.remove(request_id)
        self._waiting.remove(request_id)
        return True

    def _fill_open_slots(self) -> None:
        """按 FCFS 顺序把等待请求接纳到空闲运行槽。"""

        while self._waiting and len(self._running) < self._max_num_sequences:
            request_id = self._waiting.popleft()
            self._waiting_ids.remove(request_id)
            self._running[request_id] = None

    def _current_batch(self) -> SchedulerBatch:
        """返回当前运行集合的不可变快照。"""

        return SchedulerBatch(request_ids=tuple(self._running))


class ContinuousBatchScheduler(_IterationScheduler):
    """每轮都用等待请求补满空出的执行槽。

    完成或取消的请求在 update 阶段被移除；下一次 ``schedule`` 立即补位，
    这就是当前实现的 iteration-level continuous batching 语义。
    """

    def schedule(self) -> SchedulerBatch:
        self._fill_open_slots()
        return self._current_batch()


class RawBatchScheduler(_IterationScheduler):
    """当前静态批次清空后，才接纳下一批等待请求。

    即使某个请求提前完成并空出槽位，只要同一批还有请求运行，就不补位。
    它与 continuous 策略共用 Engine 和执行器，作为公平的 raw batching 基线。
    """

    def schedule(self) -> SchedulerBatch:
        # 只有 running 为空才形成新一批；批次存续期间成员只会减少。
        if not self._running:
            self._fill_open_slots()
        return self._current_batch()
