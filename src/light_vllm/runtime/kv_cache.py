"""KV cache 的逻辑分配与本地连续张量存储。"""

from __future__ import annotations

from dataclasses import dataclass
from heapq import heapify, heappop, heappush
from threading import RLock
from typing import Protocol

import torch
from torch import Tensor

from light_vllm.modeling.models.interfaces import KVCacheState, LayerKeyValues


class KVCacheError(RuntimeError):
    """KV cache 状态违反契约时抛出。"""


class KVCacheCapacityError(KVCacheError):
    """没有足够的 KV slot 完成本轮预留时抛出。"""


class KVCacheNotFoundError(KVCacheError):
    """访问不存在或已经释放的请求缓存时抛出。"""


@dataclass(frozen=True, slots=True)
class KVCacheReservation:
    """Scheduler 为一次执行预留的逻辑 KV 空间和可选物理位置。

    分页 manager 返回请求当前完整的 ``block_ids``；其中可能包含刚为本轮
    增加、但尚未提交的 block。连续缓存不需要 block table，可以返回 ``None``。
    执行成功后只保留实际提交 token 覆盖的资源。
    """

    block_ids: tuple[int, ...] | None
    num_committed_tokens: int
    num_reserved_tokens: int


class KVCacheManager(Protocol):
    """Scheduler 管理 KV 预留状态的接口。

    实现可以返回 block table，也可以只记录 token 数；两者都必须支持提交和释放。
    """

    def add_request(self, request_id: str) -> None: ...

    def reserve(self, request_id: str, num_tokens: int) -> KVCacheReservation:
        """预留本轮 token 的空间；空间不足时抛错。"""

        ...

    def commit(self, request_id: str, num_tokens: int) -> None:
        """提交实际算完的 token，并清掉未使用的预留。"""

        ...

    def free(self, request_id: str) -> bool:
        """释放请求状态；重复释放返回 False。"""

        ...


@dataclass(slots=True)
class _LogicalAllocation:
    block_ids: list[int]
    num_committed_tokens: int = 0
    num_reserved_tokens: int = 0


@dataclass(slots=True)
class _UnboundedAllocation:
    num_committed_tokens: int = 0
    num_reserved_tokens: int = 0


class UnboundedKVCacheManager:
    """为非分页缓存维护预留生命周期，不分配 block，也不限制容量。

    它只做状态记录，方便和分页 manager 做公平的生成流程对比。
    """

    def __init__(self) -> None:
        self._allocations: dict[str, _UnboundedAllocation] = {}

    def add_request(self, request_id: str) -> None:
        if not request_id:
            raise ValueError("request_id must not be empty")
        if request_id in self._allocations:
            raise KVCacheError(f"KV cache for request {request_id!r} already exists")
        self._allocations[request_id] = _UnboundedAllocation()

    def reserve(self, request_id: str, num_tokens: int) -> KVCacheReservation:
        # 即使不分 block，也要记录预留，防止同一请求被重复调度。
        if type(num_tokens) is not int or num_tokens <= 0:
            raise ValueError("num_tokens must be a positive integer")
        allocation = self._get(request_id)
        if allocation.num_reserved_tokens:
            raise KVCacheError(f"request {request_id!r} already has an active reservation")

        allocation.num_reserved_tokens = num_tokens
        return KVCacheReservation(
            block_ids=None,
            num_committed_tokens=allocation.num_committed_tokens,
            num_reserved_tokens=num_tokens,
        )

    def commit(self, request_id: str, num_tokens: int) -> None:
        # 只提交实际完成的前缀；剩余预留在这里一并取消。
        allocation = self._get(request_id)
        if type(num_tokens) is not int or not 0 <= num_tokens <= allocation.num_reserved_tokens:
            raise ValueError("committed token count must be within the active reservation")

        allocation.num_committed_tokens += num_tokens
        allocation.num_reserved_tokens = 0

    def free(self, request_id: str) -> bool:
        # Engine 在完成、失败和取消时都会调用，重复释放应保持安全。
        return self._allocations.pop(request_id, None) is not None

    def _get(self, request_id: str) -> _UnboundedAllocation:
        try:
            return self._allocations[request_id]
        except KeyError as exc:
            raise KVCacheNotFoundError(
                f"KV cache for request {request_id!r} was not found"
            ) from exc


class PagedKVCacheManager:
    """使用固定大小逻辑 block 管理全局 KV 容量。

    这里只实现 Scheduler 需要的预留、提交、回滚和释放语义，不保存 K/V
    tensor，也不实现 Paged Attention。未来物理分页后端可以直接消费这里
    产生的 block table，而不改变调度契约。
    """

    def __init__(self, *, num_blocks: int, block_size: int) -> None:
        if type(num_blocks) is not int or num_blocks <= 0:
            raise ValueError("num_blocks must be a positive integer")
        if type(block_size) is not int or block_size <= 0:
            raise ValueError("block_size must be a positive integer")
        self._num_blocks = num_blocks
        self._block_size = block_size
        self._free_blocks = list(range(num_blocks))
        heapify(self._free_blocks)
        self._allocations: dict[str, _LogicalAllocation] = {}

    @property
    def block_size(self) -> int:
        return self._block_size

    @property
    def num_free_blocks(self) -> int:
        return len(self._free_blocks)

    def add_request(self, request_id: str) -> None:
        if not request_id:
            raise ValueError("request_id must not be empty")
        if request_id in self._allocations:
            raise KVCacheError(f"KV cache for request {request_id!r} already exists")
        self._allocations[request_id] = _LogicalAllocation(block_ids=[])

    def reserve(self, request_id: str, num_tokens: int) -> KVCacheReservation:
        """原子地预留本轮 token 可能占用的新 block。"""

        if type(num_tokens) is not int or num_tokens <= 0:
            raise ValueError("num_tokens must be a positive integer")
        allocation = self._get(request_id)
        if allocation.num_reserved_tokens:
            raise KVCacheError(f"request {request_id!r} already has an active reservation")

        target_tokens = allocation.num_committed_tokens + num_tokens
        target_blocks = self._blocks_for(target_tokens)
        new_block_count = target_blocks - len(allocation.block_ids)
        if new_block_count > len(self._free_blocks):
            raise KVCacheCapacityError(
                f"request {request_id!r} needs {new_block_count} new KV blocks, "
                f"but only {len(self._free_blocks)} are free"
            )

        allocation.block_ids.extend(heappop(self._free_blocks) for _ in range(new_block_count))
        allocation.num_reserved_tokens = num_tokens
        return KVCacheReservation(
            block_ids=tuple(allocation.block_ids),
            num_committed_tokens=allocation.num_committed_tokens,
            num_reserved_tokens=num_tokens,
        )

    def commit(self, request_id: str, num_tokens: int) -> None:
        """提交预留前缀；未提交的尾部等价于回滚并立即归还 block。"""

        allocation = self._get(request_id)
        if type(num_tokens) is not int or not 0 <= num_tokens <= allocation.num_reserved_tokens:
            raise ValueError("committed token count must be within the active reservation")

        allocation.num_committed_tokens += num_tokens
        allocation.num_reserved_tokens = 0
        self._trim_blocks(allocation)

    def free(self, request_id: str) -> bool:
        allocation = self._allocations.pop(request_id, None)
        if allocation is None:
            return False
        for block_id in allocation.block_ids:
            heappush(self._free_blocks, block_id)
        return True

    def _trim_blocks(self, allocation: _LogicalAllocation) -> None:
        keep = self._blocks_for(allocation.num_committed_tokens)
        released = allocation.block_ids[keep:]
        del allocation.block_ids[keep:]
        for block_id in released:
            heappush(self._free_blocks, block_id)

    def _blocks_for(self, num_tokens: int) -> int:
        return (num_tokens + self._block_size - 1) // self._block_size

    def _get(self, request_id: str) -> _LogicalAllocation:
        try:
            return self._allocations[request_id]
        except KeyError as exc:
            raise KVCacheNotFoundError(
                f"KV cache for request {request_id!r} was not found"
            ) from exc


@dataclass(frozen=True, slots=True)
class KVCacheSpec:
    """本地连续 KV tensor 的逐层布局。"""

    num_layers: int
    num_kv_heads: int
    head_size: int
    dtype: torch.dtype = torch.float32
    device: str | torch.device = "cpu"

    def __post_init__(self) -> None:
        for name, value in (
            ("num_layers", self.num_layers),
            ("num_kv_heads", self.num_kv_heads),
            ("head_size", self.head_size),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not self.dtype.is_floating_point:
            raise ValueError("KV cache dtype must be floating point")
        object.__setattr__(self, "device", torch.device(self.device))


class KVCacheLease(Protocol):
    """固定一批物理缓存生命周期的幂等租约。"""

    def release(self) -> None: ...


@dataclass(slots=True)
class _CacheEntry:
    keys: Tensor
    values: Tensor
    length: int = 0
    users: int = 0
    released: bool = False

    @property
    def capacity(self) -> int:
        return self.keys.shape[1]


@dataclass(slots=True)
class _ContiguousKVCacheLease:
    cache: ContiguousKVCache
    request_ids: tuple[str, ...]
    entries: tuple[_CacheEntry, ...]
    released: bool = False

    def release(self) -> None:
        if not self.released:
            self.cache._release_lease(self.request_ids, self.entries)
            self.released = True


class ContiguousKVCache:
    """执行侧的请求级连续 K/V tensor 存储。

    这是 Paged Attention 落地前的物理基线。Scheduler 看不到这些 tensor；
    Executor 也不会修改逻辑 block 的分配策略。
    """

    def __init__(self, spec: KVCacheSpec) -> None:
        self._spec = spec
        self._entries: dict[str, _CacheEntry] = {}
        self._lock = RLock()

    @property
    def spec(self) -> KVCacheSpec:
        return self._spec

    @property
    def num_requests(self) -> int:
        with self._lock:
            return sum(not entry.released for entry in self._entries.values())

    def allocate(self, request_id: str, capacity: int) -> None:
        if not request_id:
            raise ValueError("request_id must not be empty")
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        with self._lock:
            if request_id in self._entries:
                raise KVCacheError(f"KV cache for request {request_id!r} already exists")
            shape = (
                self._spec.num_layers,
                capacity,
                self._spec.num_kv_heads,
                self._spec.head_size,
            )
            self._entries[request_id] = _CacheEntry(
                keys=torch.empty(shape, dtype=self._spec.dtype, device=self._spec.device),
                values=torch.empty(shape, dtype=self._spec.dtype, device=self._spec.device),
            )

    def contains(self, request_id: str) -> bool:
        with self._lock:
            entry = self._entries.get(request_id)
            return entry is not None and not entry.released

    def cached_tokens(self, request_id: str) -> int:
        with self._lock:
            return self._get_entry(request_id).length

    def view(self, request_id: str) -> KVCacheState:
        """返回当前有效前缀的语义视图，不复制 K/V。"""

        with self._lock:
            entry = self._get_entry(request_id)
            layers = tuple(
                LayerKeyValues(
                    keys=entry.keys[layer, : entry.length].unsqueeze(0),
                    values=entry.values[layer, : entry.length].unsqueeze(0),
                )
                for layer in range(self._spec.num_layers)
            )
            return KVCacheState(layers=layers)

    @torch.inference_mode()
    def append(self, request_id: str, updates: KVCacheState) -> None:
        """校验所有层后，一次性追加同一段 token 的 K/V。"""

        with self._lock:
            entry = self._get_entry(request_id)
            self._validate_updates(updates)
            new_length = entry.length + updates.num_tokens
            if new_length > entry.capacity:
                raise KVCacheCapacityError(
                    f"request {request_id!r} needs {new_length} KV slots, "
                    f"but only {entry.capacity} were allocated"
                )
            target = slice(entry.length, new_length)
            for layer_index, layer in enumerate(updates.layers):
                entry.keys[layer_index, target].copy_(layer.keys[0])
                entry.values[layer_index, target].copy_(layer.values[0])
            entry.length = new_length

    def free(self, request_id: str) -> bool:
        """幂等释放请求；正在使用的 tensor 延迟到租约退出后销毁。"""

        with self._lock:
            entry = self._entries.get(request_id)
            if entry is None or entry.released:
                return False
            entry.released = True
            if entry.users == 0:
                del self._entries[request_id]
            return True

    def acquire(self, request_ids: tuple[str, ...]) -> KVCacheLease:
        request_ids = tuple(request_ids)
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("reserved request IDs must be unique")
        with self._lock:
            entries = tuple(self._get_active_entry(request_id) for request_id in request_ids)
            for entry in entries:
                entry.users += 1
            return _ContiguousKVCacheLease(self, request_ids, entries)

    def _release_lease(
        self,
        request_ids: tuple[str, ...],
        entries: tuple[_CacheEntry, ...],
    ) -> None:
        with self._lock:
            for request_id, entry in zip(request_ids, entries, strict=True):
                entry.users -= 1
                if entry.released and entry.users == 0:
                    self._entries.pop(request_id, None)

    def _validate_updates(self, updates: KVCacheState) -> None:
        if len(updates.layers) != self._spec.num_layers:
            raise KVCacheError("KV cache update layer count does not match the cache spec")
        for layer in updates.layers:
            expected_shape = (
                1,
                updates.num_tokens,
                self._spec.num_kv_heads,
                self._spec.head_size,
            )
            if layer.keys.shape != expected_shape:
                raise KVCacheError(f"KV cache update must have shape {expected_shape}")
            if layer.keys.dtype != self._spec.dtype or layer.keys.device != self._spec.device:
                raise KVCacheError("KV cache update dtype and device must match the cache spec")

    def _get_entry(self, request_id: str) -> _CacheEntry:
        entry = self._entries.get(request_id)
        if entry is None or (entry.released and entry.users == 0):
            raise KVCacheNotFoundError(f"KV cache for request {request_id!r} was not found")
        return entry

    def _get_active_entry(self, request_id: str) -> _CacheEntry:
        entry = self._get_entry(request_id)
        if entry.released:
            raise KVCacheNotFoundError(f"KV cache for request {request_id!r} was released")
        return entry
