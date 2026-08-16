"""KV cache 的逻辑分配与本地连续张量存储。"""

from __future__ import annotations

from dataclasses import dataclass
from heapq import heapify, heappop, heappush
from threading import RLock
from typing import Protocol

import torch
from torch import Tensor

from light_vllm.modeling.attention.interfaces import ModelKVCacheSpec


class KVCacheError(RuntimeError):
    """KV cache 状态违反契约时抛出。"""


class KVCacheCapacityError(KVCacheError):
    """没有足够的 KV slot 完成本轮预留时抛出。"""


class KVCacheNotFoundError(KVCacheError):
    """访问不存在或已经释放的请求缓存时抛出。"""


@dataclass(frozen=True, slots=True)
class ContiguousLayerKV:
    """连续布局中一层 K/V 的有效片段。

    K 和 V 都使用 ``[batch, sequence, kv_heads, head_size]``；它既可以是
    已缓存历史的只读视图，也可以是本轮等待追加的新片段。
    """

    keys: Tensor
    values: Tensor

    def __post_init__(self) -> None:
        if self.keys.ndim != 4:
            raise ValueError("keys must have shape [batch, sequence, kv_heads, head_size]")
        if self.values.shape != self.keys.shape:
            raise ValueError("keys and values must have the same shape")
        if self.keys.shape[0] <= 0 or self.keys.shape[2] <= 0 or self.keys.shape[3] <= 0:
            raise ValueError("key/value batch, head count and head size must be positive")
        if self.values.dtype != self.keys.dtype or self.values.device != self.keys.device:
            raise ValueError("keys and values must use the same dtype and device")


@dataclass(frozen=True, slots=True)
class ContiguousKVCacheState:
    """连续缓存中同一段 token 对应的逐层 K/V。"""

    layers: tuple[ContiguousLayerKV, ...]

    def __post_init__(self) -> None:
        layers = tuple(self.layers)
        if not layers:
            raise ValueError("contiguous KV cache state must contain at least one layer")
        batch_and_sequence = layers[0].keys.shape[:2]
        if any(layer.keys.shape[:2] != batch_and_sequence for layer in layers[1:]):
            raise ValueError("all contiguous KV layers must share batch and sequence dimensions")
        object.__setattr__(self, "layers", layers)

    @property
    def num_tokens(self) -> int:
        return self.layers[0].keys.shape[1]


@dataclass(frozen=True, slots=True)
class KVCacheReservation:
    """Scheduler 为一次模型计算提前留出的 KV cache 空间。

    分页 manager 返回请求当前完整的 ``block_ids``；其中可能包含刚为本轮
    增加、但尚未提交的 block。连续缓存不需要 block table，可以返回 ``None``。
    执行成功后只保留实际提交 token 覆盖的资源。
    """

    block_ids: tuple[int, ...] | None
    num_committed_tokens: int
    num_reserved_tokens: int


class KVCacheManager(Protocol):
    """Scheduler 用来预留、确认和释放 KV cache 空间的接口。

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


class KVBlockCapacity(Protocol):
    """让 Scheduler 和实际分页缓存共用同一份页数与页大小。"""

    @property
    def num_blocks(self) -> int: ...

    @property
    def block_size(self) -> int: ...


@dataclass(frozen=True, slots=True)
class FixedKVBlockCapacity:
    """直接指定分页缓存的页数和每页 token 数，主要用于配置和测试。"""

    num_blocks: int
    block_size: int

    def __post_init__(self) -> None:
        if type(self.num_blocks) is not int or self.num_blocks <= 0:
            raise ValueError("num_blocks must be a positive integer")
        if type(self.block_size) is not int or self.block_size <= 0:
            raise ValueError("block_size must be a positive integer")


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

    它只为 Scheduler 分配 page ID，不保存真正的 K/V 张量。
    分页执行 Handler 使用这里生成的 block table 访问实际物理页。
    """

    def __init__(self, capacity: KVBlockCapacity) -> None:
        self._capacity = capacity
        self._num_blocks: int | None = None
        self._free_blocks: list[int] = []
        self._allocations: dict[str, _LogicalAllocation] = {}

    @property
    def block_size(self) -> int:
        return self._capacity.block_size

    @property
    def num_free_blocks(self) -> int:
        self._sync_capacity()
        return len(self._free_blocks)

    def add_request(self, request_id: str) -> None:
        self._sync_capacity()
        if not request_id:
            raise ValueError("request_id must not be empty")
        if request_id in self._allocations:
            raise KVCacheError(f"KV cache for request {request_id!r} already exists")
        self._allocations[request_id] = _LogicalAllocation(block_ids=[])

    def reserve(self, request_id: str, num_tokens: int) -> KVCacheReservation:
        """原子地预留本轮 token 可能占用的新 block。"""

        self._sync_capacity()
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
        return (num_tokens + self.block_size - 1) // self.block_size

    def _sync_capacity(self) -> None:
        """Step Handler 规划完成后初始化；模型重载且空闲时允许重新规划。"""

        num_blocks = self._capacity.num_blocks
        if self._num_blocks == num_blocks:
            return
        if self._allocations:
            raise KVCacheError("cannot change paged KV capacity with active requests")
        self._num_blocks = num_blocks
        self._free_blocks = list(range(num_blocks))
        heapify(self._free_blocks)

    def _get(self, request_id: str) -> _LogicalAllocation:
        try:
            return self._allocations[request_id]
        except KeyError as exc:
            raise KVCacheNotFoundError(
                f"KV cache for request {request_id!r} was not found"
            ) from exc


@dataclass(frozen=True, slots=True)
class ContiguousKVCacheConfig:
    """连续 KV tensor 的设备和 dtype 配置。"""

    dtype: torch.dtype = torch.float32
    device: str | torch.device = "cpu"

    def __post_init__(self) -> None:
        if not self.dtype.is_floating_point:
            raise ValueError("KV cache dtype must be floating point")
        object.__setattr__(self, "device", torch.device(self.device))


class KVCacheLease(Protocol):
    """保证一次模型计算结束前，对应的物理缓存仍然存在。"""

    def release(self) -> None: ...


@dataclass(slots=True)
class _LayerCacheEntry:
    keys: Tensor
    values: Tensor


@dataclass(slots=True)
class _CacheEntry:
    layers: tuple[_LayerCacheEntry, ...]
    length: int = 0
    users: int = 0
    released: bool = False

    @property
    def capacity(self) -> int:
        return self.layers[0].keys.shape[0]


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

    这是便于验证正确性的非分页实现。Scheduler 不直接访问这些张量，
    Executor 也不负责决定请求之间如何分配 KV cache。
    """

    def __init__(
        self,
        model_spec: ModelKVCacheSpec,
        config: ContiguousKVCacheConfig,
    ) -> None:
        self._model_spec = model_spec
        self._config = config
        self._entries: dict[str, _CacheEntry] = {}
        self._lock = RLock()

    @property
    def model_spec(self) -> ModelKVCacheSpec:
        return self._model_spec

    @property
    def config(self) -> ContiguousKVCacheConfig:
        return self._config

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
            layers = tuple(
                _LayerCacheEntry(
                    keys=torch.empty(
                        (capacity, layer.num_kv_heads, layer.head_size),
                        dtype=self._config.dtype,
                        device=self._config.device,
                    ),
                    values=torch.empty(
                        (capacity, layer.num_kv_heads, layer.head_size),
                        dtype=self._config.dtype,
                        device=self._config.device,
                    ),
                )
                for layer in self._model_spec.layers
            )
            self._entries[request_id] = _CacheEntry(layers=layers)

    def contains(self, request_id: str) -> bool:
        with self._lock:
            entry = self._entries.get(request_id)
            return entry is not None and not entry.released

    def cached_tokens(self, request_id: str) -> int:
        with self._lock:
            return self._get_entry(request_id).length

    def truncate(self, request_id: str, num_cached_tokens: int) -> None:
        """回退有效长度；旧张量会在这些位置再次使用时被覆盖。"""

        if type(num_cached_tokens) is not int or num_cached_tokens < 0:
            raise ValueError("cached token count must be a non-negative integer")
        with self._lock:
            entry = self._get_entry(request_id)
            if num_cached_tokens > entry.length:
                raise KVCacheError("cannot extend contiguous KV cache while truncating")
            entry.length = num_cached_tokens

    def view(self, request_id: str) -> ContiguousKVCacheState:
        """返回当前有效前缀的语义视图，不复制 K/V。"""

        with self._lock:
            entry = self._get_entry(request_id)
            state_layers = tuple(
                ContiguousLayerKV(
                    keys=layer.keys[: entry.length].unsqueeze(0),
                    values=layer.values[: entry.length].unsqueeze(0),
                )
                for layer in entry.layers
            )
            return ContiguousKVCacheState(layers=state_layers)

    @torch.inference_mode()
    def append(self, request_id: str, updates: ContiguousKVCacheState) -> None:
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
                target_layer = entry.layers[layer_index]
                target_layer.keys[target].copy_(layer.keys[0])
                target_layer.values[target].copy_(layer.values[0])
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

    def _validate_updates(self, updates: ContiguousKVCacheState) -> None:
        if len(updates.layers) != len(self._model_spec.layers):
            raise KVCacheError("KV cache update layer count does not match the cache spec")
        for layer, spec in zip(updates.layers, self._model_spec.layers, strict=True):
            expected_shape = (
                1,
                updates.num_tokens,
                spec.num_kv_heads,
                spec.head_size,
            )
            if layer.keys.shape != expected_shape:
                raise KVCacheError(f"KV cache update must have shape {expected_shape}")
            if layer.keys.dtype != self._config.dtype or layer.keys.device != self._config.device:
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
