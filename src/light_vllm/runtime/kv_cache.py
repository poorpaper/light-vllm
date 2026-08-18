"""KV cache 的逻辑分配与本地连续张量存储。"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass
from heapq import heapify, heappop, heappush
from threading import RLock
from typing import Protocol

import torch
from torch import Tensor

from light_vllm.modeling.attention.interfaces import ModelKVCacheSpec

_UNSET_CACHE_EPOCH = object()


class KVCacheError(RuntimeError):
    """KV cache 状态违反契约时抛出。"""


class KVCacheCapacityError(KVCacheError):
    """没有足够的 KV slot 完成本轮预留时抛出。"""


class KVCacheNotFoundError(KVCacheError):
    """访问不存在或已经释放的请求缓存时抛出。"""


@dataclass(frozen=True, slots=True)
class KVCacheStats:
    """KV cache 当前可观测的容量事实。

    分页缓存按已经分配、不能立即回收的 token slot 计数；可淘汰的 prefix
    page 视为可用容量。``claimed_token_slots`` 是已向运行请求承诺、
    但还没有转成实际 block 的容量。连续缓存没有固定上限，
    因此三个字段都返回 ``None``。
    """

    used_token_slots: int | None = None
    claimed_token_slots: int | None = None
    capacity_token_slots: int | None = None

    def __post_init__(self) -> None:
        known = (
            self.used_token_slots is not None,
            self.claimed_token_slots is not None,
            self.capacity_token_slots is not None,
        )
        if len(set(known)) != 1:
            raise ValueError("KV cache usage, claims, and capacity must be known together")
        if self.used_token_slots is None:
            return
        if type(self.used_token_slots) is not int or self.used_token_slots < 0:
            raise ValueError("used_token_slots must be a non-negative integer")
        if type(self.claimed_token_slots) is not int or self.claimed_token_slots < 0:
            raise ValueError("claimed_token_slots must be a non-negative integer")
        if type(self.capacity_token_slots) is not int or self.capacity_token_slots <= 0:
            raise ValueError("capacity_token_slots must be a positive integer")
        if self.used_token_slots + self.claimed_token_slots > self.capacity_token_slots:
            raise ValueError("KV cache usage and claims must not exceed capacity")

    @property
    def usage_ratio(self) -> float | None:
        if self.used_token_slots is None or self.capacity_token_slots is None:
            return None
        return self.used_token_slots / self.capacity_token_slots


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
class KVCacheMatch:
    """新请求可以直接复用的已计算 prompt 前缀。"""

    num_cached_tokens: int = 0


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
    # 这些页已经完整写入，只能读取；本轮写入必须从后面的页开始。
    num_readonly_prefix_blocks: int = 0


class KVCacheManager(Protocol):
    """Scheduler 用来预留、确认和释放 KV cache 空间的接口。

    实现可以返回 block table，也可以只记录 token 数；两者都必须支持提交和释放。
    """

    @property
    def stats(self) -> KVCacheStats: ...

    def try_add_request(
        self,
        request_id: str,
        *,
        token_ids: tuple[int, ...],
        max_num_committed_tokens: int,
        cache_epoch: int | None,
        min_free_token_slots: int = 0,
    ) -> KVCacheMatch | None:
        """原子地固定 prefix 并为请求取得完成容量承诺。

        返回 ``None`` 表示暂时容量不足，manager 不得留下部分状态。
        """

        ...

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


@dataclass(frozen=True, slots=True)
class _PrefixBlockKey:
    parent_digest: bytes
    token_ids: tuple[int, ...]
    digest: bytes


@dataclass(slots=True)
class _LogicalAllocation:
    block_ids: list[int]
    completion_block_limit: int
    prompt_block_keys: tuple[_PrefixBlockKey, ...] = ()
    num_readonly_prefix_blocks: int = 0
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

    @property
    def stats(self) -> KVCacheStats:
        # 连续缓存按请求动态增长，没有可用于计算占用率的固定容量。
        return KVCacheStats()

    def try_add_request(
        self,
        request_id: str,
        *,
        token_ids: tuple[int, ...],
        max_num_committed_tokens: int,
        cache_epoch: int | None,
        min_free_token_slots: int = 0,
    ) -> KVCacheMatch | None:
        if not request_id:
            raise ValueError("request_id must not be empty")
        if not token_ids:
            raise ValueError("token_ids must not be empty")
        if type(max_num_committed_tokens) is not int or max_num_committed_tokens < len(token_ids):
            raise ValueError("max_num_committed_tokens must cover the prompt")
        if type(min_free_token_slots) is not int or min_free_token_slots < 0:
            raise ValueError("min_free_token_slots must be a non-negative integer")
        if request_id in self._allocations:
            raise KVCacheError(f"KV cache for request {request_id!r} already exists")
        self._allocations[request_id] = _UnboundedAllocation()
        return KVCacheMatch()

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

    def __init__(
        self,
        capacity: KVBlockCapacity,
        *,
        enable_prefix_caching: bool = False,
    ) -> None:
        self._capacity = capacity
        self._enable_prefix_caching = enable_prefix_caching
        self._num_blocks: int | None = None
        self._free_blocks: list[int] = []
        self._block_ref_counts: list[int] = []
        self._cached_blocks: dict[_PrefixBlockKey, int] = {}
        self._block_keys: dict[int, _PrefixBlockKey] = {}
        self._evictable_blocks: OrderedDict[int, None] = OrderedDict()
        self._cache_epoch: object = _UNSET_CACHE_EPOCH
        self._allocations: dict[str, _LogicalAllocation] = {}
        self._num_completion_claimed_blocks = 0

    @property
    def block_size(self) -> int:
        return self._capacity.block_size

    @property
    def stats(self) -> KVCacheStats:
        self._sync_capacity()
        assert self._num_blocks is not None
        used_blocks = self._num_blocks - self.num_free_blocks
        return KVCacheStats(
            used_token_slots=used_blocks * self.block_size,
            claimed_token_slots=self._num_completion_claimed_blocks * self.block_size,
            capacity_token_slots=self._num_blocks * self.block_size,
        )

    @property
    def num_free_blocks(self) -> int:
        self._sync_capacity()
        return len(self._free_blocks) + len(self._evictable_blocks)

    def try_add_request(
        self,
        request_id: str,
        *,
        token_ids: tuple[int, ...],
        max_num_committed_tokens: int,
        cache_epoch: int | None,
        min_free_token_slots: int = 0,
    ) -> KVCacheMatch | None:
        self._sync_capacity()
        if not request_id:
            raise ValueError("request_id must not be empty")
        if not token_ids:
            raise ValueError("token_ids must not be empty")
        if any(type(token_id) is not int or token_id < 0 for token_id in token_ids):
            raise ValueError("token_ids must contain non-negative integers")
        if type(max_num_committed_tokens) is not int or max_num_committed_tokens < len(token_ids):
            raise ValueError("max_num_committed_tokens must cover the prompt")
        if type(min_free_token_slots) is not int or min_free_token_slots < 0:
            raise ValueError("min_free_token_slots must be a non-negative integer")
        if request_id in self._allocations:
            raise KVCacheError(f"KV cache for request {request_id!r} already exists")
        if self._enable_prefix_caching and (type(cache_epoch) is not int or cache_epoch < 0):
            raise ValueError("prefix caching requires a non-negative cache epoch")
        self._ensure_cache_epoch(cache_epoch)

        prompt_block_keys = self._prompt_block_keys(token_ids, cache_epoch)
        matched_blocks: list[int] = []
        if self._enable_prefix_caching:
            for block_key in prompt_block_keys:
                block_id = self._cached_blocks.get(block_key)
                if block_id is None:
                    break
                matched_blocks.append(block_id)

        completion_block_limit = self._blocks_for(max_num_committed_tokens)
        new_claims = completion_block_limit - len(matched_blocks)
        newly_pinned_blocks = sum(
            self._block_ref_counts[block_id] == 0 for block_id in matched_blocks
        )
        min_free_blocks = self._blocks_for(min_free_token_slots)
        used_blocks = self._num_blocks - self.num_free_blocks
        assert self._num_blocks is not None
        if (
            used_blocks
            + self._num_completion_claimed_blocks
            + newly_pinned_blocks
            + new_claims
            + min_free_blocks
            > self._num_blocks
        ):
            return None

        # 找齐完整前缀后再统一增加引用，避免中途异常留下半注册状态。
        for block_id in matched_blocks:
            self._acquire_block(block_id)
        num_cached_tokens = len(matched_blocks) * self.block_size
        self._allocations[request_id] = _LogicalAllocation(
            block_ids=matched_blocks,
            completion_block_limit=completion_block_limit,
            prompt_block_keys=prompt_block_keys,
            num_readonly_prefix_blocks=len(matched_blocks),
            num_committed_tokens=num_cached_tokens,
        )
        self._num_completion_claimed_blocks += new_claims
        return KVCacheMatch(num_cached_tokens=num_cached_tokens)

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
        available_claim = allocation.completion_block_limit - len(allocation.block_ids)
        if new_block_count > available_claim:
            raise KVCacheError(f"request {request_id!r} exceeded its KV completion claim")

        new_blocks: list[int] = []
        try:
            for _ in range(new_block_count):
                new_blocks.append(self._allocate_block())
        except Exception:
            for block_id in reversed(new_blocks):
                self._release_block(block_id)
            raise
        allocation.block_ids.extend(new_blocks)
        self._num_completion_claimed_blocks -= new_block_count
        allocation.num_reserved_tokens = num_tokens
        return KVCacheReservation(
            block_ids=tuple(allocation.block_ids),
            num_committed_tokens=allocation.num_committed_tokens,
            num_reserved_tokens=num_tokens,
            num_readonly_prefix_blocks=allocation.num_readonly_prefix_blocks,
        )

    def commit(self, request_id: str, num_tokens: int) -> None:
        """提交预留前缀；未提交的尾部等价于回滚并立即归还 block。"""

        allocation = self._get(request_id)
        if type(num_tokens) is not int or not 0 <= num_tokens <= allocation.num_reserved_tokens:
            raise ValueError("committed token count must be within the active reservation")

        allocation.num_committed_tokens += num_tokens
        allocation.num_reserved_tokens = 0
        self._publish_prompt_blocks(allocation)
        self._trim_blocks(allocation)

    def free(self, request_id: str) -> bool:
        allocation = self._allocations.pop(request_id, None)
        if allocation is None:
            return False
        self._num_completion_claimed_blocks -= allocation.completion_block_limit - len(
            allocation.block_ids
        )
        # 先释放复用概率更低的后缀，避免根页先被 LRU 淘汰后
        # 留下无法从根达到的子页。
        for block_id in reversed(allocation.block_ids):
            self._release_block(block_id)
        return True

    def _trim_blocks(self, allocation: _LogicalAllocation) -> None:
        keep = self._blocks_for(allocation.num_committed_tokens)
        released = allocation.block_ids[keep:]
        del allocation.block_ids[keep:]
        self._num_completion_claimed_blocks += len(released)
        for block_id in reversed(released):
            self._release_block(block_id)

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
        self._reset_block_pool(num_blocks)
        self._cache_epoch = _UNSET_CACHE_EPOCH

    def _ensure_cache_epoch(self, cache_epoch: int | None) -> None:
        if not self._enable_prefix_caching:
            return
        if self._cache_epoch == cache_epoch:
            return
        if self._allocations:
            raise KVCacheError("cannot change prefix cache epoch with active requests")
        if self._num_blocks is None:
            raise KVCacheError("paged KV capacity is not initialized")
        # Worker 换代后物理 tensor 会重建，旧 page ID 即使相同也没有旧 K/V。
        self._reset_block_pool(self._num_blocks)
        self._cache_epoch = cache_epoch

    def _prompt_block_keys(
        self,
        token_ids: tuple[int, ...],
        cache_epoch: int | None,
    ) -> tuple[_PrefixBlockKey, ...]:
        if not self._enable_prefix_caching:
            return ()
        # Prefix cache 只保存完整页，并至少留一个 token 重新得到下一 token 的 logits。
        num_cacheable_tokens = ((len(token_ids) - 1) // self.block_size) * self.block_size
        parent_hash = hashlib.sha256(f"light-vllm:{cache_epoch!r}".encode()).digest()
        block_keys: list[_PrefixBlockKey] = []
        for start in range(0, num_cacheable_tokens, self.block_size):
            block_tokens = token_ids[start : start + self.block_size]
            digest = hashlib.sha256(parent_hash + repr(block_tokens).encode()).digest()
            block_keys.append(
                _PrefixBlockKey(
                    parent_digest=parent_hash,
                    token_ids=block_tokens,
                    digest=digest,
                )
            )
            parent_hash = digest
        return tuple(block_keys)

    def _publish_prompt_blocks(self, allocation: _LogicalAllocation) -> None:
        num_full_prompt_blocks = min(
            allocation.num_committed_tokens // self.block_size,
            len(allocation.prompt_block_keys),
        )
        for index in range(allocation.num_readonly_prefix_blocks, num_full_prompt_blocks):
            block_key = allocation.prompt_block_keys[index]
            block_id = allocation.block_ids[index]
            if block_key not in self._cached_blocks:
                self._cached_blocks[block_key] = block_id
                self._block_keys[block_id] = block_key
        allocation.num_readonly_prefix_blocks = num_full_prompt_blocks

    def _allocate_block(self) -> int:
        if self._free_blocks:
            block_id = heappop(self._free_blocks)
        else:
            try:
                block_id, _ = self._evictable_blocks.popitem(last=False)
            except KeyError as exc:
                raise KVCacheCapacityError("no paged KV block is available") from exc
            block_key = self._block_keys.pop(block_id)
            if self._cached_blocks.get(block_key) == block_id:
                del self._cached_blocks[block_key]
        self._block_ref_counts[block_id] = 1
        return block_id

    def _acquire_block(self, block_id: int) -> None:
        if self._block_ref_counts[block_id] == 0:
            self._evictable_blocks.pop(block_id, None)
        self._block_ref_counts[block_id] += 1

    def _release_block(self, block_id: int) -> None:
        ref_count = self._block_ref_counts[block_id]
        if ref_count <= 0:
            raise KVCacheError("paged KV block reference count is already zero")
        ref_count -= 1
        self._block_ref_counts[block_id] = ref_count
        if ref_count:
            return
        if block_id in self._block_keys:
            # 仍在缓存索引中的页进入 LRU；需要空间时才真正清掉摘要。
            self._evictable_blocks[block_id] = None
        else:
            heappush(self._free_blocks, block_id)

    def _reset_block_pool(self, num_blocks: int) -> None:
        self._free_blocks = list(range(num_blocks))
        heapify(self._free_blocks)
        self._block_ref_counts = [0] * num_blocks
        self._cached_blocks.clear()
        self._block_keys.clear()
        self._evictable_blocks.clear()
        self._num_completion_claimed_blocks = 0

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
