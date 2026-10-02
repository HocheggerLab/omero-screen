"""Size-bounded LRU cache layered over a zarr 3 store.

zarr 2 shipped ``zarr.LRUStoreCache``; zarr 3 has no equivalent. Both the
napari display path and the crop API rely on one: without it every timepoint
scrub, re-zoom or overlapping crop re-reads its chunks from disk (see the
cache-size notes in :mod:`reader` and :mod:`crop`).

Like ``LRUStoreCache``, this caches the raw values the store returns (encoded
chunk bytes and metadata documents) — not decoded arrays — so the size bound
is in on-disk bytes. Only whole-value reads are cached; byte-range reads pass
through, as do all writes (the stores opened here are read-only).
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import TYPE_CHECKING, Self

import zarr
from zarr.storage import LocalStore, WrapperStore

if TYPE_CHECKING:
    from zarr.abc.store import ByteRequest
    from zarr.core.buffer import Buffer, BufferPrototype


class LRUCacheStore(WrapperStore[LocalStore]):  # type: ignore[misc]
    """Read-through LRU cache of store values, bounded by total bytes.

    zarr 3 drives stores from its own event-loop thread while napari and
    the crop API call in from others, so the bookkeeping is lock-guarded.

    Args:
        store: The store to wrap.
        max_size: Upper bound on the summed size of cached values, in bytes.
    """

    def __init__(self, store: LocalStore, max_size: int = 256 * 2**20) -> None:
        super().__init__(store)
        self.max_size = max_size
        self._values: OrderedDict[str, Buffer] = OrderedDict()
        self._current_size = 0
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def _with_store(self, store: LocalStore) -> Self:
        return type(self)(store, max_size=self.max_size)

    async def get(
        self,
        key: str,
        prototype: BufferPrototype,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        """Return the value for ``key``, serving whole-value reads from cache."""
        if byte_range is not None:
            return await self._store.get(key, prototype, byte_range)
        with self._lock:
            cached = self._values.get(key)
            if cached is not None:
                self._values.move_to_end(key)
                self.hits += 1
                return cached
        value = await self._store.get(key, prototype)
        with self._lock:
            self.misses += 1
            if value is not None:
                self._insert(key, value)
        return value

    def _insert(self, key: str, value: Buffer) -> None:
        """Add a value, evicting least-recently-used entries to fit.

        Must be called with ``self._lock`` held. A value larger than the whole
        cache is returned to the caller but not stored.
        """
        size = len(value)
        if size > self.max_size or key in self._values:
            return
        while self._current_size + size > self.max_size:
            _, evicted = self._values.popitem(last=False)
            self._current_size -= len(evicted)
        self._values[key] = value
        self._current_size += size

    def __repr__(self) -> str:
        return (
            f"LRUCacheStore({self._store!r}, max_size={self.max_size}, "
            f"cached={self._current_size})"
        )


def open_cached_group(path: str, max_size: int) -> zarr.Group:
    """Open a zarr group read-only behind a bounded LRU value cache.

    Args:
        path: Filesystem path of the group (e.g. a ``plate_<id>.zarr`` root).
        max_size: Cache bound in bytes.

    Returns:
        The opened group; all arrays reached through it share the cache.
    """
    store = LRUCacheStore(LocalStore(path, read_only=True), max_size=max_size)
    return zarr.open_group(store=store, mode="r")
