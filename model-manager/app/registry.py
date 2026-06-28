import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .config import settings
from .redis_client import redis_client
from .vllm_client import VLLMError, vllm_client

logger = logging.getLogger("model_manager.registry")


class AdapterState(str, Enum):
    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    UNLOADING = "unloading"


class NotFoundError(Exception):
    pass


class AlreadyExistsError(Exception):
    pass


class CapacityError(Exception):
    """Raised when no adapter can be evicted to make room (all busy past the
    drain deadline). Maps to a 503 at the API layer."""


class ColdStartTimeoutError(Exception):
    pass


@dataclass
class Adapter:
    id: str
    path: str
    state: AdapterState
    pinned: bool = False
    rank: Optional[int] = None
    description: Optional[str] = None
    last_used: float = field(default_factory=time.time)
    in_flight: int = 0


class Registry:
    """Single source of truth for which LoRA adapters exist, which are
    currently loaded into vLLM, and enforces admission control (capacity +
    LRU eviction + drain-before-evict).

    Concurrency model: ALL state transitions (load, unload, evict) and the
    in-flight claim that goes with them are serialized behind one
    asyncio.Lock, held for the full duration of each transition (including
    the network call to vLLM and any drain wait). This is deliberately
    simple rather than maximally concurrent: an earlier per-adapter-lock +
    separate-capacity-lock design could deadlock (the two locks were
    acquired in opposite order on the load path vs. the evict path) and had
    a gap between "confirmed loaded" and "in-flight claimed" that eviction
    could race into. A single lock closes both holes. The cost is that an
    unrelated, already-loaded model's request briefly queues behind a slow
    load/evict of some other model -- acceptable on one GPU at modest
    concurrency; revisit if this ever needs to scale beyond that.

    Also why this must run as a single process/worker: the lock and the
    registry state are in-memory, not safe to share across multiple Model
    Manager replicas without externalizing both.
    """

    def __init__(self):
        self._adapters: dict[str, Adapter] = {}
        self._lock = asyncio.Lock()

    def _get(self, adapter_id: str) -> Adapter:
        adapter = self._adapters.get(adapter_id)
        if adapter is None:
            raise NotFoundError(adapter_id)
        return adapter

    # ---- catalog persistence (sync, no internal await -> always atomic
    # with respect to other coroutines, no lock needed) ---------------------

    def _load_catalog_from_disk(self) -> None:
        if not os.path.exists(settings.registry_file):
            return
        try:
            with open(settings.registry_file) as f:
                catalog = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("could not read registry file, starting empty: %s", exc)
            return
        for entry in catalog:
            self._adapters[entry["id"]] = Adapter(
                id=entry["id"],
                path=entry["path"],
                rank=entry.get("rank"),
                description=entry.get("description"),
                state=AdapterState.UNLOADED,
            )

    async def _sync_adapter_to_redis(self, adapter: "Adapter") -> None:
        key = f"adapter:{adapter.id}"
        mapping = {
            "id": adapter.id,
            "path": adapter.path,
            "state": adapter.state.value,
            "pinned": str(adapter.pinned),
            "rank": str(adapter.rank) if adapter.rank is not None else "",
            "description": adapter.description or "",
            "last_used": str(adapter.last_used),
            "in_flight": str(adapter.in_flight),
        }
        await redis_client.hset(key, mapping, ttl=settings.redis_registry_ttl)

    async def _load_adapter_from_redis(self, adapter_id: str) -> Optional["Adapter"]:
        key = f"adapter:{adapter_id}"
        data = await redis_client.hgetall(key)
        if not data:
            return None
        try:
            return Adapter(
                id=data.get("id"),
                path=data.get("path"),
                state=AdapterState(data.get("state", "unloaded")),
                pinned=data.get("pinned") == "True",
                rank=int(data["rank"]) if data.get("rank") else None,
                description=data.get("description") or None,
                last_used=float(data.get("last_used", time.time())),
                in_flight=int(data.get("in_flight", "0")),
            )
        except (ValueError, KeyError) as exc:
            logger.warning("failed to load adapter from redis: %s", exc)
            return None

    def _persist(self) -> None:
        catalog = [
            {
                "id": a.id,
                "path": a.path,
                "rank": a.rank,
                "description": a.description,
            }
            for a in self._adapters.values()
            if not a.pinned
        ]
        tmp_path = settings.registry_file + ".tmp"
        os.makedirs(os.path.dirname(settings.registry_file), exist_ok=True)
        with open(tmp_path, "w") as f:
            json.dump(catalog, f, indent=2)
        os.replace(tmp_path, settings.registry_file)

    async def register(
        self,
        adapter_id: str,
        path: str,
        rank: Optional[int] = None,
        description: Optional[str] = None,
    ) -> Adapter:
        if adapter_id in self._adapters:
            raise AlreadyExistsError(adapter_id)
        full_path = path if os.path.isabs(path) else os.path.join(settings.lora_dir, path)
        adapter = Adapter(
            id=adapter_id,
            path=full_path,
            rank=rank,
            description=description,
            state=AdapterState.UNLOADED,
        )
        self._adapters[adapter_id] = adapter
        self._persist()
        await self._sync_adapter_to_redis(adapter)
        return adapter

    def list(self) -> list[Adapter]:
        return sorted(self._adapters.values(), key=lambda a: (not a.pinned, a.id))

    # ---- startup reconciliation -----------------------------------------------

    async def reconcile(self) -> None:
        """Load the known catalog from disk, then ask vLLM what's *actually*
        loaded right now and trust that over any stale assumption -- covers
        Model Manager restarts as well as vLLM having been restarted
        independently. Runs once at startup before traffic is accepted, so
        no locking needed here."""
        self._load_catalog_from_disk()
        base_adapter = Adapter(
            id=settings.base_model_id,
            path=settings.base_model,
            state=AdapterState.LOADED,
            pinned=True,
            description="base model (always resident)",
        )
        self._adapters[settings.base_model_id] = base_adapter
        await self._sync_adapter_to_redis(base_adapter)
        try:
            live = await vllm_client.list_models()
            live_ids = {m["id"] for m in live.get("data", [])}
        except Exception as exc:
            logger.warning("could not reach vLLM during reconcile: %s", exc)
            live_ids = set()
        for adapter_id, adapter in self._adapters.items():
            if adapter.pinned:
                continue
            adapter.state = (
                AdapterState.LOADED if adapter_id in live_ids else AdapterState.UNLOADED
            )
            if adapter.state == AdapterState.LOADED:
                adapter.last_used = time.time()
            await self._sync_adapter_to_redis(adapter)
        logger.info(
            "reconciled %d known adapters (%d currently loaded)",
            len(self._adapters) - 1,
            sum(
                1
                for a in self._adapters.values()
                if a.state == AdapterState.LOADED and not a.pinned
            ),
        )

    # ---- in-flight accounting ----------------------------------------------
    # release() is intentionally lock-free: it only ever decrements a
    # per-adapter counter, which is what the drain-wait loop below polls
    # while *holding* the lock -- release() must be able to make progress
    # without contending for it.

    def release(self, adapter_id: str) -> None:
        adapter = self._adapters.get(adapter_id)
        if adapter is not None:
            adapter.in_flight = max(0, adapter.in_flight - 1)
            adapter.last_used = time.time()

    # ---- core lifecycle: load / ensure_loaded / unload ---------------------

    async def ensure_loaded(self, adapter_id: str, claim: bool = False) -> Adapter:
        """Make sure `adapter_id` is loaded into vLLM, evicting the LRU
        adapter first if at capacity. If `claim` is set, atomically marks an
        in-flight request against it in the same critical section that
        confirms it's loaded -- so a caller that gets a successful return is
        guaranteed the adapter can't be evicted out from under it before it
        calls release(). Admin-triggered loads (no request attached) pass
        claim=False."""
        async with self._lock:
            adapter = self._get(adapter_id)
            if adapter.pinned or adapter.state == AdapterState.LOADED:
                if claim:
                    adapter.in_flight += 1
                adapter.last_used = time.time()
                return adapter

            await self._make_room_for(adapter_id)
            adapter.state = AdapterState.LOADING
            await self._sync_adapter_to_redis(adapter)
            try:
                await asyncio.wait_for(
                    vllm_client.load_lora_adapter(adapter.id, adapter.path),
                    timeout=settings.cold_start_timeout_seconds,
                )
            except asyncio.TimeoutError as exc:
                adapter.state = AdapterState.UNLOADED
                await self._sync_adapter_to_redis(adapter)
                raise ColdStartTimeoutError(adapter_id) from exc
            except VLLMError:
                adapter.state = AdapterState.UNLOADED
                await self._sync_adapter_to_redis(adapter)
                raise
            adapter.state = AdapterState.LOADED
            if claim:
                adapter.in_flight += 1
            adapter.last_used = time.time()
            await self._sync_adapter_to_redis(adapter)
            return adapter

    async def load(self, adapter_id: str) -> Adapter:
        """Explicit admin load -- same path as cold-start, just without
        claiming an in-flight slot. wake() is an alias of this."""
        return await self.ensure_loaded(adapter_id, claim=False)

    async def unload(self, adapter_id: str) -> Adapter:
        """Explicit admin unload -- drains in-flight requests first.
        sleep() is an alias of this; LoRA adapters are cheap enough that
        there's no meaningful intermediate "asleep but cached" state."""
        async with self._lock:
            adapter = self._get(adapter_id)
            if adapter.pinned:
                raise ValueError("cannot unload the pinned base model")
            if adapter.state != AdapterState.LOADED:
                return adapter
            deadline = time.time() + settings.drain_timeout_seconds
            ok = await self._drain_and_unload(adapter, deadline)
            if not ok:
                raise ColdStartTimeoutError(adapter_id)
            await self._sync_adapter_to_redis(adapter)
            return adapter

    async def forget(self, adapter_id: str) -> None:
        async with self._lock:
            adapter = self._get(adapter_id)
            if adapter.pinned:
                raise ValueError("cannot remove the pinned base model")
            if adapter.state == AdapterState.LOADED:
                deadline = time.time() + settings.drain_timeout_seconds
                ok = await self._drain_and_unload(adapter, deadline)
                if not ok:
                    raise ColdStartTimeoutError(adapter_id)
            del self._adapters[adapter_id]
        self._persist()
        await redis_client.delete(f"adapter:{adapter_id}")

    # ---- internals. Both require the caller to already hold self._lock. ---

    async def _make_room_for(self, adapter_id: str) -> None:
        loaded = [
            a
            for a in self._adapters.values()
            if a.state == AdapterState.LOADED and not a.pinned and a.id != adapter_id
        ]
        if len(loaded) < settings.adapter_capacity:
            return
        candidates = sorted(loaded, key=lambda a: a.last_used)
        deadline = time.time() + settings.drain_timeout_seconds
        for victim in candidates:
            if await self._drain_and_unload(victim, deadline):
                return
        raise CapacityError(
            f"adapter capacity ({settings.adapter_capacity}) full and no adapter "
            f"could be drained within {settings.drain_timeout_seconds}s"
        )

    async def _drain_and_unload(self, victim: Adapter, deadline: float) -> bool:
        while victim.in_flight > 0:
            if time.time() >= deadline:
                logger.warning(
                    "eviction of '%s' aborted: still %d in-flight after drain deadline",
                    victim.id,
                    victim.in_flight,
                )
                return False
            await asyncio.sleep(settings.drain_poll_interval_seconds)
        victim.state = AdapterState.UNLOADING
        await self._sync_adapter_to_redis(victim)
        try:
            await vllm_client.unload_lora_adapter(victim.id)
        except VLLMError as exc:
            logger.warning("failed to unload '%s' during eviction: %s", victim.id, exc)
            victim.state = AdapterState.LOADED
            await self._sync_adapter_to_redis(victim)
            return False
        victim.state = AdapterState.UNLOADED
        await self._sync_adapter_to_redis(victim)
        logger.info("evicted LRU adapter '%s'", victim.id)
        return True


registry = Registry()
