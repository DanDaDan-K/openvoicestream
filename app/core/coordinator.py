"""Backend execution coordinator.

execution_policy in profile JSON drives this:
- concurrent  : no lock, ASR and TTS run in parallel
- serialized  : single asyncio.Lock shared by both slots; mutually exclusive
- exclusive   : same lock + slot tracking; switching slot calls dormant
                backend.unload() before yielding. Best-effort: backends not
                overriding unload() stay resident.
"""
from __future__ import annotations
import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator, Callable, Dict, Literal, Optional

Slot = Literal["asr", "tts"]


class BackendCoordinator:
    def __init__(self, policy: dict):
        self._mode = policy.get("mode", "concurrent")
        self._lock: Optional[asyncio.Lock] = None
        if self._mode in ("serialized", "exclusive"):
            self._lock = asyncio.Lock()
        self._active_slot: Optional[Slot] = None
        # store callables to fetch backends lazily (set after services start)
        self._backend_getters: Dict[Slot, Callable] = {}

    @property
    def mode(self) -> str:
        return self._mode

    def register_backend(self, slot: Slot, getter: Callable):
        """Register a callable returning the currently-loaded backend for the slot."""
        self._backend_getters[slot] = getter

    @asynccontextmanager
    async def acquire(self, slot: Slot) -> AsyncIterator[None]:
        if self._mode == "concurrent" or self._lock is None:
            yield
            return
        async with self._lock:
            if self._mode == "exclusive" and self._active_slot not in (None, slot):
                # unload the previously active slot's backend if available
                other = self._active_slot
                getter = self._backend_getters.get(other)
                if getter is not None:
                    backend = getter()
                    if backend is not None and hasattr(backend, "unload"):
                        backend.unload()
            self._active_slot = slot
            yield


_coordinator: Optional[BackendCoordinator] = None


def init_coordinator(policy: dict) -> BackendCoordinator:
    global _coordinator
    _coordinator = BackendCoordinator(policy)
    return _coordinator


def get_coordinator() -> BackendCoordinator:
    if _coordinator is None:
        raise RuntimeError("coordinator not initialized; call init_coordinator() at startup")
    return _coordinator
