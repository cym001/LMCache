# SPDX-License-Identifier: Apache-2.0
# Standard
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import MagicMock
import threading
import time

# Third Party
import pytest

# First Party
from lmcache.v1.cache_engine import LMCacheEngine
from lmcache.v1.kv_transfer_status import (
    KV_TRANSFER_ALREADY_SATISFIED,
    KV_TRANSFER_FAILED,
)

from lmcache_kv_transfer.backend import (
    KV_TRANSFER_MEM_INDEX_UNAVAILABLE,
    KvTransferPeerResult,
)
from lmcache_kv_transfer.migration import GlobalKvMigrationPlugin, _KvTransferJob


def _make_test_config() -> SimpleNamespace:
    return SimpleNamespace(
        enable_kv_transfer=False,
        enable_globalkv_server=False,
        get_extra_config_value=lambda key, default: default,
    )


def _make_memory_obj() -> MagicMock:
    memory_obj = MagicMock()
    memory_obj.meta.fmt.token_dim.return_value = 0
    memory_obj.meta.shape = [16]
    return memory_obj


def _make_plugin_with_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[GlobalKvMigrationPlugin, LMCacheEngine, MagicMock]:
    engine = LMCacheEngine.__new__(LMCacheEngine)
    engine._foreground_condition = threading.Condition()
    engine._foreground_ops = 0
    engine.metadata = MagicMock(worker_id=0)

    key = MagicMock()
    key.chunk_hash = 101
    memory_obj = _make_memory_obj()
    storage_manager = MagicMock()
    storage_manager.loop = object()
    storage_manager.batched_get.return_value = [memory_obj]
    storage_manager.storage_backends = {"KvTransferBackend": MagicMock()}
    kv_backend = storage_manager.storage_backends["KvTransferBackend"]
    kv_backend.transfer_channel.get_local_mem_indices.return_value = [7]
    kv_backend.transfer_to_peer = MagicMock(return_value=object())
    engine.storage_manager = storage_manager
    engine.lookup_pins = {"evt": {"LocalCPUBackend": [key]}}
    engine.lookup = MagicMock(return_value=16)
    engine.lookup_unpin = MagicMock()

    transfer_future: Future[KvTransferPeerResult] = Future()
    transfer_future.set_result(
        KvTransferPeerResult(
            num_read_chunks=1,
            num_existing_chunks=0,
            num_requested_chunks=1,
        )
    )
    monkeypatch.setattr(
        "lmcache_kv_transfer.migration.asyncio.run_coroutine_threadsafe",
        MagicMock(return_value=transfer_future),
    )

    plugin = GlobalKvMigrationPlugin(config=_make_test_config())
    plugin.start(engine)
    return plugin, engine, memory_obj


def test_kv_transfer_waits_for_foreground_operation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin, engine, memory_obj = _make_plugin_with_engine(monkeypatch)
    result_holder: dict[str, int] = {}

    try:
        engine._enter_foreground_operation()

        worker = threading.Thread(
            target=lambda: result_holder.setdefault(
                "result",
                plugin.transfer(
                    hashes=[101],
                    offsets=[16],
                    old_position="LocalCPUBackend",
                    peer_ip="127.0.0.1",
                    peer_init_port=5555,
                    event_id="evt",
                ),
            )
        )
        worker.start()
        time.sleep(0.05)

        engine.lookup.assert_not_called()
        assert "result" not in result_holder

        engine._exit_foreground_operation()
        worker.join(timeout=1.0)

        assert result_holder["result"] == 16
        memory_obj.ref_count_down.assert_called_once()
    finally:
        plugin.stop()


def test_kv_transfer_releases_memory_obj_on_missing_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin, engine, memory_obj = _make_plugin_with_engine(monkeypatch)
    engine.storage_manager.storage_backends = {}

    try:
        result = plugin.transfer(
            hashes=[101],
            offsets=[16],
            old_position="LocalCPUBackend",
            peer_ip="127.0.0.1",
            peer_init_port=5555,
            event_id="evt",
        )

        assert result == KV_TRANSFER_FAILED
        memory_obj.ref_count_down.assert_called_once()
        engine.lookup_unpin.assert_called_once_with("evt")
    finally:
        plugin.stop()


def test_execute_kv_transfer_passes_full_sequence_put_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = LMCacheEngine.__new__(LMCacheEngine)
    engine._foreground_condition = threading.Condition()
    engine._foreground_ops = 0
    engine.metadata = MagicMock(worker_id=0)

    key = MagicMock()
    key.chunk_hash = 101
    memory_obj = _make_memory_obj()
    storage_manager = MagicMock()
    storage_manager.loop = object()
    storage_manager.batched_get.return_value = [memory_obj]
    kv_backend = MagicMock()
    kv_backend.transfer_channel.get_local_mem_indices.return_value = [7]
    storage_manager.storage_backends = {"KvTransferBackend": kv_backend}
    engine.storage_manager = storage_manager
    engine.lookup_pins = {"evt": {"LocalCPUBackend": [key]}}
    engine.lookup = MagicMock(return_value=16)
    engine.lookup_unpin = MagicMock()

    transfer_future: Future[KvTransferPeerResult] = Future()
    transfer_future.set_result(
        KvTransferPeerResult(
            num_read_chunks=1,
            num_existing_chunks=0,
            num_requested_chunks=2,
        )
    )
    monkeypatch.setattr(
        "lmcache_kv_transfer.migration.asyncio.run_coroutine_threadsafe",
        MagicMock(return_value=transfer_future),
    )

    plugin = GlobalKvMigrationPlugin(config=_make_test_config())
    plugin._engine = engine

    job = _KvTransferJob(
        hashes=[101, 202],
        offsets=[16, 16],
        old_position="LocalCPUBackend",
        peer_ip="127.0.0.1",
        peer_init_port=5555,
        event_id="evt",
        do_copy=True,
        token_ids=list(range(32)),
        result=Future(),
    )

    assert plugin._execute_kv_transfer_job(job) == 16

    kwargs = kv_backend.transfer_to_peer.call_args.kwargs
    assert kwargs["hashes"] == [101, 202]
    assert kwargs["offsets"] == [16, 16]
    assert kwargs["token_ids"] == list(range(32))
    assert kwargs["mem_indexes"] == [7, KV_TRANSFER_MEM_INDEX_UNAVAILABLE]
    memory_obj.ref_count_down.assert_called_once()


def test_kv_transfer_returns_already_satisfied_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin, engine, memory_obj = _make_plugin_with_engine(monkeypatch)
    transfer_future: Future[KvTransferPeerResult] = Future()
    transfer_future.set_result(
        KvTransferPeerResult(
            num_read_chunks=0,
            num_existing_chunks=1,
            num_requested_chunks=1,
        )
    )
    monkeypatch.setattr(
        "lmcache_kv_transfer.migration.asyncio.run_coroutine_threadsafe",
        MagicMock(return_value=transfer_future),
    )

    try:
        result = plugin.transfer(
            hashes=[101],
            offsets=[16],
            old_position="LocalCPUBackend",
            peer_ip="127.0.0.1",
            peer_init_port=5555,
            event_id="evt",
        )

        assert result == KV_TRANSFER_ALREADY_SATISFIED
        memory_obj.ref_count_down.assert_called_once()
    finally:
        plugin.stop()
