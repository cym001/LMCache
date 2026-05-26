# SPDX-License-Identifier: Apache-2.0
# Standard
from concurrent.futures import Future
from unittest.mock import MagicMock
import queue
import threading
import time

# Third Party
import pytest

# First Party
from lmcache.v1.cache_engine import LMCacheEngine, _KvTransferJob
from lmcache.v1.kv_transfer_status import (
    KV_TRANSFER_ALREADY_SATISFIED,
    KV_TRANSFER_FAILED,
)
from lmcache.v1.storage_backend.kv_transfer_backend import (
    KV_TRANSFER_MEM_INDEX_UNAVAILABLE,
    KvTransferPeerResult,
)


def _make_transfer_job(result: Future[int] | None = None) -> _KvTransferJob:
    return _KvTransferJob(
        hashes=[101],
        offsets=[16],
        old_position="LocalCPUBackend",
        peer_ip="127.0.0.1",
        peer_init_port=5555,
        event_id="evt",
        do_copy=True,
        token_ids=None,
        result=result or Future(),
    )


def _make_memory_obj() -> MagicMock:
    memory_obj = MagicMock()
    memory_obj.meta.fmt.token_dim.return_value = 0
    memory_obj.meta.shape = [16]
    return memory_obj


def _make_engine(monkeypatch: pytest.MonkeyPatch) -> tuple[LMCacheEngine, MagicMock]:
    engine = LMCacheEngine.__new__(LMCacheEngine)
    engine._foreground_condition = threading.Condition()
    engine._foreground_ops = 0
    engine._migration_worker_shutdown = False
    engine._migration_queue = queue.Queue(maxsize=8)

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
        "lmcache.v1.cache_engine.asyncio.run_coroutine_threadsafe",
        MagicMock(return_value=transfer_future),
    )

    engine._migration_worker = threading.Thread(
        target=engine._migration_worker_loop,
        daemon=True,
    )
    engine._migration_worker.start()
    return engine, memory_obj


def test_kv_transfer_waits_for_foreground_operation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, memory_obj = _make_engine(monkeypatch)
    result_holder: dict[str, int] = {}

    try:
        engine._enter_foreground_operation()

        worker = threading.Thread(
            target=lambda: result_holder.setdefault(
                "result",
                engine.kv_transfer(
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
        engine._stop_migration_worker()


def test_kv_transfer_releases_memory_obj_on_missing_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, memory_obj = _make_engine(monkeypatch)
    engine.storage_manager.storage_backends = {}

    try:
        result = engine.kv_transfer(
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
        engine._stop_migration_worker()


def test_kv_transfer_move_removes_before_releasing_get_ref(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, memory_obj = _make_engine(monkeypatch)
    order: list[str] = []
    engine.lookup_unpin.side_effect = lambda event_id: order.append("unpin")
    engine.storage_manager.batched_remove.side_effect = (
        lambda keys, locations: order.append("remove")
    )
    memory_obj.ref_count_down.side_effect = lambda: order.append("ref_down")

    try:
        result = engine.kv_transfer(
            hashes=[101],
            offsets=[16],
            old_position="LocalCPUBackend",
            peer_ip="127.0.0.1",
            peer_init_port=5555,
            event_id="evt",
            do_copy=False,
        )

        assert result == 16
        assert order.index("unpin") < order.index("remove")
        assert order.index("remove") < order.index("ref_down")
    finally:
        engine._stop_migration_worker()


def test_kv_transfer_returns_already_satisfied_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, memory_obj = _make_engine(monkeypatch)
    transfer_future: Future[KvTransferPeerResult] = Future()
    transfer_future.set_result(
        KvTransferPeerResult(
            num_read_chunks=0,
            num_existing_chunks=1,
            num_requested_chunks=1,
        )
    )
    monkeypatch.setattr(
        "lmcache.v1.cache_engine.asyncio.run_coroutine_threadsafe",
        MagicMock(return_value=transfer_future),
    )

    try:
        result = engine.kv_transfer(
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
        engine._stop_migration_worker()


def test_kv_transfer_returns_failed_when_peer_satisfies_no_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, memory_obj = _make_engine(monkeypatch)
    transfer_future: Future[KvTransferPeerResult] = Future()
    transfer_future.set_result(
        KvTransferPeerResult(
            num_read_chunks=0,
            num_existing_chunks=0,
            num_requested_chunks=1,
        )
    )
    monkeypatch.setattr(
        "lmcache.v1.cache_engine.asyncio.run_coroutine_threadsafe",
        MagicMock(return_value=transfer_future),
    )

    try:
        result = engine.kv_transfer(
            hashes=[101],
            offsets=[16],
            old_position="LocalCPUBackend",
            peer_ip="127.0.0.1",
            peer_init_port=5555,
            event_id="evt",
        )

        assert result == KV_TRANSFER_FAILED
        memory_obj.ref_count_down.assert_called_once()
    finally:
        engine._stop_migration_worker()


def test_kv_transfer_queue_applies_backpressure() -> None:
    engine = LMCacheEngine.__new__(LMCacheEngine)
    engine._migration_queue = queue.Queue(maxsize=1)

    first_job = _make_transfer_job()
    second_result: Future[int] = Future()
    second_job = _make_transfer_job(second_result)
    engine._migration_queue.put(first_job)
    result_holder: dict[str, int] = {}

    worker = threading.Thread(
        target=lambda: result_holder.setdefault(
            "result", engine._submit_kv_transfer_job(second_job)
        )
    )
    worker.start()
    time.sleep(0.05)

    assert "result" not in result_holder

    queued_first_job = engine._migration_queue.get_nowait()
    queued_first_job.result.set_result(-2)
    engine._migration_queue.task_done()
    queued_second_job = engine._migration_queue.get(timeout=1.0)
    queued_second_job.result.set_result(123)
    engine._migration_queue.task_done()

    worker.join(timeout=1.0)
    assert result_holder["result"] == 123


def test_execute_kv_transfer_passes_full_sequence_put_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = LMCacheEngine.__new__(LMCacheEngine)
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
        "lmcache.v1.cache_engine.asyncio.run_coroutine_threadsafe",
        MagicMock(return_value=transfer_future),
    )

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

    assert engine._execute_kv_transfer_job(job) == 16

    kwargs = kv_backend.transfer_to_peer.call_args.kwargs
    assert kwargs["hashes"] == [101, 202]
    assert kwargs["offsets"] == [16, 16]
    assert kwargs["token_ids"] == list(range(32))
    assert kwargs["mem_indexes"] == [7, KV_TRANSFER_MEM_INDEX_UNAVAILABLE]
    memory_obj.ref_count_down.assert_called_once()
    engine.lookup_unpin.assert_called_once_with("evt")
