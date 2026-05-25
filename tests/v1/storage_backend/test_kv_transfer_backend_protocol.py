# SPDX-License-Identifier: Apache-2.0
# Standard
from collections import OrderedDict
from unittest.mock import AsyncMock, MagicMock

# Third Party
import msgspec
import pytest
import torch

# First Party
from lmcache.v1.memory_management import MemoryFormat
from lmcache.v1.storage_backend.kv_transfer_backend import (
    BatchedLookupAndGetMsg,
    BatchedLookupAndPutMsg,
    KvTransferBackend,
)
from lmcache.v1.token_database import ChunkedTokenDatabase
from tests.v1.utils import create_test_config, create_test_metadata


def _make_backend_stub(worker_id: int) -> tuple[KvTransferBackend, int]:
    config = create_test_config(chunk_size=16)
    metadata = create_test_metadata(
        worker_id=worker_id,
        world_size=2,
        kv_shape=(4, 2, 16, 8, 128),
    )

    backend = KvTransferBackend.__new__(KvTransferBackend)
    backend.token_database = ChunkedTokenDatabase(config, metadata)
    backend.local_cpu_backend = MagicMock()
    backend.transfer_channel = MagicMock()
    backend.kv_events = None
    backend.full_size_shape = [4, 2, 16, 8, 128]
    backend.fmt = MemoryFormat.KV_2LTD
    backend.dtype = torch.bfloat16
    return backend, metadata.worker_id


def test_kv_transfer_messages_use_hashes_offsets() -> None:
    get_msg = BatchedLookupAndGetMsg(
        event_id="evt_get",
        lookup_id="lookup_1",
        receiver_id="peerA",
        hashes=[11, 22],
        offsets=[16, 16],
        mem_indexes=[1, 2],
    )
    put_msg = BatchedLookupAndPutMsg(
        event_id="evt_put",
        sender_id="peerB",
        hashes=[33, 44],
        offsets=[16, 16],
        mem_indexes=[3, 4],
    )

    get_decoded = msgspec.msgpack.decode(
        msgspec.msgpack.encode(get_msg),
        type=BatchedLookupAndGetMsg,
    )
    put_decoded = msgspec.msgpack.decode(
        msgspec.msgpack.encode(put_msg),
        type=BatchedLookupAndPutMsg,
    )

    assert get_decoded.hashes == [11, 22]
    assert get_decoded.offsets == [16, 16]
    assert put_decoded.hashes == [33, 44]
    assert put_decoded.offsets == [16, 16]
    assert not hasattr(get_decoded, "keys")
    assert not hasattr(put_decoded, "keys")


def test_build_local_keys_uses_local_worker_id() -> None:
    backend, worker_id = _make_backend_stub(worker_id=7)

    keys = backend._build_local_keys_from_hashes_offsets(
        hashes=[1001, 1002],
        offsets=[16, 8],
    )

    assert [key.chunk_hash for key in keys] == [1001, 1002]
    assert [key.worker_id for key in keys] == [worker_id, worker_id]


def test_build_local_keys_accepts_bytes_hashes() -> None:
    backend, worker_id = _make_backend_stub(worker_id=9)
    hashes = [b"\x01" * 32, b"\x02" * 32]

    keys = backend._build_local_keys_from_hashes_offsets(
        hashes=hashes,
        offsets=[16, 16],
    )

    assert [key.chunk_hash for key in keys] == hashes
    assert [key.worker_id for key in keys] == [worker_id, worker_id]


@pytest.mark.anyio
async def test_handle_put_rebuilds_local_keys_before_store() -> None:
    backend, worker_id = _make_backend_stub(worker_id=3)
    backend.local_cpu_backend.contains.return_value = False
    backend.local_cpu_backend.allocate.side_effect = [MagicMock(), MagicMock()]
    backend.local_cpu_backend.batched_submit_put_task = MagicMock()
    backend.transfer_channel.async_batched_read = AsyncMock()

    msg = BatchedLookupAndPutMsg(
        event_id="evt_put",
        sender_id="sender_peer",
        hashes=[5001, 5002],
        offsets=[16, 16],
        mem_indexes=[9, 10],
    )

    ret = await backend._handle_kv_transfer_msg(msg)

    assert ret.num_read_chunks == 2
    assert ret.num_existing_chunks == 0
    called_keys = backend.local_cpu_backend.batched_submit_put_task.call_args.kwargs[
        "keys"
    ]
    assert [k.chunk_hash for k in called_keys] == [5001, 5002]
    assert [k.worker_id for k in called_keys] == [worker_id, worker_id]
    for call in backend.local_cpu_backend.allocate.call_args_list:
        assert call.kwargs["busy_loop"] is False


@pytest.mark.anyio
async def test_handle_put_publishes_per_chunk_kv_events() -> None:
    backend, worker_id = _make_backend_stub(worker_id=3)
    backend.kv_events = []
    backend.local_cpu_backend.contains.return_value = False
    backend.local_cpu_backend.allocate.side_effect = [MagicMock(), MagicMock()]
    backend.local_cpu_backend.batched_submit_put_task = MagicMock()
    backend.transfer_channel.async_batched_read = AsyncMock()

    msg = BatchedLookupAndPutMsg(
        event_id="evt_put",
        sender_id="sender_peer",
        hashes=[5001, 5002],
        offsets=[16, 16],
        mem_indexes=[9, 10],
        token_ids=list(range(101, 117)) + list(range(201, 217)),
    )

    await backend._handle_kv_transfer_msg(msg)

    assert len(backend.kv_events) == 2
    assert [event.block_size for event in backend.kv_events] == [16, 16]
    assert [event.block_hashes for event in backend.kv_events] == [[5001], [5002]]
    assert backend.kv_events[0].parent_block_hash is None
    assert backend.kv_events[1].parent_block_hash == 5001
    assert backend.kv_events[0].token_ids == list(range(101, 117))
    assert backend.kv_events[1].token_ids == list(range(201, 217))


@pytest.mark.anyio
async def test_handle_put_uses_explicit_parent_hashes_for_kv_events() -> None:
    backend, _ = _make_backend_stub(worker_id=3)
    backend.kv_events = []
    backend.local_cpu_backend.contains.return_value = False
    backend.local_cpu_backend.allocate.side_effect = [MagicMock(), MagicMock()]
    backend.local_cpu_backend.batched_submit_put_task = MagicMock()
    backend.transfer_channel.async_batched_read = AsyncMock()

    msg = BatchedLookupAndPutMsg(
        event_id="evt_put_suffix",
        sender_id="sender_peer",
        hashes=[5001, 5002],
        offsets=[16, 16],
        mem_indexes=[9, 10],
        parent_hashes=[4000, 5001],
    )

    await backend._handle_kv_transfer_msg(msg)

    assert len(backend.kv_events) == 2
    assert [event.block_hashes for event in backend.kv_events] == [[5001], [5002]]
    assert [event.parent_block_hash for event in backend.kv_events] == [4000, 5001]


@pytest.mark.anyio
async def test_handle_put_returns_existing_count_when_all_chunks_exist() -> None:
    backend, _ = _make_backend_stub(worker_id=3)
    backend.local_cpu_backend.contains.return_value = True
    backend.local_cpu_backend.batched_submit_put_task = MagicMock()
    backend.transfer_channel.async_batched_read = AsyncMock()

    msg = BatchedLookupAndPutMsg(
        event_id="evt_existing",
        sender_id="sender_peer",
        hashes=[5001, 5002],
        offsets=[16, 16],
        mem_indexes=[9, 10],
    )

    ret = await backend._handle_kv_transfer_msg(msg)

    assert ret.num_read_chunks == 0
    assert ret.num_existing_chunks == 2
    backend.local_cpu_backend.allocate.assert_not_called()
    backend.transfer_channel.async_batched_read.assert_not_called()
    called_keys = backend.local_cpu_backend.batched_submit_put_task.call_args.kwargs[
        "keys"
    ]
    assert called_keys == []


@pytest.mark.anyio
async def test_handle_put_counts_existing_and_reads_missing_chunks() -> None:
    backend, worker_id = _make_backend_stub(worker_id=3)
    new_mem_obj = MagicMock()
    backend.local_cpu_backend.contains.side_effect = [True, False, True]
    backend.local_cpu_backend.allocate.return_value = new_mem_obj
    backend.local_cpu_backend.batched_submit_put_task = MagicMock()
    backend.transfer_channel.async_batched_read = AsyncMock()

    msg = BatchedLookupAndPutMsg(
        event_id="evt_mixed",
        sender_id="sender_peer",
        hashes=[5001, 5002, 5003],
        offsets=[16, 16, 16],
        mem_indexes=[9, 10, 11],
    )

    ret = await backend._handle_kv_transfer_msg(msg)

    assert ret.num_read_chunks == 1
    assert ret.num_existing_chunks == 2
    backend.transfer_channel.async_batched_read.assert_awaited_once()
    called_keys = backend.local_cpu_backend.batched_submit_put_task.call_args.kwargs[
        "keys"
    ]
    assert len(called_keys) == 1
    assert called_keys[0].chunk_hash == 5002
    assert called_keys[0].worker_id == worker_id


@pytest.mark.anyio
async def test_handle_put_partial_when_allocation_fails() -> None:
    backend, worker_id = _make_backend_stub(worker_id=3)
    first_mem_obj = MagicMock()
    backend.local_cpu_backend.contains.return_value = False
    backend.local_cpu_backend.allocate.side_effect = [first_mem_obj, None]
    backend.local_cpu_backend.batched_submit_put_task = MagicMock()
    backend.transfer_channel.async_batched_read = AsyncMock()

    msg = BatchedLookupAndPutMsg(
        event_id="evt_put",
        sender_id="sender_peer",
        hashes=[5001, 5002],
        offsets=[16, 16],
        mem_indexes=[9, 10],
    )

    ret = await backend._handle_kv_transfer_msg(msg)

    assert ret.num_read_chunks == 1
    assert ret.num_existing_chunks == 0
    called_keys = backend.local_cpu_backend.batched_submit_put_task.call_args.kwargs[
        "keys"
    ]
    assert len(called_keys) == 1
    assert called_keys[0].chunk_hash == 5001
    assert called_keys[0].worker_id == worker_id


@pytest.mark.anyio
async def test_handle_get_rebuilds_local_keys_for_lookup() -> None:
    backend, worker_id = _make_backend_stub(worker_id=5)
    backend.local_cpu_backend.batched_async_contains = AsyncMock(return_value=1)
    mem_obj = MagicMock()
    backend.local_cpu_backend.batched_get_non_blocking = AsyncMock(return_value=[mem_obj])
    backend.transfer_channel.async_batched_write = AsyncMock()

    msg = BatchedLookupAndGetMsg(
        event_id="evt_get",
        lookup_id="lookup_1",
        receiver_id="receiver_peer",
        hashes=[7001, 7002],
        offsets=[16, 16],
        mem_indexes=[1, 2],
    )

    ret = await backend._handle_kv_transfer_msg(msg)

    assert ret.num_hit_chunks == 1
    lookup_keys = backend.local_cpu_backend.batched_async_contains.call_args.kwargs[
        "keys"
    ]
    assert [k.chunk_hash for k in lookup_keys] == [7001, 7002]
    assert [k.worker_id for k in lookup_keys] == [worker_id, worker_id]
    mem_obj.ref_count_down.assert_called_once()
    mem_obj.unpin.assert_called_once()


@pytest.mark.anyio
async def test_batched_get_returns_empty_when_allocation_fails() -> None:
    backend, _ = _make_backend_stub(worker_id=6)
    backend.lookup_id_to_peer_mapping = {"lookup_oom": ("peer://a", "cpu")}
    backend.peer_last_used_time = {}
    backend.connected_peers = OrderedDict()
    backend.local_init_url = "local://1"
    backend._send_request_and_wait = AsyncMock()
    backend.local_cpu_backend.allocate.return_value = None

    ret = await backend.batched_get_non_blocking(
        lookup_id="lookup_oom",
        keys=backend._build_local_keys_from_hashes_offsets(
            hashes=[1111, 2222], offsets=[16, 16]
        ),
        transfer_spec={"cum_chunk_lengths": [0, 16, 32]},
    )

    assert ret == []
    backend._send_request_and_wait.assert_not_called()
    alloc_call = backend.local_cpu_backend.allocate.call_args
    assert alloc_call.kwargs["busy_loop"] is False
