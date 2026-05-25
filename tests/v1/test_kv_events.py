# SPDX-License-Identifier: Apache-2.0
# Standard
from unittest.mock import MagicMock

# Third Party
import torch

# First Party
from lmcache.v1.cache_engine import LMCacheEngine
from lmcache.v1.token_database import ChunkedTokenDatabase
from tests.v1.utils import create_test_config, create_test_metadata


def _make_event_engine(chunk_size: int = 4) -> LMCacheEngine:
    config = create_test_config(chunk_size=chunk_size)
    config.enable_kv_events = True
    metadata = create_test_metadata(
        worker_id=0,
        world_size=1,
        kv_shape=(4, 2, chunk_size, 8, 128),
    )

    engine = LMCacheEngine.__new__(LMCacheEngine)
    engine.config = config
    engine.metadata = metadata
    engine.token_database = ChunkedTokenDatabase(config, metadata)
    engine.kv_events_enabled = True
    engine.kv_events = []
    engine._kv_event_published_hashes = set()
    engine.retrieve_locations = ["LocalCPUBackend"]
    engine.storage_manager = MagicMock()
    engine.use_layerwise = False
    engine._init_failed = False
    engine._health_monitor = None
    engine.stats_monitor = MagicMock()
    engine.stats_monitor.on_lookup_request.return_value = MagicMock()
    return engine


def test_masked_store_event_infos_keep_real_parent_chain() -> None:
    engine = _make_event_engine()
    tokens = list(range(12))
    mask = torch.ones(len(tokens), dtype=torch.bool)
    mask[:8] = False

    full_infos = engine._build_kv_event_chunk_infos(tokens=tokens)
    suffix_infos = engine._build_kv_event_chunk_infos(tokens=tokens, mask=mask)
    existing_prefix_hashes = {
        full_infos[0].key.chunk_hash,
        full_infos[1].key.chunk_hash,
    }
    engine.storage_manager.contains.side_effect = (
        lambda key, _: key.chunk_hash in existing_prefix_hashes
    )

    engine._publish_kv_event_parent_chain(full_infos, suffix_infos[0], tokens=tokens)
    engine._append_kv_store_event(suffix_infos[0], tokens=tokens)

    assert [event.block_hashes[0] for event in engine.kv_events] == [
        full_infos[0].key.chunk_hash,
        full_infos[1].key.chunk_hash,
        suffix_infos[0].key.chunk_hash,
    ]
    assert [event.parent_block_hash for event in engine.kv_events] == [
        None,
        full_infos[0].key.chunk_hash,
        full_infos[1].key.chunk_hash,
    ]


def test_lookup_republishes_hit_prefix_events_after_events_are_drained() -> None:
    engine = _make_event_engine()
    tokens = list(range(12))
    full_infos = engine._build_kv_event_chunk_infos(tokens=tokens)
    keys = [info.key for info in full_infos]
    engine.storage_manager.batched_contains.return_value = (
        2,
        {"LocalCPUBackend": keys[:2]},
    )

    assert engine.lookup(tokens=tokens) == 8
    assert [event.block_hashes[0] for event in engine.kv_events] == [
        full_infos[0].key.chunk_hash,
        full_infos[1].key.chunk_hash,
    ]
    assert [event.parent_block_hash for event in engine.kv_events] == [
        None,
        full_infos[0].key.chunk_hash,
    ]

    assert len(list(engine.get_kv_events())) == 2
    assert engine.lookup(tokens=tokens) == 8
    assert [event.block_hashes[0] for event in engine.kv_events] == [
        full_infos[0].key.chunk_hash,
        full_infos[1].key.chunk_hash,
    ]
