# SPDX-License-Identifier: Apache-2.0
# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.kv_event_utils import (
    build_full_sequence_store_events,
    build_migrated_store_events,
    build_stored_block_metadata,
    chunk_hash_to_32_bytes,
    validate_full_sequence_token_ids,
)
from lmcache.v1.token_database import ChunkedTokenDatabase
from tests.v1.utils import create_test_config, create_test_metadata


def _make_token_database() -> ChunkedTokenDatabase:
    config = create_test_config(chunk_size=16)
    metadata = create_test_metadata(
        worker_id=3,
        world_size=2,
        kv_shape=(4, 2, 16, 8, 128),
    )
    return ChunkedTokenDatabase(config, metadata)


def _keys_from_tokens(
    token_database: ChunkedTokenDatabase, tokens: list[int]
) -> list[CacheEngineKey]:
    keys: list[CacheEngineKey] = []
    for _, _, key in token_database.process_tokens(tokens=tokens):
        assert isinstance(key, CacheEngineKey)
        keys.append(key)
    return keys


def test_build_migrated_store_events_partial_exist_prefix_local() -> None:
    token_database = _make_token_database()
    tokens = list(range(64))
    full_events = build_full_sequence_store_events(token_database, tokens)
    keys = _keys_from_tokens(token_database, tokens)

    events = build_migrated_store_events(
        token_database,
        tokens,
        migrated_keys=[keys[2], keys[3]],
        pre_existing_hashes={keys[0].chunk_hash, keys[1].chunk_hash},
    )

    assert len(events) == 2
    assert events[0].block_hashes == full_events[2].block_hashes
    assert events[0].parent_block_hash == full_events[1].block_hashes[0]
    assert events[0].parent_block_hash is not None
    assert events[1].parent_block_hash == full_events[2].block_hashes[0]


def test_build_migrated_store_events_parent_from_pre_existing() -> None:
    token_database = _make_token_database()
    tokens = list(range(32))
    full_events = build_full_sequence_store_events(token_database, tokens)
    keys = _keys_from_tokens(token_database, tokens)

    events = build_migrated_store_events(
        token_database,
        tokens,
        migrated_keys=[keys[1]],
        pre_existing_hashes={keys[0].chunk_hash},
    )

    assert len(events) == 1
    assert events[0].block_hashes == full_events[1].block_hashes
    assert events[0].parent_block_hash == full_events[0].block_hashes[0]


def test_build_migrated_store_events_skips_when_parent_missing() -> None:
    token_database = _make_token_database()
    tokens = list(range(48))
    keys = _keys_from_tokens(token_database, tokens)

    events = build_migrated_store_events(
        token_database,
        tokens,
        migrated_keys=[keys[2]],
        pre_existing_hashes=set(),
    )

    assert events == []


def test_build_migrated_store_events_dedup_single_migrated_chunk() -> None:
    token_database = _make_token_database()
    tokens = list(range(48))
    full_events = build_full_sequence_store_events(token_database, tokens)
    keys = _keys_from_tokens(token_database, tokens)

    events = build_migrated_store_events(
        token_database,
        tokens,
        migrated_keys=[keys[2]],
        pre_existing_hashes={keys[0].chunk_hash, keys[1].chunk_hash},
    )

    assert len(events) == 1
    assert events[0].block_hashes == full_events[2].block_hashes


def test_validate_full_sequence_token_ids_rejects_mismatch() -> None:
    assert (
        validate_full_sequence_token_ids(
            token_ids=[1, 2, 3],
            offsets=[16, 16],
            event_id="evt_bad",
        )
        is False
    )


def test_stored_block_metadata_uses_full_parent_chain_for_suffix() -> None:
    token_database = _make_token_database()
    tokens = list(range(64))
    keys = _keys_from_tokens(token_database, tokens)

    descriptors = build_stored_block_metadata(
        token_database,
        tokens,
        stored_keys=[keys[2], keys[3]],
    )

    assert [descriptor.position for descriptor in descriptors] == [2, 3]
    assert descriptors[0].seq_hash == chunk_hash_to_32_bytes(keys[2].chunk_hash)
    assert descriptors[0].parent_hash == chunk_hash_to_32_bytes(keys[1].chunk_hash)
    assert descriptors[0].token_ids == tuple(tokens[32:48])
    assert descriptors[1].parent_hash == descriptors[0].seq_hash


def test_stored_block_metadata_reports_only_actual_keys() -> None:
    token_database = _make_token_database()
    tokens = list(range(48))
    keys = _keys_from_tokens(token_database, tokens)

    descriptors = build_stored_block_metadata(
        token_database,
        tokens,
        stored_keys=[keys[0]],
    )

    assert len(descriptors) == 1
    assert descriptors[0].position == 0
    assert descriptors[0].parent_hash is None
    assert descriptors[0].token_ids == tuple(tokens[:16])
