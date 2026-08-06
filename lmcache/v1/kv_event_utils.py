# SPDX-License-Identifier: Apache-2.0
"""Utilities for building KV cache store events from token sequences."""

# Standard
from collections.abc import Sequence
from typing import Any, Optional

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey, CacheStoreEvent
from lmcache.v1.plugin.kv_migration import KvBlockMetadata
from lmcache.v1.token_database import TokenDatabase

logger = init_logger(__name__)


def chunk_hash_to_32_bytes(chunk_hash: Any) -> bytes:
    """Normalize LMCache's integer/bytes chunk hash representation."""
    if isinstance(chunk_hash, int):
        return (chunk_hash & ((1 << 256) - 1)).to_bytes(32, "big", signed=False)
    value = bytes(chunk_hash)
    if len(value) > 32:
        return value[-32:]
    return value.rjust(32, b"\x00")


def build_full_sequence_chunk_infos(
    token_database: TokenDatabase,
    token_ids: list[int],
    request_configs: Optional[dict] = None,
) -> list[tuple[int, int, CacheEngineKey, Any | None]]:
    """Build per-chunk metadata with correct parent hashes for a full token sequence."""
    full_hash_infos = list(
        token_database.process_tokens(
            tokens=token_ids,
            mask=None,
            make_key=False,
            request_configs=request_configs,
        )
    )
    parent_by_span: dict[tuple[int, int], Any | None] = {}
    prev_hash: Any | None = None
    for start, end, chunk_hash in full_hash_infos:
        parent_by_span[(start, end)] = prev_hash
        prev_hash = chunk_hash

    infos: list[tuple[int, int, CacheEngineKey, Any | None]] = []
    for start, end, key in token_database.process_tokens(
        tokens=token_ids,
        request_configs=request_configs,
    ):
        assert isinstance(key, CacheEngineKey)
        infos.append((start, end, key, parent_by_span[(start, end)]))
    return infos


def build_stored_block_metadata(
    token_database: TokenDatabase,
    token_ids: list[int],
    stored_keys: Sequence[CacheEngineKey],
    request_configs: Optional[dict] = None,
) -> list[KvBlockMetadata]:
    """Build descriptors for the exact keys that reached the local hot cache."""
    stored_hashes = {key.chunk_hash for key in stored_keys}
    descriptors: list[KvBlockMetadata] = []
    for position, (start, end, key, parent) in enumerate(
        build_full_sequence_chunk_infos(
            token_database,
            token_ids,
            request_configs=request_configs,
        )
    ):
        if key.chunk_hash not in stored_hashes:
            continue
        descriptors.append(
            KvBlockMetadata(
                seq_hash=chunk_hash_to_32_bytes(key.chunk_hash),
                parent_hash=(
                    None if parent is None else chunk_hash_to_32_bytes(parent)
                ),
                position=position,
                offset=end - start,
                token_ids=tuple(token_ids[start:end]),
            )
        )
    return descriptors


def build_full_sequence_store_events(
    token_database: TokenDatabase,
    token_ids: list[int],
    request_configs: Optional[dict] = None,
) -> list[CacheStoreEvent]:
    """Build ordered CacheStoreEvent list for a full sequence starting at root."""
    events: list[CacheStoreEvent] = []
    for start, end, key, parent in build_full_sequence_chunk_infos(
        token_database,
        token_ids,
        request_configs=request_configs,
    ):
        events.append(
            CacheStoreEvent(
                block_hashes=[key.chunk_hash],
                parent_block_hash=parent,
                token_ids=list(token_ids[start:end]),
                block_size=end - start,
                lora_id=None,
                medium="cpu",
                lora_name=None,
            )
        )
    return events


def build_migrated_store_events(
    token_database: TokenDatabase,
    token_ids: list[int],
    migrated_keys: Sequence[CacheEngineKey],
    pre_existing_hashes: set[Any],
    request_configs: Optional[dict] = None,
) -> list[CacheStoreEvent]:
    """Build store events only for newly migrated chunks with valid parent chain."""
    if not migrated_keys:
        return []

    migrated_hashes = {key.chunk_hash for key in migrated_keys}
    available_hashes = set(pre_existing_hashes)
    events: list[CacheStoreEvent] = []

    for start, end, key, parent in build_full_sequence_chunk_infos(
        token_database,
        token_ids,
        request_configs=request_configs,
    ):
        if key.chunk_hash not in migrated_hashes:
            continue

        chunk_tokens = token_ids[start:end]
        if not chunk_tokens:
            logger.warning(
                "Skipping migrated KV store event for empty token_ids slice "
                "at span [%d, %d) for chunk_hash=%s",
                start,
                end,
                key.chunk_hash,
            )
            continue

        if parent is not None and parent not in available_hashes:
            logger.warning(
                "Skipping migrated KV store event and remaining chunks: "
                "parent_block_hash=%s not available for chunk_hash=%s",
                parent,
                key.chunk_hash,
            )
            break

        events.append(
            CacheStoreEvent(
                block_hashes=[key.chunk_hash],
                parent_block_hash=parent,
                token_ids=list(chunk_tokens),
                block_size=end - start,
                lora_id=None,
                medium="cpu",
                lora_name=None,
            )
        )
        available_hashes.add(key.chunk_hash)

    return events


def validate_full_sequence_token_ids(
    token_ids: list[int] | None,
    offsets: list[int],
    event_id: str,
) -> bool:
    """Return True when token_ids cover the full sequence described by offsets."""
    if not token_ids:
        return False

    expected = sum(offsets)
    if len(token_ids) != expected:
        logger.warning(
            "Full-sequence transfer event token_ids length mismatch for "
            "event_id=%s: got=%d expected=%d. Skipping KV event publication.",
            event_id,
            len(token_ids),
            expected,
        )
        return False
    return True
