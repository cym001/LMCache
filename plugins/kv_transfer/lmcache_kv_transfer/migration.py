# SPDX-License-Identifier: Apache-2.0
"""GlobalKV migration plugin implementation."""

# Standard
from concurrent.futures import Future
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Optional
import asyncio
import queue
import threading
import time

# First Party
from lmcache.logging import init_logger
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.kv_transfer_status import (
    KV_TRANSFER_ALREADY_SATISFIED,
    KV_TRANSFER_FAILED,
    KV_TRANSFER_NOT_FOUND,
)
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.plugin.kv_migration import (
    KvBlockMetadata,
    KvMetadataReporterInterface,
    KvMigrationPluginInterface,
    find_kv_transfer_backend,
)

from lmcache_kv_transfer.backend import (
    KV_TRANSFER_MEM_INDEX_UNAVAILABLE,
    KvTransferBlockResult,
    KvTransferBlockStatus,
    KvTransferPeerResult,
)
from lmcache_kv_transfer.globalkv_server import GlobalKvServer
from lmcache_kv_transfer.metadata_client import KvCacheClient

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.cache_engine import LMCacheEngine

logger = init_logger(__name__)


@dataclass
class _KvTransferJob:
    hashes: List[int | bytes]
    offsets: List[int]
    old_position: str
    peer_ip: str
    peer_init_port: int
    event_id: str
    do_copy: bool
    token_ids: Optional[List[int]]
    compatibility_group_id: bytes
    expected_target_epoch: str
    expected_target_instance_id: str
    expected_target_worker_id: int
    result: Future[KvTransferPeerResult]


@dataclass(frozen=True)
class _MetadataEvent:
    kind: str
    blocks: tuple[KvBlockMetadata, ...] = ()
    hashes: tuple[bytes, ...] = ()
    request_id: str = ""
    revision: int = 0


class KvCacheMetadataReporter(KvMetadataReporterInterface):
    """Adapter that exposes KvCacheClient through the core reporter interface."""

    def __init__(self, client: KvCacheClient) -> None:
        self._client = client
        self._ledger: dict[bytes, KvBlockMetadata] = {}
        self._replica_versions: dict[bytes, int] = {}
        self._ledger_lock = threading.Lock()
        self._ledger_revision = 0
        self._synced_revision = 0
        self._engine: LMCacheEngine | None = None
        queue_size = max(
            1,
            int(
                client.config.get_extra_config_value(
                    "globalkv_metadata_queue_size", 4096
                )
            ),
        )
        self._metadata_queue: queue.Queue[_MetadataEvent | None] = queue.Queue(
            maxsize=queue_size
        )
        self._reporter_shutdown = False
        self._reporter_thread = threading.Thread(
            target=self._reporter_loop,
            name="GlobalKvMetadataReporter",
            daemon=True,
        )
        self._reporter_thread.start()

    def on_kv_stored(self, tokens: list[int]) -> None:
        if self._client.protocol in {"v1", "dual"}:
            self._client.upload_kv_meta(tokens)

    def on_kv_stored_structured(
        self, blocks: list[KvBlockMetadata]
    ) -> dict[bytes, int]:
        if not blocks:
            return {}
        with self._ledger_lock:
            for block in blocks:
                self._ledger[block.seq_hash] = block
            self._ledger_revision += 1
            revision = self._ledger_revision
        if self._client.protocol in {"v1", "dual"}:
            self._client.upload_kv_meta(
                [token for block in blocks for token in block.token_ids]
            )
        if self._client.protocol in {"dual", "v2"}:
            self._enqueue_metadata(
                _MetadataEvent(
                    kind="store", blocks=tuple(blocks), revision=revision
                ),
                mutation=True,
            )
        with self._ledger_lock:
            return {
                block.seq_hash: self._replica_versions.get(block.seq_hash, 0)
                for block in blocks
            }

    def on_kv_stored_structured_sync(
        self, blocks: list[KvBlockMetadata]
    ) -> dict[bytes, int]:
        if not blocks:
            return {}
        with self._ledger_lock:
            for block in blocks:
                self._ledger[block.seq_hash] = block
            self._ledger_revision += 1
        if self._client.protocol in {"v1", "dual"}:
            self._client.upload_kv_meta(
                [token for block in blocks for token in block.token_ids]
            )
        version = 0
        if self._client.protocol in {"dual", "v2"}:
            version = self._client.report_stored_blocks(blocks)
        with self._ledger_lock:
            if version > 0:
                for block in blocks:
                    self._replica_versions[block.seq_hash] = max(
                        version, self._replica_versions.get(block.seq_hash, 0)
                    )
            return {block.seq_hash: version for block in blocks}

    def on_kv_retrieved(self, hit_tokens: list[int]) -> None:
        if self._client.protocol in {"v1", "dual"}:
            self._client.upload_kv_meta(hit_tokens)

    def on_kv_removed(self, chunk_hashes: list[bytes]) -> None:
        with self._ledger_lock:
            for chunk_hash in chunk_hashes:
                self._ledger.pop(chunk_hash, None)
            self._ledger_revision += 1
            revision = self._ledger_revision
        if self._client.protocol in {"v1", "dual"}:
            self._client.remove_kv_meta(chunk_hashes)
        if self._client.protocol in {"dual", "v2"}:
            self._enqueue_metadata(
                _MetadataEvent(
                    kind="remove",
                    hashes=tuple(chunk_hashes),
                    revision=revision,
                ),
                mutation=True,
            )

    def on_request_start(self, request_id: str, tokens: list[int]) -> None:
        if self._client.protocol in {"v1", "dual"}:
            self._client.new_request(request_id, tokens)
        if self._client.protocol in {"dual", "v2"}:
            with self._ledger_lock:
                candidates = list(self._ledger.values())
            matched: list[KvBlockMetadata] = []
            cursor = 0
            parent_hash: bytes | None = None
            while cursor < len(tokens):
                block = next(
                    (
                        candidate
                        for candidate in candidates
                        if candidate.parent_hash == parent_hash
                        and list(candidate.token_ids)
                        == tokens[cursor : cursor + len(candidate.token_ids)]
                    ),
                    None,
                )
                if block is None:
                    break
                matched.append(block)
                cursor += len(block.token_ids)
                parent_hash = block.seq_hash
            self._enqueue_metadata(
                _MetadataEvent(
                    kind="request_start",
                    request_id=request_id,
                    blocks=tuple(matched),
                ),
                mutation=False,
            )

    def on_request_end(self, request_id: str, tokens: list[int]) -> None:
        if self._client.protocol in {"v1", "dual"}:
            self._client.request_end(request_id, tokens)
        if self._client.protocol in {"dual", "v2"}:
            self._enqueue_metadata(
                _MetadataEvent(kind="request_end", request_id=request_id),
                mutation=False,
            )

    def close(self) -> None:
        self._reporter_shutdown = True
        try:
            self._metadata_queue.put_nowait(None)
        except queue.Full:
            pass
        self._reporter_thread.join(timeout=1.0)
        self._client.close()

    def _enqueue_metadata(self, event: _MetadataEvent, *, mutation: bool) -> None:
        try:
            self._metadata_queue.put_nowait(event)
        except queue.Full:
            if mutation:
                self._client.sync_required = True
            logger.warning("GlobalKV metadata queue full; kind=%s", event.kind)

    def _reporter_loop(self) -> None:
        heartbeat_seconds = max(
            0.1,
            float(
                self._client.config.get_extra_config_value(
                    "globalkv_heartbeat_interval_seconds", 5.0
                )
            ),
        )
        inventory_page_size = max(
            1,
            int(
                self._client.config.get_extra_config_value(
                    "globalkv_inventory_page_size", 256
                )
            ),
        )
        next_heartbeat = time.monotonic()
        while not self._reporter_shutdown:
            timeout = max(0.0, next_heartbeat - time.monotonic())
            try:
                event = self._metadata_queue.get(timeout=timeout)
            except queue.Empty:
                event = None
            if time.monotonic() >= next_heartbeat:
                self._client.heartbeat(self._capacity_snapshot())
                lease_interval = (
                    self._client.lease_ttl_ms / 3000.0
                    if self._client.lease_ttl_ms > 0
                    else heartbeat_seconds
                )
                next_heartbeat = time.monotonic() + min(
                    heartbeat_seconds, max(0.1, lease_interval)
                )
            if self._client.sync_required:
                with self._ledger_lock:
                    snapshot = list(self._ledger.values())
                    snapshot_revision = self._ledger_revision
                if self._client.sync_inventory(snapshot, inventory_page_size):
                    self._synced_revision = max(
                        self._synced_revision, snapshot_revision
                    )
            if event is None:
                if self._reporter_shutdown:
                    self._metadata_queue.task_done()
                    return
                continue
            if (
                event.kind in {"store", "remove"}
                and event.revision <= self._synced_revision
            ):
                self._metadata_queue.task_done()
                continue
            try:
                if event.kind == "store":
                    version = self._client.report_stored_blocks(list(event.blocks))
                    if version > 0:
                        with self._ledger_lock:
                            for block in event.blocks:
                                self._replica_versions[block.seq_hash] = max(
                                    version,
                                    self._replica_versions.get(block.seq_hash, 0),
                                )
                elif event.kind == "remove":
                    version = self._client.report_removed_blocks(list(event.hashes))
                    if version > 0:
                        with self._ledger_lock:
                            for seq_hash in event.hashes:
                                self._replica_versions[seq_hash] = max(
                                    version,
                                    self._replica_versions.get(seq_hash, 0),
                                )
                elif event.kind == "request_start":
                    self._client.report_request_start_v2(
                        event.request_id, list(event.blocks)
                    )
                elif event.kind == "request_end":
                    self._client.report_request_end_v2(event.request_id)
            except Exception as exc:
                if event.kind in {"store", "remove"}:
                    self._client.sync_required = True
                logger.error(
                    "GlobalKV metadata reporter event failed: kind=%s error=%s",
                    event.kind,
                    exc,
                    exc_info=True,
                )
            finally:
                self._metadata_queue.task_done()

    def ledger_snapshot(self) -> dict[bytes, KvBlockMetadata]:
        with self._ledger_lock:
            return dict(self._ledger)

    def bind_engine(self, engine: "LMCacheEngine") -> None:
        self._engine = engine

    def _capacity_snapshot(self) -> dict[str, int] | None:
        engine = self._engine
        storage_manager = getattr(engine, "storage_manager", None)
        if storage_manager is None:
            return None
        storage_backends = getattr(storage_manager, "storage_backends", {})
        backend = storage_backends.get("LocalCPUBackend")
        if backend is None or not hasattr(backend, "capacity_snapshot"):
            return None
        try:
            return backend.capacity_snapshot()
        except Exception:
            logger.exception("Failed to read LocalCPUBackend capacity snapshot")
            return None

    def replica_version(self, seq_hash: bytes) -> int:
        with self._ledger_lock:
            return self._replica_versions.get(seq_hash, 0)

    @property
    def instance_epoch(self) -> str:
        return self._client.instance_epoch

    @property
    def compatibility_group_id(self) -> bytes:
        return self._client.compatibility_group_id

    def matches_instance(
        self, instance_id: str, worker_id: int, epoch: str
    ) -> bool:
        key = self._client.instance_key
        return (
            key is not None
            and key.lmcache_instance_id == instance_id
            and key.worker_id == worker_id
            and self._client.instance_epoch == epoch
        )

    def bind_instance_identity(self, metadata: LMCacheMetadata) -> None:
        instance_id = self._client.config.lmcache_instance_id
        if instance_id is None:
            return
        self._client.bind_instance_identity(instance_id, metadata)


class GlobalKvMigrationPlugin(KvMigrationPluginInterface):
    """Migration plugin for GlobalKV metadata reporting and KV transfer."""

    def __init__(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata | None = None,
        plugin_params: dict[str, Any] | None = None,
    ) -> None:
        self.config = config
        self.metadata = metadata
        self.plugin_params = plugin_params or {}
        self._engine: LMCacheEngine | None = None
        self._metadata_reporter: KvMetadataReporterInterface | None = None
        self._globalkv_server: GlobalKvServer | None = None
        self._migration_worker_shutdown = False
        queue_size = int(
            self.config.get_extra_config_value("kv_transfer_queue_size", 1024)
        )
        self._migration_queue: queue.Queue[_KvTransferJob | None] = queue.Queue(
            maxsize=queue_size
        )
        self._migration_worker: threading.Thread | None = None

    def create_metadata_reporter(
        self,
        *,
        meta_host: str | None = None,
        meta_port: int | None = None,
    ) -> KvMetadataReporterInterface | None:
        """Create the GlobalKV metadata reporter if transfer is configured."""
        if self._metadata_reporter is not None:
            return self._metadata_reporter

        if not getattr(self.config, "enable_kv_transfer", False):
            return None

        host = meta_host or self.plugin_params.get(
            "meta_host",
            self.config.get_extra_config_value("globalkv_meta_host", "127.0.0.1"),
        )
        port = int(
            meta_port
            if meta_port is not None
            else self.plugin_params.get(
                "meta_port",
                self.config.get_extra_config_value("globalkv_meta_port", 17070),
            )
        )
        try:
            client = KvCacheClient(self.config, str(host), port)
        except Exception as exc:
            if getattr(self.config, "globalkv_protocol", "v1") == "v2":
                raise
            logger.warning("Failed to create GlobalKV metadata client: %s", exc)
            return None

        self._metadata_reporter = KvCacheMetadataReporter(client)
        return self._metadata_reporter

    def get_metadata_reporter(self) -> KvMetadataReporterInterface | None:
        return self._metadata_reporter

    def start(self, engine: "LMCacheEngine") -> None:
        self._engine = engine

        if self._metadata_reporter is None:
            self.create_metadata_reporter()

        if (
            isinstance(self._metadata_reporter, KvCacheMetadataReporter)
            and engine.metadata is not None
        ):
            self._metadata_reporter.bind_instance_identity(engine.metadata)
            self._metadata_reporter.bind_engine(engine)

        enable_rpc_server = bool(
            self.plugin_params.get(
                "enable_rpc_server",
                getattr(self.config, "enable_globalkv_server", False),
            )
        )
        if enable_rpc_server:
            rpc_port = getattr(self.config, "kv_transfer_rpc_port", 17071)
            self._globalkv_server = GlobalKvServer(
                cache_engine=engine,
                host=getattr(self.config, "kv_transfer_host", "0.0.0.0"),
                port=rpc_port,
                max_workers=getattr(self.config, "globalkv_server_max_workers", 10),
            )
            self._globalkv_server.start()
            logger.info(
                "GlobalKvServer started on %s:%s",
                self._globalkv_server.host,
                rpc_port,
            )

        self._migration_worker = threading.Thread(
            target=self._migration_worker_loop,
            name=(
                f"LMCacheKvTransferWorker-"
                f"{engine.metadata.worker_id if engine.metadata else 0}"
            ),
            daemon=True,
        )
        self._migration_worker.start()

    def stop(self) -> None:
        self._stop_migration_worker()
        if self._globalkv_server is not None and self._globalkv_server.is_running():
            logger.info("Stopping GlobalKvServer...")
            self._globalkv_server.stop()
            logger.info("GlobalKvServer stopped.")
        self._engine = None

    def configure_kv_events_sink(self, engine: "LMCacheEngine") -> None:
        if not engine.kv_events_enabled or engine.storage_manager is None:
            return

        kv_transfer_backend = find_kv_transfer_backend(engine.storage_manager)
        if kv_transfer_backend is not None:
            kv_transfer_backend.set_kv_events_sink(engine.kv_events)

    def transfer(
        self,
        hashes: list[int],
        offsets: list[int],
        old_position: str,
        peer_ip: str,
        peer_init_port: int,
        event_id: str,
        do_copy: bool = True,
        token_ids: list[int] | None = None,
    ) -> int:
        if self._migration_worker_shutdown:
            logger.warning("KV transfer rejected because migration worker is stopped")
            return KV_TRANSFER_FAILED

        job = _KvTransferJob(
            hashes=hashes,
            offsets=offsets,
            old_position=old_position,
            peer_ip=peer_ip,
            peer_init_port=peer_init_port,
            event_id=event_id,
            do_copy=do_copy,
            token_ids=token_ids,
            compatibility_group_id=b"",
            expected_target_epoch="",
            expected_target_instance_id="",
            expected_target_worker_id=-1,
            result=Future(),
        )
        result = self._submit_kv_transfer_job(job)
        if result.all_already_satisfied:
            return KV_TRANSFER_ALREADY_SATISFIED
        if result.num_satisfied_chunks == 0:
            if result.block_results and result.block_results[0].status == (
                KvTransferBlockStatus.SOURCE_MISSING
            ):
                return KV_TRANSFER_NOT_FOUND
            return KV_TRANSFER_FAILED
        return sum(
            offset
            for offset, block_result in zip(
                offsets, result.block_results, strict=False
            )
            if block_result.status
            in {
                KvTransferBlockStatus.COPIED,
                KvTransferBlockStatus.ALREADY_PRESENT,
            }
        )

    def transfer_v2(
        self,
        *,
        hashes: list[bytes],
        offsets: list[int],
        old_position: str,
        peer_ip: str,
        peer_init_port: int,
        event_id: str,
        token_ids: list[int] | None,
        compatibility_group_id: bytes,
        expected_target_epoch: str,
        expected_target_instance_id: str,
        expected_target_worker_id: int,
    ) -> KvTransferPeerResult:
        if self._migration_worker_shutdown:
            return self._failure_result(hashes, "WORKER_STOPPED")
        return self._submit_kv_transfer_job(
            _KvTransferJob(
                hashes=hashes,
                offsets=offsets,
                old_position=old_position,
                peer_ip=peer_ip,
                peer_init_port=peer_init_port,
                event_id=event_id,
                do_copy=True,
                token_ids=token_ids,
                compatibility_group_id=compatibility_group_id,
                expected_target_epoch=expected_target_epoch,
                expected_target_instance_id=expected_target_instance_id,
                expected_target_worker_id=expected_target_worker_id,
                result=Future(),
            )
        )

    @staticmethod
    def _failure_result(
        hashes: list[int] | list[bytes],
        detail: str,
        status: KvTransferBlockStatus = KvTransferBlockStatus.READ_FAILED,
    ) -> KvTransferPeerResult:
        return KvTransferPeerResult(
            block_results=tuple(
                KvTransferBlockResult(
                    seq_hash=(
                        chunk_hash.to_bytes(32, "big")
                        if isinstance(chunk_hash, int)
                        else bytes(chunk_hash)
                    ),
                    status=status,
                    error_detail_code=detail,
                )
                for chunk_hash in hashes
            ),
            failed=True,
        )

    @staticmethod
    def _hash_bytes(chunk_hash: int | bytes) -> bytes:
        if isinstance(chunk_hash, int):
            return (chunk_hash & ((1 << 256) - 1)).to_bytes(32, "big")
        return bytes(chunk_hash).rjust(32, b"\x00")[-32:]

    def _source_missing_result(
        self, hashes: list[int | bytes]
    ) -> KvTransferPeerResult:
        return KvTransferPeerResult(
            block_results=tuple(
                KvTransferBlockResult(
                    seq_hash=self._hash_bytes(chunk_hash),
                    status=(
                        KvTransferBlockStatus.SOURCE_MISSING
                        if index == 0
                        else KvTransferBlockStatus.NOT_ATTEMPTED
                    ),
                    error_detail_code="SOURCE_MISSING",
                )
                for index, chunk_hash in enumerate(hashes)
            ),
            failed=True,
        )

    def _wait_for_foreground_idle(self) -> bool:
        engine = self._engine
        if engine is None:
            return False
        with engine._foreground_condition:
            while engine._foreground_ops > 0 and not self._migration_worker_shutdown:
                engine._foreground_condition.wait(timeout=0.1)
            return not self._migration_worker_shutdown

    def _migration_worker_loop(self) -> None:
        while True:
            job = self._migration_queue.get()
            try:
                if job is None:
                    return
                if not self._wait_for_foreground_idle():
                    job.result.set_result(
                        self._failure_result(job.hashes, "WORKER_STOPPED")
                    )
                    continue
                result = self._execute_kv_transfer_job(job)
                job.result.set_result(result)
            except Exception as exc:
                logger.error("KV transfer worker failed: %s", exc, exc_info=True)
                if job is not None:
                    job.result.set_result(
                        self._failure_result(job.hashes, type(exc).__name__)
                    )
            finally:
                self._migration_queue.task_done()

    def _submit_kv_transfer_job(self, job: _KvTransferJob) -> KvTransferPeerResult:
        self._migration_queue.put(job)
        return job.result.result()

    def _stop_migration_worker(self) -> None:
        self._migration_worker_shutdown = True
        engine = self._engine
        if engine is not None:
            with engine._foreground_condition:
                engine._foreground_condition.notify_all()

        worker = self._migration_worker
        if worker is None or not worker.is_alive():
            return

        while True:
            try:
                self._migration_queue.put_nowait(None)
                break
            except queue.Full:
                try:
                    pending_job = self._migration_queue.get_nowait()
                except queue.Empty:
                    continue
                if pending_job is not None:
                    pending_job.result.set_result(
                        self._failure_result(
                            pending_job.hashes, "WORKER_STOPPED"
                        )
                    )
                self._migration_queue.task_done()
        worker.join(timeout=1.0)

    def _execute_kv_transfer_job(self, job: _KvTransferJob) -> KvTransferPeerResult:
        engine = self._engine
        if engine is None or engine.storage_manager is None:
            return self._failure_result(job.hashes, "ENGINE_UNAVAILABLE")

        memory_objs: list[MemoryObj | None] | None = None
        num_tokens = engine.lookup(
            hashes=job.hashes,
            offsets=job.offsets,
            search_range=[job.old_position],
            lookup_id=job.event_id,
            pin=True,
        )
        if not num_tokens and any(isinstance(value, bytes) for value in job.hashes):
            num_tokens = engine.lookup(
                hashes=[
                    int.from_bytes(value, "big")
                    if isinstance(value, bytes)
                    else value
                    for value in job.hashes
                ],
                offsets=job.offsets,
                search_range=[job.old_position],
                lookup_id=job.event_id,
                pin=True,
            )

        if not num_tokens:
            logger.info(
                "KV transfer is not performed as there are no tokens to transfer."
            )
            return self._source_missing_result(job.hashes)

        try:
            block_mapping = engine.lookup_pins[job.event_id]
            keys = block_mapping.get(job.old_position, [])
            if not keys:
                logger.info(
                    "KV transfer is not performed as no matching keys were found "
                    "in %s.",
                    job.old_position,
                )
                return self._source_missing_result(job.hashes)

            memory_objs = engine.storage_manager.batched_get(
                keys=keys,
                location=job.old_position,
            )
            if memory_objs is None or any(obj is None for obj in memory_objs):
                logger.error("Failed to get memory objects to transfer")
                return self._failure_result(job.hashes, "SOURCE_READ_FAILED")

            transfer_objs = [obj for obj in memory_objs if obj is not None]
            logger.info(
                "Trying to transfer %d memory objects to peer %s:%s",
                len(transfer_objs),
                job.peer_ip,
                job.peer_init_port,
            )

            kv_transfer_backend = find_kv_transfer_backend(engine.storage_manager)
            if kv_transfer_backend is None:
                logger.error("KvTransferBackend is not available in storage backends")
                return self._failure_result(job.hashes, "BACKEND_UNAVAILABLE")

            hash_to_mem_index: dict[Any, int] = {}
            local_indexes = kv_transfer_backend.transfer_channel.get_local_mem_indices(
                transfer_objs
            )
            for key, mem_index in zip(keys, local_indexes, strict=False):
                hash_to_mem_index[self._hash_bytes(key.chunk_hash)] = mem_index

            full_mem_indexes: list[int] = []
            source_prefix_broken = False
            for chunk_hash in job.hashes:
                normalized_hash = self._hash_bytes(chunk_hash)
                if source_prefix_broken:
                    full_mem_indexes.append(KV_TRANSFER_MEM_INDEX_UNAVAILABLE)
                elif normalized_hash in hash_to_mem_index:
                    full_mem_indexes.append(hash_to_mem_index[normalized_hash])
                else:
                    full_mem_indexes.append(KV_TRANSFER_MEM_INDEX_UNAVAILABLE)
                    source_prefix_broken = True

            try:
                future = asyncio.run_coroutine_threadsafe(
                    kv_transfer_backend.transfer_to_peer(
                        peer_ip=job.peer_ip,
                        peer_init_port=job.peer_init_port,
                        hashes=job.hashes,
                        objs=transfer_objs,
                        offsets=job.offsets,
                        event_id=job.event_id,
                        token_ids=job.token_ids,
                        mem_indexes=full_mem_indexes,
                        compatibility_group_id=job.compatibility_group_id,
                        expected_target_epoch=job.expected_target_epoch,
                        expected_target_instance_id=job.expected_target_instance_id,
                        expected_target_worker_id=job.expected_target_worker_id,
                    ),
                    engine.storage_manager.loop,
                )
                transfer_result = future.result()
                if transfer_result.num_satisfied_chunks != len(job.hashes):
                    logger.warning(
                        "Only %d/%d chunks were satisfied by peer "
                        "(%d newly read, %d already existed)",
                        transfer_result.num_satisfied_chunks,
                        len(job.hashes),
                        transfer_result.num_read_chunks,
                        transfer_result.num_existing_chunks,
                    )
                elif transfer_result.all_already_satisfied:
                    logger.info(
                        "KV transfer already satisfied: peer %s:%s already "
                        "has all %d requested chunks",
                        job.peer_ip,
                        job.peer_init_port,
                        len(keys),
                    )
            except Exception as exc:
                logger.error("KV transfer failed with exception: %s", exc)
                return self._failure_result(job.hashes, type(exc).__name__)

            if not job.do_copy:
                engine.lookup_unpin(job.event_id)
                engine.storage_manager.batched_remove(
                    keys, locations=[job.old_position]
                )
                logger.info(
                    "Removed %d chunks from source location %s after transfer",
                    len(keys),
                    job.old_position,
                )

            logger.info(
                "KV transfer completed: status=%s from %s to peer %s:%s",
                transfer_result.num_satisfied_chunks,
                job.old_position,
                job.peer_ip,
                job.peer_init_port,
            )
            return transfer_result
        finally:
            engine.lookup_unpin(job.event_id)
            if memory_objs is not None:
                for memory_obj in memory_objs:
                    if memory_obj is not None:
                        memory_obj.ref_count_down()
