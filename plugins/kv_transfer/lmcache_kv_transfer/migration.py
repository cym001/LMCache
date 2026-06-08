# SPDX-License-Identifier: Apache-2.0
"""GlobalKV migration plugin implementation."""

# Standard
from concurrent.futures import Future
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Optional
import asyncio
import queue
import threading

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
    KvMetadataReporterInterface,
    KvMigrationPluginInterface,
    find_kv_transfer_backend,
)

from lmcache_kv_transfer.backend import KV_TRANSFER_MEM_INDEX_UNAVAILABLE
from lmcache_kv_transfer.globalkv_server import GlobalKvServer
from lmcache_kv_transfer.metadata_client import KvCacheClient

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.cache_engine import LMCacheEngine

logger = init_logger(__name__)


@dataclass
class _KvTransferJob:
    hashes: List[int]
    offsets: List[int]
    old_position: str
    peer_ip: str
    peer_init_port: int
    event_id: str
    do_copy: bool
    token_ids: Optional[List[int]]
    result: Future[int]


class KvCacheMetadataReporter(KvMetadataReporterInterface):
    """Adapter that exposes KvCacheClient through the core reporter interface."""

    def __init__(self, client: KvCacheClient) -> None:
        self._client = client

    def on_kv_stored(self, tokens: list[int]) -> None:
        self._client.upload_kv_meta(tokens)

    def on_kv_retrieved(self, hit_tokens: list[int]) -> None:
        self._client.upload_kv_meta(hit_tokens)

    def on_kv_removed(self, chunk_hashes: list[bytes]) -> None:
        self._client.remove_kv_meta(chunk_hashes)

    def on_request_start(self, request_id: str, tokens: list[int]) -> None:
        self._client.new_request(request_id, tokens)

    def on_request_end(self, request_id: str, tokens: list[int]) -> None:
        self._client.request_end(request_id, tokens)

    def close(self) -> None:
        self._client.close()


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
            result=Future(),
        )
        return self._submit_kv_transfer_job(job)

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
                    job.result.set_result(KV_TRANSFER_FAILED)
                    continue
                result = self._execute_kv_transfer_job(job)
                job.result.set_result(result)
            except Exception as exc:
                logger.error("KV transfer worker failed: %s", exc, exc_info=True)
                if job is not None:
                    job.result.set_result(KV_TRANSFER_FAILED)
            finally:
                self._migration_queue.task_done()

    def _submit_kv_transfer_job(self, job: _KvTransferJob) -> int:
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
                    pending_job.result.set_result(KV_TRANSFER_FAILED)
                self._migration_queue.task_done()
        worker.join(timeout=1.0)

    def _execute_kv_transfer_job(self, job: _KvTransferJob) -> int:
        engine = self._engine
        if engine is None or engine.storage_manager is None:
            return KV_TRANSFER_FAILED

        memory_objs: list[MemoryObj | None] | None = None
        num_tokens = engine.lookup(
            hashes=job.hashes,
            offsets=job.offsets,
            search_range=[job.old_position],
            lookup_id=job.event_id,
            pin=True,
        )

        if not num_tokens:
            logger.info(
                "KV transfer is not performed as there are no tokens to transfer."
            )
            return KV_TRANSFER_NOT_FOUND

        try:
            block_mapping = engine.lookup_pins[job.event_id]
            keys = block_mapping.get(job.old_position, [])
            if not keys:
                logger.info(
                    "KV transfer is not performed as no matching keys were found "
                    "in %s.",
                    job.old_position,
                )
                return KV_TRANSFER_NOT_FOUND

            memory_objs = engine.storage_manager.batched_get(
                keys=keys,
                location=job.old_position,
            )
            if memory_objs is None or any(obj is None for obj in memory_objs):
                logger.error("Failed to get memory objects to transfer")
                return KV_TRANSFER_FAILED

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
                return KV_TRANSFER_FAILED

            hash_to_mem_index: dict[Any, int] = {}
            local_indexes = kv_transfer_backend.transfer_channel.get_local_mem_indices(
                transfer_objs
            )
            for key, mem_index in zip(keys, local_indexes, strict=False):
                hash_to_mem_index[key.chunk_hash] = mem_index

            full_mem_indexes: list[int] = []
            source_prefix_broken = False
            for chunk_hash in job.hashes:
                if source_prefix_broken:
                    full_mem_indexes.append(KV_TRANSFER_MEM_INDEX_UNAVAILABLE)
                elif chunk_hash in hash_to_mem_index:
                    full_mem_indexes.append(hash_to_mem_index[chunk_hash])
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
                    ),
                    engine.storage_manager.loop,
                )
                transfer_result = future.result()
                num_satisfied = transfer_result.num_satisfied_chunks
                transfer_status = num_tokens

                if num_satisfied != len(job.hashes):
                    logger.warning(
                        "Only %d/%d chunks were satisfied by peer "
                        "(%d newly read, %d already existed)",
                        num_satisfied,
                        len(job.hashes),
                        transfer_result.num_read_chunks,
                        transfer_result.num_existing_chunks,
                    )
                    if num_satisfied == 0:
                        return KV_TRANSFER_FAILED
                elif transfer_result.all_already_satisfied:
                    logger.info(
                        "KV transfer already satisfied: peer %s:%s already "
                        "has all %d requested chunks",
                        job.peer_ip,
                        job.peer_init_port,
                        len(keys),
                    )
                    transfer_status = KV_TRANSFER_ALREADY_SATISFIED
                else:
                    transfer_status = num_tokens
            except Exception as exc:
                logger.error("KV transfer failed with exception: %s", exc)
                return KV_TRANSFER_FAILED

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
                transfer_status,
                job.old_position,
                job.peer_ip,
                job.peer_init_port,
            )
            return transfer_status
        finally:
            engine.lookup_unpin(job.event_id)
            if memory_objs is not None:
                for memory_obj in memory_objs:
                    if memory_obj is not None:
                        memory_obj.ref_count_down()
