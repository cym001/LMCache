# SPDX-License-Identifier: Apache-2.0
"""
GlobalKV gRPC Server implementation.

This module provides a gRPC server that exposes the kv_transfer functionality
of the LMCacheEngine, allowing remote nodes to request KV cache transfers.
"""

# Third Party
import grpc
from concurrent import futures
from typing import Optional, List, Union
import uuid
import time

try:
    from lmcache_kv_transfer.proto import kvserver_pb2
    from lmcache_kv_transfer.proto import kvserver_pb2_grpc
    from lmcache_kv_transfer.proto import kvserver_v2_pb2
    from lmcache_kv_transfer.proto import kvserver_v2_pb2_grpc
except ImportError as exc:
    raise ImportError(
        "GlobalKV server proto modules are missing from lmcache_kv_transfer.proto"
    ) from exc

from lmcache.logging import init_logger
from lmcache.v1.kv_transfer_status import (
    KV_TRANSFER_ALREADY_SATISFIED,
    KV_TRANSFER_FAILED,
    KV_TRANSFER_NOT_FOUND,
)

logger = init_logger(__name__)
HashValue = Union[int, bytes]


class LmcacheServerV2Servicer(
    kvserver_v2_pb2_grpc.LmcacheServerV2Servicer
):
    """Structured copy-only transfer endpoint used by CedFs V2."""

    def __init__(self, cache_engine):
        self.cache_engine = cache_engine

    def TransferKvV2(self, request, context):
        blocks = list(request.blocks)
        config = self.cache_engine.config
        max_blocks = int(
            config.get_extra_config_value("globalkv_transfer_max_blocks", 128)
        )
        max_tokens = int(
            config.get_extra_config_value("globalkv_transfer_max_tokens", 32768)
        )
        max_bytes = int(
            config.get_extra_config_value(
                "globalkv_transfer_max_bytes", 4 * 1024 * 1024 * 1024
            )
        )
        estimated_bytes_per_token = int(
            config.get_extra_config_value(
                "globalkv_transfer_estimated_bytes_per_token", 96 * 1024
            )
        )
        if not request.do_copy:
            return self._uniform_response(request, 9, "MOVE_NOT_SUPPORTED")
        if not blocks or len(blocks) > max_blocks:
            return self._uniform_response(request, 9, "BLOCK_LIMIT")
        if sum(block.offset for block in blocks) > max_tokens:
            return self._uniform_response(request, 9, "TOKEN_LIMIT")
        if (
            sum(block.offset for block in blocks) * estimated_bytes_per_token
            > max_bytes
        ):
            return self._uniform_response(request, 9, "BYTE_LIMIT")
        if request.deadline_unix_ms and request.deadline_unix_ms <= int(
            time.time() * 1000
        ):
            return self._uniform_response(request, 6, "DEADLINE_EXPIRED")
        if any(len(block.seq_hash) != 32 for block in blocks):
            return self._uniform_response(request, 9, "INVALID_HASH")

        plugin = getattr(self.cache_engine, "_kv_migration_plugin", None)
        reporter = getattr(self.cache_engine, "metadata_reporter", None)
        if plugin is None or not hasattr(plugin, "transfer_v2"):
            return self._uniform_response(request, 9, "V2_PLUGIN_UNAVAILABLE")
        if reporter is None:
            return self._uniform_response(request, 9, "REPORTER_UNAVAILABLE")
        client = getattr(reporter, "_client", None)
        instance_key = getattr(client, "instance_key", None)
        source_key = request.source.key
        if (
            instance_key is None
            or instance_key.lmcache_instance_id
            != source_key.lmcache_instance_id
            or instance_key.worker_id != source_key.worker_id
            or reporter.instance_epoch != request.source.epoch
        ):
            return self._uniform_response(request, 9, "STALE_SOURCE_EPOCH")
        if reporter.compatibility_group_id != request.compatibility_group_id:
            return self._uniform_response(request, 7, "INCOMPATIBLE_GROUP")
        target_ip = request.target_endpoints.host
        target_port = request.target_endpoints.nixl_init_port
        if not target_ip or target_port <= 0:
            return self._uniform_response(request, 9, "INVALID_TARGET")

        result = plugin.transfer_v2(
            hashes=[bytes(block.seq_hash) for block in blocks],
            offsets=[block.offset for block in blocks],
            old_position="LocalCPUBackend",
            peer_ip=target_ip,
            peer_init_port=target_port,
            event_id=request.transfer_id or str(uuid.uuid4()),
            token_ids=[token for block in blocks for token in block.token_ids],
            compatibility_group_id=bytes(request.compatibility_group_id),
            expected_target_epoch=request.target.epoch,
            expected_target_instance_id=request.target.key.lmcache_instance_id,
            expected_target_worker_id=request.target.key.worker_id,
        )
        return kvserver_v2_pb2.TransferKvV2Response(
            transfer_id=request.transfer_id,
            results=[
                kvserver_v2_pb2.BlockTransferResultV2(
                    seq_hash=block_result.seq_hash,
                    status=int(block_result.status),
                    target_replica_version=block_result.target_replica_version,
                    bytes_transferred=block_result.bytes_transferred,
                    error_detail_code=block_result.error_detail_code,
                )
                for block_result in result.block_results
            ],
        )

    @staticmethod
    def _uniform_response(request, status: int, detail: str):
        return kvserver_v2_pb2.TransferKvV2Response(
            transfer_id=request.transfer_id,
            results=[
                kvserver_v2_pb2.BlockTransferResultV2(
                    seq_hash=block.seq_hash,
                    status=status,
                    error_detail_code=detail,
                )
                for block in request.blocks
            ],
        )


class LmcacheServerServicer(kvserver_pb2_grpc.LmcacheServerServicer):
    """
    gRPC service implementation for KV cache transfer operations.
    
    This servicer wraps the LMCacheEngine's kv_transfer method, allowing
    remote nodes to request KV cache transfers via gRPC.
    """
    
    def __init__(self, cache_engine):
        """
        Initialize the LmcacheServerServicer.
        
        Args:
            cache_engine: An instance of LMCacheEngine that will handle
                         the actual KV cache transfer operations.
        """
        self.cache_engine = cache_engine
        logger.info("LmcacheServerServicer initialized")
    
    def TransferKv(self, request, context):
        """
        Handle a KV cache transfer request.
        
        This method is called when a remote node requests a KV cache transfer.
        It extracts the parameters from the gRPC request and calls the
        cache_engine's kv_transfer method.
        
        Args:
            request: TransferKvRequest containing:
                - hash: bytes - The hash identifying the KV cache chunks
                - position: str - Source storage backend location
                - offset: repeated uint32 - Token counts for each chunk
                - target_ip: str - IP address of the target peer
                - target_port: int32 - Port of the target peer
                - do_copy: bool - Whether to copy (True) or move (False)
            context: gRPC context for the request
            
        Returns:
            TransferKvResponse with success status
        """
        try:
            # Extract offsets from the repeated field
            offsets = list(request.offset)
            
            # Parse the hash bytes to extract chunk hashes
            # The number of hashes should match the number of offsets
            hash_bytes = request.hash
            hashes = self._parse_hashes(hash_bytes, len(offsets))
            # Log hashes in hex for easier cross-node debugging.
            # hashes_hex = [h.hex() for h in hashes]
            # logger.info(f"TransferKv: hashes(hex)={hashes_hex}")
            
            # Get the source position (storage backend location)
            old_position = request.position if request.position else "LocalCPUBackend"
            
            # Get target peer information
            target_ip = request.target_ip
            target_port = request.target_port
            
            # Get copy/move flag
            do_copy = request.do_copy

            # Optional flat token ids for transfer event reporting
            token_ids = list(request.tokens)
            
            # Generate a unique event ID for this transfer
            event_id = str(uuid.uuid4())
            
            # logger.info(
            #     f"TransferKv request received: "
            #     f"hashes_count={len(hashes)}, "
            #     f"offsets={offsets}, "
            #     f"position={old_position}, "
            #     f"target={target_ip}:{target_port}, "
            #     f"do_copy={do_copy}, "
            #     f"event_id={event_id}"
            # )
            
            # Validate inputs
            if not hashes:
                logger.warning("TransferKv: No hashes provided")
                return kvserver_pb2.TransferKvResponse(status=KV_TRANSFER_FAILED)
            
            if not offsets:
                logger.warning("TransferKv: No offsets provided")
                return kvserver_pb2.TransferKvResponse(status=KV_TRANSFER_FAILED)
            
            if len(hashes) != len(offsets):
                logger.warning(
                    f"TransferKv: Mismatch between hashes ({len(hashes)}) "
                    f"and offsets ({len(offsets)})"
                )
                return kvserver_pb2.TransferKvResponse(status=KV_TRANSFER_FAILED)
            
            if not target_ip or target_port <= 0:
                logger.warning(
                    f"TransferKv: Invalid target: {target_ip}:{target_port}"
                )
                return kvserver_pb2.TransferKvResponse(status=KV_TRANSFER_FAILED)
            
            # Call the cache engine's kv_transfer method
            # Returns:
            #   KV_TRANSFER_NOT_FOUND if source KV cache does not exist
            #   KV_TRANSFER_FAILED if transfer failed for other reasons
            #   KV_TRANSFER_ALREADY_SATISFIED if target already has all chunks
            #   Other positive values: number of tokens successfully transferred
            num_tokens = self.cache_engine.kv_transfer(
                hashes=hashes,
                offsets=offsets,
                old_position=old_position,
                peer_ip=target_ip,
                peer_init_port=target_port,
                event_id=event_id,
                do_copy=do_copy,
                token_ids=token_ids,
            )
            if num_tokens == KV_TRANSFER_NOT_FOUND:
                fallback_hashes = self._convert_hashes_to_int(hashes)
                logger.info(
                    "TransferKv retry with int hashes for type compatibility."
                )
                num_tokens = self.cache_engine.kv_transfer(
                    hashes=fallback_hashes,
                    offsets=offsets,
                    old_position=old_position,
                    peer_ip=target_ip,
                    peer_init_port=target_port,
                    event_id=event_id,
                    do_copy=do_copy,
                    token_ids=token_ids,
                )
            
            if num_tokens == KV_TRANSFER_ALREADY_SATISFIED:
                logger.info(
                    "TransferKv completed: target already has all requested chunks"
                )

            # Return status directly from kv_transfer
            # KV_TRANSFER_NOT_FOUND: source KV cache does not exist
            # KV_TRANSFER_FAILED: transfer failed for other reasons
            # KV_TRANSFER_ALREADY_SATISFIED: target already has all requested chunks
            # Other positive values: number of tokens transferred successfully
            return kvserver_pb2.TransferKvResponse(status=num_tokens)
            
        except Exception as e:
            logger.error(f"TransferKv failed with exception: {e}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return kvserver_pb2.TransferKvResponse(status=KV_TRANSFER_FAILED)
    
    def _parse_hashes(self, hash_bytes: bytes, num_hashes: int) -> List[bytes]:
        """
        Parse the hash bytes into a list of 32-byte hashes based on offset count.
        
        The hash bytes are expected to be a sequence of 32-byte (256-bit)
        hash values concatenated together. The number of hashes to parse
        is determined by the num_hashes parameter (which equals len(offsets)).
        
        Args:
            hash_bytes: Raw bytes containing concatenated hash values
            num_hashes: Number of hashes to parse (should match len(offsets))
            
        Returns:
            List of 32-byte hash values (bytes)
        """
        if not hash_bytes or num_hashes <= 0:
            return []
        
        # Each hash is 32 bytes (256 bits)
        hash_size = 32
        expected_size = hash_size * num_hashes
        
        # Validate hash_bytes length
        if len(hash_bytes) < expected_size:
            logger.warning(
                f"Hash bytes length ({len(hash_bytes)}) is less than expected "
                f"({expected_size} = {hash_size} * {num_hashes}). "
                f"Will parse as many complete hashes as possible."
            )
        
        hashes: List[bytes] = []
        for i in range(num_hashes):
            start = i * hash_size
            end = start + hash_size
            
            if end <= len(hash_bytes):
                # Extract a complete 32-byte hash.
                chunk = hash_bytes[start:end]
                hashes.append(chunk)
            elif start < len(hash_bytes):
                # Handle partial chunk at the end (left pad to keep big-endian semantics)
                chunk = hash_bytes[start:]
                padded = chunk.rjust(hash_size, b"\x00")
                hashes.append(padded)
                logger.warning(
                    f"Hash {i} has only {len(chunk)} bytes, padded to {hash_size} bytes"
                )
            else:
                # No more bytes available
                logger.warning(
                    f"No bytes available for hash {i}, expected {num_hashes} hashes"
                )
                break
        
        return hashes

    def _convert_hashes_to_int(self, hashes: List[bytes]) -> List[int]:
        """Convert 32-byte hashes to big-endian integers for fallback path."""
        return [int.from_bytes(h, byteorder="big", signed=False) for h in hashes]


class GlobalKvServer:
    """
    gRPC server for global KV cache operations.
    
    This server exposes the LMCacheEngine's kv_transfer functionality
    via gRPC, allowing remote nodes to request KV cache transfers.
    
    The server runs in non-blocking mode by default, allowing the main
    thread to continue with other tasks.
    """
    
    def __init__(
        self,
        cache_engine,
        host: str = "0.0.0.0",
        port: int = 50052,
        max_workers: int = 10,
    ):
        """
        Initialize the GlobalKvServer.
        
        Args:
            cache_engine: An instance of LMCacheEngine
            host: Host address to bind the server to
            port: Port number for the gRPC server
            max_workers: Maximum number of worker threads
        """
        self.cache_engine = cache_engine
        self.host = host
        self.port = port
        self.max_workers = max_workers
        self.server: Optional[grpc.Server] = None
        self._running = False
        
        logger.info(
            f"GlobalKvServer initialized with host={host}, port={port}, "
            f"max_workers={max_workers}"
        )
    
    def start(self):
        """
        Start the gRPC server in non-blocking mode.
        
        The server runs in background threads managed by gRPC's ThreadPoolExecutor.
        Call stop() to shutdown the server gracefully.
        
        Returns:
            self: Returns the server instance for method chaining.
            
        Raises:
            RuntimeError: If the server fails to bind to the specified address.
        """
        if self._running:
            logger.warning("GlobalKvServer is already running")
            return self
        
        server_address = f"{self.host}:{self.port}"
        
        self.server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=self.max_workers)
        )
        
        # Create and register the servicer
        servicer = LmcacheServerServicer(self.cache_engine)
        kvserver_pb2_grpc.add_LmcacheServerServicer_to_server(servicer, self.server)
        kvserver_v2_pb2_grpc.add_LmcacheServerV2Servicer_to_server(
            LmcacheServerV2Servicer(self.cache_engine), self.server
        )
        
        # Bind to the specified address and verify binding succeeded
        try:
            bound_port = self.server.add_insecure_port(server_address)
            if bound_port == 0:
                raise RuntimeError(f"gRPC returned port 0 for {server_address}")
        except Exception as e:
            error_msg = (
                f"Failed to bind gRPC server to {server_address}. "
                f"The port {self.port} may be in use by another process. "
                f"Check with: lsof -i :{self.port} or netstat -tlnp | grep {self.port}\n"
                f"Original error: {e}"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
        
        # Start the server (non-blocking)
        self.server.start()
        self._running = True
        logger.info(f"GlobalKvServer started on {server_address} (non-blocking)")
        
        return self
    
    def stop(self, grace: Optional[float] = 5.0):
        """
        Stop the gRPC server gracefully.
        
        Args:
            grace: Grace period in seconds for ongoing RPCs to complete.
                  Defaults to 5.0 seconds. If None, stops immediately.
        """
        if not self._running or self.server is None:
            logger.warning("GlobalKvServer is not running")
            return
        
        logger.info(f"Stopping GlobalKvServer with grace period: {grace}s")
        self.server.stop(grace)
        self._running = False
        logger.info("GlobalKvServer stopped")
    
    def wait_for_termination(self, timeout: Optional[float] = None):
        """
        Block until the server terminates.
        
        Args:
            timeout: Maximum time to wait in seconds. If None, wait indefinitely.
        """
        if self.server:
            self.server.wait_for_termination(timeout)
    
    def is_running(self) -> bool:
        """
        Check if the server is currently running.
        
        Returns:
            True if the server is running, False otherwise.
        """
        return self._running
    
    def __enter__(self):
        """Context manager entry - starts the server."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - stops the server."""
        self.stop()
        return False


def serve(cache_engine, host: str = "0.0.0.0", port: int = 50052, blocking: bool = True):
    """
    Convenience function to start a GlobalKvServer.
    
    Args:
        cache_engine: An instance of LMCacheEngine
        host: Host address to bind the server to
        port: Port number for the gRPC server
        blocking: If True, block until the server is terminated.
                 If False, return the server instance immediately.
    
    Returns:
        GlobalKvServer instance (useful when blocking=False)
    """
    server = GlobalKvServer(cache_engine, host, port)
    server.start()
    
    if blocking:
        try:
            server.wait_for_termination()
        except KeyboardInterrupt:
            logger.info("Received interrupt signal, stopping server...")
            server.stop()
    
    return server


def start_server_non_blocking(cache_engine, host: str = "0.0.0.0", port: int = 50052) -> GlobalKvServer:
    """
    Start a GlobalKvServer in non-blocking mode.
    
    This is the recommended way to start the server when you need to
    continue with other tasks in the main thread.
    
    Args:
        cache_engine: An instance of LMCacheEngine
        host: Host address to bind the server to
        port: Port number for the gRPC server
    
    Returns:
        GlobalKvServer instance. Call server.stop() when done.
    
    Example:
        server = start_server_non_blocking(cache_engine)
        # ... do other work ...
        server.stop()  # Stop when done
    """
    server = GlobalKvServer(cache_engine, host, port)
    server.start()
    return server


if __name__ == "__main__":
    # Example usage - this would typically be called with a real cache_engine
    print("GlobalKvServer module loaded.")
    print()
    print("Usage examples:")
    print()
    print("1. Non-blocking mode (recommended):")
    print("   from lmcache.v1.remote.server import GlobalKvServer")
    print("   server = GlobalKvServer(cache_engine, host='0.0.0.0', port=50052)")
    print("   server.start()  # Returns immediately")
    print("   # ... do other work ...")
    print("   server.stop()   # Stop when done")
    print()
    print("2. Using context manager:")
    print("   with GlobalKvServer(cache_engine) as server:")
    print("       # Server is running")
    print("       # ... do work ...")
    print("   # Server is automatically stopped")
    print()
    print("3. Blocking mode:")
    print("   from lmcache.v1.remote.server import serve")
    print("   serve(cache_engine, blocking=True)  # Blocks until terminated")
