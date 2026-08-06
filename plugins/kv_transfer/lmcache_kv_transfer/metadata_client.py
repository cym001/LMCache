# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass
from typing import List
import hashlib
import ipaddress
import os
import threading
import time
import uuid

# Third Party
import grpc

from lmcache.logging import init_logger
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.plugin.kv_migration import KvBlockMetadata

try:
    from lmcache_kv_transfer.proto import kvcache_pb2
    from lmcache_kv_transfer.proto import kvcache_pb2_grpc
except ImportError as exc:
    raise ImportError(
        "GlobalKV proto modules are missing from lmcache_kv_transfer.proto"
    ) from exc

try:
    from lmcache_kv_transfer.proto import kvcache_v2_pb2
    from lmcache_kv_transfer.proto import kvcache_v2_pb2_grpc
    from lmcache_kv_transfer.proto import v2_descriptor_sha256
except (ImportError, RuntimeError):
    kvcache_v2_pb2 = None
    kvcache_v2_pb2_grpc = None
    v2_descriptor_sha256 = None

logger = init_logger(__name__)


@dataclass(frozen=True)
class InstanceKey:
    """Stable V2 identity for one distributed cache rank."""

    lmcache_instance_id: str
    worker_id: int


class KvCacheClient:
    """KvCache gRPC客户端"""
    
    def __init__(self, config: LMCacheEngineConfig,
                 meta_server_host: str = 'localhost',
                 meta_server_port: int = 50051):
        """
        初始化客户端
        
        """
        self.config = config
        self.meta_server_host = meta_server_host
        self.meta_server_port = meta_server_port
        self.channel = None
        self.stub = None
        self.v2_stub = None
        self.instance_key: InstanceKey | None = None
        self.instance_epoch = str(uuid.uuid4())
        self.compatibility_group_id = b""
        self.lease_id = ""
        self.event_seq = 0
        self._mutation_lock = threading.Lock()
        self.meta_generation = ""
        self.lease_ttl_ms = 0
        self.sync_required = False
        self._metadata: LMCacheMetadata | None = None
        self.protocol = getattr(config, "globalkv_protocol", "v1")
        self.rpc_timeout = float(
            config.get_extra_config_value("globalkv_rpc_timeout_seconds", 5.0)
        )
        
        # 数据服务器信息
        self.data_server_ip = config.kv_transfer_host
        self.data_server_http_port = config.kv_transfer_http_port
        self.data_server_init_port = config.kv_transfer_init_port
        self.data_server_rpc_port = config.kv_transfer_rpc_port
        self.model_name = config.kv_transfer_model_name
        self.advertised_host = str(
            config.get_extra_config_value(
                "globalkv_advertised_host", self.data_server_ip
            )
        )
        self.api_scheme = str(
            config.get_extra_config_value("globalkv_api_scheme", "http")
        )
        self.api_path = str(
            config.get_extra_config_value("globalkv_api_path", "/v1")
        )
        if self.api_scheme not in {"http", "https"}:
            raise ValueError("globalkv_api_scheme must be http or https")
        if not self.api_path.startswith("/"):
            raise ValueError("globalkv_api_path must start with /")
        if not self.advertised_host or self.advertised_host in {"0.0.0.0", "::"}:
            raise ValueError(
                "globalkv_advertised_host must be a routable host, not a wildcard"
            )
        if self.protocol in {"v1", "dual"}:
            try:
                ipaddress.ip_address(self.advertised_host)
            except ValueError as exc:
                raise ValueError(
                    "V1/dual globalkv_advertised_host must be an IP address"
                ) from exc
        self.api_host = (
            f"[{self.advertised_host}]"
            if ":" in self.advertised_host
            else self.advertised_host
        )
        
        # 生成server_id: ip+http_port的hash值
        id_str = f"{self.advertised_host}:{self.data_server_http_port}"
        self.server_id = hash(id_str) & 0xFFFFFFFF  # 转换为无符号32位整数

        self.connect()
        if self.protocol in {"v1", "dual"}:
            success = self.register_instance()
            logger.info("GlobalKV V1 instance registration result: %s", success)
        if self.protocol in {"dual", "v2"}:
            self._negotiate_v2_capabilities()
    
    def connect(self):
        """建立连接到元数据服务器"""
        self.channel = grpc.insecure_channel(f'{self.meta_server_host}:{self.meta_server_port}')
        if self.protocol in {"v1", "dual"}:
            self.stub = kvcache_pb2_grpc.KvMeta2DataStub(self.channel)
        if kvcache_v2_pb2_grpc is not None:
            self.v2_stub = kvcache_v2_pb2_grpc.KvMeta2DataV2Stub(self.channel)
        logger.info(
            "Connected to GlobalKV metadata server %s:%s as legacy server id %s",
            self.meta_server_host,
            self.meta_server_port,
            self.server_id,
        )

    def _negotiate_v2_capabilities(self) -> None:
        """Verify protocol version and the cross-language descriptor checksum."""
        if (
            self.v2_stub is None
            or kvcache_v2_pb2 is None
            or v2_descriptor_sha256 is None
        ):
            message = "generated GlobalKV V2 Python bindings are unavailable"
            if self.protocol == "v2":
                raise RuntimeError(message)
            logger.warning("%s; V1 reporting remains active", message)
            return
        try:
            response = self.v2_stub.GetCapabilities(
                kvcache_v2_pb2.GetCapabilitiesRequestV2(),
                timeout=self.rpc_timeout,
            )
            if response.protocol_major != 2:
                raise RuntimeError(
                    f"unsupported GlobalKV protocol major {response.protocol_major}"
                )
            local_digest = v2_descriptor_sha256()
            if bytes(response.descriptor_sha256) != local_digest:
                raise RuntimeError("GlobalKV V2 descriptor checksum mismatch")
        except grpc.RpcError as exc:
            self.sync_required = True
            logger.warning(
                "GlobalKV V2 capability handshake is temporarily unavailable: %s",
                exc,
            )
        except RuntimeError as exc:
            if self.protocol == "v2":
                raise RuntimeError("GlobalKV V2 capability handshake failed") from exc
            logger.warning(
                "GlobalKV V2 capability handshake failed in dual mode; "
                "V1 reporting remains active: %s",
                exc,
            )
            self.v2_stub = None

    def bind_instance_identity(
        self,
        lmcache_instance_id: str,
        metadata: LMCacheMetadata,
    ) -> None:
        """Bind V2 identity after LMCache engine metadata becomes available.

        worker_id is the global distributed rank embedded in CacheEngineKey.
        local_worker_id is deliberately excluded because it can repeat across hosts.
        """
        if not lmcache_instance_id:
            raise ValueError("lmcache_instance_id must not be empty")
        if metadata.worker_id < 0:
            raise ValueError("worker_id must be non-negative")
        self.instance_key = InstanceKey(lmcache_instance_id, metadata.worker_id)
        self._metadata = metadata
        if self.protocol in {"dual", "v2"}:
            self._register_instance_v2(metadata)

    def _register_instance_v2(self, metadata: LMCacheMetadata) -> None:
        if kvcache_v2_pb2 is None or kvcache_v2_pb2_grpc is None:
            message = "generated GlobalKV V2 Python bindings are unavailable"
            if self.protocol == "v2":
                raise RuntimeError(message)
            logger.warning("%s; V1 reporting remains active", message)
            return
        assert self.channel is not None
        assert self.instance_key is not None
        if self.v2_stub is None:
            return
        identity = kvcache_v2_pb2.InstanceIdentityV2(
            key=kvcache_v2_pb2.InstanceKeyV2(
                lmcache_instance_id=self.instance_key.lmcache_instance_id,
                worker_id=self.instance_key.worker_id,
            ),
            epoch=self.instance_epoch,
        )
        world_size = metadata.world_size
        tp_size = int(
            self.config.get_extra_config_value(
                "globalkv_tensor_parallel_size", world_size
            )
        )
        pp_size = int(
            self.config.get_extra_config_value("globalkv_pipeline_parallel_size", 1)
        )
        if tp_size <= 0 or pp_size <= 0 or tp_size * pp_size != world_size:
            raise ValueError(
                "GlobalKV TP/PP sizes must be positive and multiply to world_size"
            )
        fingerprint = kvcache_v2_pb2.CompatibilityFingerprintV2(
            model_name=metadata.model_name,
            model_revision=str(
                self.config.get_extra_config_value("globalkv_model_revision", "")
            ),
            tokenizer_name=str(
                self.config.get_extra_config_value("globalkv_tokenizer_name", "")
            ),
            tokenizer_revision=str(
                self.config.get_extra_config_value("globalkv_tokenizer_revision", "")
            ),
            hash_algorithm=self.config.pre_caching_hash_algorithm,
            hash_seed=int(self.config.get_extra_config_value("globalkv_hash_seed", 0)),
            python_hash_seed=os.environ.get("PYTHONHASHSEED", ""),
            chunk_size=metadata.chunk_size,
            save_unfull_chunk=self.config.save_unfull_chunk,
            kv_dtype=",".join(str(dtype) for dtype in metadata.get_dtypes()),
            kv_layout="mla" if metadata.use_mla else "kv",
            tensor_parallel_size=tp_size,
            pipeline_parallel_size=pp_size,
            world_size=world_size,
            worker_id=metadata.worker_id,
            tensor_parallel_rank=metadata.worker_id % tp_size,
            pipeline_parallel_rank=metadata.worker_id // tp_size,
        )
        request = kvcache_v2_pb2.RegisterInstanceV2Request(
            protocol_major=2,
            protocol_minor=0,
            instance=identity,
            endpoints=kvcache_v2_pb2.InstanceEndpointsV2(
                host=self.advertised_host,
                http_port=self.data_server_http_port,
                nixl_init_port=self.data_server_init_port,
                transfer_rpc_port=self.data_server_rpc_port,
                api_path=self.api_path,
            ),
            fingerprint=fingerprint,
            known_meta_generation=self.meta_generation,
        )
        try:
            response = self.v2_stub.RegisterInstance(
                request,
                timeout=self.rpc_timeout,
            )
            if response.status != kvcache_v2_pb2.REGISTER_STATUS_V2_ACCEPTED:
                raise RuntimeError(f"GlobalKV V2 registration rejected: {response.error_detail}")
            self.compatibility_group_id = bytes(response.compatibility_group_id)
            self.lease_id = response.lease_id
            self.lease_ttl_ms = int(response.lease_ttl_ms)
            self.meta_generation = response.meta_generation
            self.sync_required = bool(response.require_inventory_sync)
        except grpc.RpcError as exc:
            self.sync_required = True
            logger.warning("GlobalKV V2 registration temporarily failed: %s", exc)
        except RuntimeError as exc:
            if self.protocol == "v2":
                raise
            logger.warning("GlobalKV V2 registration failed; V1 remains active: %s", exc)
            self.v2_stub = None

    def report_stored_blocks(self, blocks: list[KvBlockMetadata]) -> int:
        if not blocks or self.v2_stub is None:
            return 0
        proto_blocks = [self._proto_block(block) for block in blocks]
        return self._report_mutation(
            store=kvcache_v2_pb2.StoreBlocksV2(blocks=proto_blocks)
        )

    def report_removed_blocks(self, seq_hashes: list[bytes]) -> int:
        if not seq_hashes or self.v2_stub is None:
            return 0
        return self._report_mutation(
            remove=kvcache_v2_pb2.RemoveBlocksV2(seq_hashes=seq_hashes)
        )

    def _report_mutation(self, **payload) -> int:
        with self._mutation_lock:
            return self._report_mutation_locked(**payload)

    def _report_mutation_locked(self, **payload) -> int:
        assert self.instance_key is not None
        self.event_seq += 1
        identity = kvcache_v2_pb2.InstanceIdentityV2(
            key=kvcache_v2_pb2.InstanceKeyV2(
                lmcache_instance_id=self.instance_key.lmcache_instance_id,
                worker_id=self.instance_key.worker_id,
            ),
            epoch=self.instance_epoch,
        )
        request = kvcache_v2_pb2.ReportCacheMutationsV2Request(
            session=kvcache_v2_pb2.InstanceSessionV2(
                instance=identity,
                lease_id=self.lease_id,
            ),
            compatibility_group_id=self.compatibility_group_id,
            events=[
                kvcache_v2_pb2.CacheMutationEventV2(
                    event_seq=self.event_seq,
                    **payload,
                )
            ],
        )
        try:
            response = None
            for attempt in range(3):
                try:
                    response = self.v2_stub.ReportCacheMutations(
                        request,
                        timeout=self.rpc_timeout,
                    )
                    break
                except grpc.RpcError:
                    if attempt == 2:
                        raise
                    time.sleep(0.05 * (2**attempt))
            assert response is not None
            accepted = {
                kvcache_v2_pb2.MUTATION_STATUS_V2_COMMITTED,
                kvcache_v2_pb2.MUTATION_STATUS_V2_DUPLICATE,
            }
            if response.status not in accepted:
                self.sync_required = bool(response.require_inventory_sync)
                raise RuntimeError(
                    "GlobalKV V2 mutation rejected: "
                    f"status={response.status} detail={response.error_detail}"
                )
            return int(response.committed_through_seq)
        except (grpc.RpcError, RuntimeError) as exc:
            self.sync_required = True
            logger.warning("GlobalKV V2 mutation requires recovery: %s", exc)
            return 0
    
    def heartbeat(self, capacity: dict[str, int] | None = None) -> bool:
        if self.v2_stub is None or self.instance_key is None:
            return False
        try:
            request = kvcache_v2_pb2.HeartbeatV2Request(
                session=self._session(),
                known_meta_generation=self.meta_generation,
            )
            if capacity is not None:
                request.capacity.CopyFrom(
                    kvcache_v2_pb2.CapacitySnapshotV2(**capacity)
                )
            response = self.v2_stub.Heartbeat(
                request,
                timeout=self.rpc_timeout,
            )
            if response.require_registration:
                if self._metadata is not None:
                    self._register_instance_v2(self._metadata)
                return False
            self.meta_generation = response.meta_generation
            self.lease_ttl_ms = int(response.lease_ttl_ms)
            self.sync_required = self.sync_required or bool(
                response.require_inventory_sync
            )
            return bool(response.accepted)
        except grpc.RpcError as exc:
            logger.warning("GlobalKV V2 heartbeat failed: %s", exc)
            return False

    def sync_inventory(self, blocks: list[KvBlockMetadata], page_size: int) -> bool:
        if self.v2_stub is None or self.instance_key is None:
            return False
        proto_blocks = [self._proto_block(block) for block in blocks]
        pages = [
            proto_blocks[index : index + page_size]
            for index in range(0, len(proto_blocks), page_size)
        ] or [[]]
        try:
            begin = self.v2_stub.BeginInventorySync(
                kvcache_v2_pb2.BeginInventorySyncV2Request(
                    session=self._session(),
                    compatibility_group_id=self.compatibility_group_id,
                    base_event_seq=self.event_seq,
                    total_blocks=len(proto_blocks),
                    total_pages=len(pages),
                    inventory_checksum=self._inventory_checksum(proto_blocks),
                ),
                timeout=self.rpc_timeout,
            )
            if not begin.accepted:
                return False
            effective_page_size = min(page_size, int(begin.page_size_limit))
            if effective_page_size != page_size:
                self.v2_stub.AbortInventorySync(
                    kvcache_v2_pb2.AbortInventorySyncV2Request(
                        session=self._session(), sync_id=begin.sync_id
                    ),
                    timeout=self.rpc_timeout,
                )
                return self.sync_inventory(blocks, effective_page_size)
            for page_id, page in enumerate(pages):
                uploaded = self.v2_stub.UploadInventoryPage(
                    kvcache_v2_pb2.UploadInventoryPageV2Request(
                        session=self._session(),
                        sync_id=begin.sync_id,
                        page_id=page_id,
                        blocks=page,
                        page_checksum=self._inventory_checksum(page),
                    ),
                    timeout=self.rpc_timeout,
                )
                if not uploaded.accepted:
                    return False
            committed = self.v2_stub.CommitInventorySync(
                kvcache_v2_pb2.CommitInventorySyncV2Request(
                    session=self._session(), sync_id=begin.sync_id
                ),
                timeout=self.rpc_timeout,
            )
            self.sync_required = not committed.committed
            return bool(committed.committed)
        except grpc.RpcError as exc:
            logger.warning("GlobalKV V2 inventory sync failed: %s", exc)
            self.sync_required = True
            return False

    def report_request_start_v2(
        self, request_id: str, blocks: list[KvBlockMetadata]
    ) -> None:
        if self.v2_stub is None or self.instance_key is None:
            return
        try:
            self.v2_stub.ReportRequestStart(
                kvcache_v2_pb2.ReportRequestStartV2Request(
                    request=kvcache_v2_pb2.RequestIdentityV2(
                        instance=self._session().instance,
                        request_id=request_id,
                    ),
                    blocks=[self._proto_block(block) for block in blocks],
                ),
                timeout=self.rpc_timeout,
            )
        except grpc.RpcError as exc:
            logger.debug("GlobalKV request start dropped: %s", exc)

    def report_request_end_v2(self, request_id: str) -> None:
        if self.v2_stub is None or self.instance_key is None:
            return
        try:
            self.v2_stub.ReportRequestEnd(
                kvcache_v2_pb2.ReportRequestEndV2Request(
                    request=kvcache_v2_pb2.RequestIdentityV2(
                        instance=self._session().instance,
                        request_id=request_id,
                    )
                ),
                timeout=self.rpc_timeout,
            )
        except grpc.RpcError as exc:
            logger.debug("GlobalKV request end dropped: %s", exc)

    def _session(self):
        assert self.instance_key is not None
        return kvcache_v2_pb2.InstanceSessionV2(
            instance=kvcache_v2_pb2.InstanceIdentityV2(
                key=kvcache_v2_pb2.InstanceKeyV2(
                    lmcache_instance_id=self.instance_key.lmcache_instance_id,
                    worker_id=self.instance_key.worker_id,
                ),
                epoch=self.instance_epoch,
            ),
            lease_id=self.lease_id,
        )

    @staticmethod
    def _proto_block(block: KvBlockMetadata):
        return kvcache_v2_pb2.BlockDescriptorV2(
            seq_hash=block.seq_hash,
            parent_hash=block.parent_hash or b"",
            position=block.position,
            offset=block.offset,
            token_ids=block.token_ids,
        )

    @staticmethod
    def _inventory_checksum(blocks) -> bytes:
        digest = hashlib.sha256()
        for block in blocks:
            for value in (bytes(block.seq_hash), bytes(block.parent_hash)):
                digest.update(len(value).to_bytes(8, "big"))
                digest.update(value)
            digest.update(int(block.position).to_bytes(4, "big"))
            digest.update(int(block.offset).to_bytes(4, "big"))
            digest.update(len(block.token_ids).to_bytes(8, "big"))
            for token in block.token_ids:
                digest.update(int(token).to_bytes(4, "big"))
        return digest.digest()

    def close(self):
        """关闭连接"""
        if self.v2_stub is not None and self.instance_key is not None:
            try:
                self.v2_stub.UnregisterInstance(
                    kvcache_v2_pb2.UnregisterInstanceV2Request(
                        session=self._session()
                    ),
                    timeout=self.rpc_timeout,
                )
            except grpc.RpcError:
                pass
        if self.channel:
            self.channel.close()
            print("连接已关闭")
    
    def upload_kv_meta(self, tokens):
        # 构建请求消息
        request = kvcache_pb2.UploadKvMetaRequest(
            server_id=self.server_id,
            tokens=tokens
        )
        
        # 发送请求
        response = self.stub.UploadKvMeta(request)
        return response.success
    
    def remove_kv_meta(self, tokens_hash_list: List[bytes]) -> bool:
        """
        删除KV缓存元数据。

        Args:
            tokens_hash_list: token hash 列表，每个 hash 应为 32 字节 bytes。

        Returns:
            bool: 服务端是否处理成功。
        """
        request = kvcache_pb2.RemoveKvMetaRequest(
            id=self.server_id,
            remove_nums=len(tokens_hash_list),
            tokens_hash=tokens_hash_list,
        )
        response = self.stub.RemoveKvMeta(request)
        return bool(response.success)

    def register_instance(self):
        """
        注册推理实例（使用初始化时的数据服务器信息）

        Returns:
            bool: 是否成功
        """
        # 构建请求消息
        data_server = kvcache_pb2.DataServer(
            id=self.server_id,
            ip=self.advertised_host,
            http_port=self.data_server_http_port,
            init_port=self.data_server_init_port,
            rpc_port=self.data_server_rpc_port,
            model_name=self.model_name,
            url=(
                f"{self.api_scheme}://{self.api_host}:"
                f"{self.data_server_http_port}{self.api_path}"
            ),
        )
        
        request = kvcache_pb2.RegisterInstanceRequest(
            data_server=data_server
        )
        
        # 发送请求
        response = self.stub.RegisterInstance(request)
        return response.success

    def new_request(self, request_id: str, tokens: List[int]) -> bool:
        """
        向元数据服务器上报新请求开始（NewRequest）。

        Args:
            request_id: 请求唯一标识。
            tokens: 与该请求相关的 token id 列表。

        Returns:
            bool: 服务端是否处理成功。
        """
        request = kvcache_pb2.NewRequestRequest(
            server_id=self.server_id,
            request_id=request_id,
            tokens=tokens,
        )
        response = self.stub.NewRequest(request)
        return response.success

    def request_end(self, request_id: str, tokens: List[int]) -> bool:
        """
        向元数据服务器上报请求结束（RequestEnd）。

        Args:
            request_id: 请求唯一标识。
            tokens: 与该请求相关的 token id 列表。

        Returns:
            bool: 服务端是否处理成功。
        """
        request = kvcache_pb2.RequestEndRequest(
            server_id=self.server_id,
            request_id=request_id,
            tokens=tokens,
        )
        response = self.stub.RequestEnd(request)
        return response.success

    def kvblockmeta_from_key(self, tokens, start, end, kv_ref=1):
        """
        将tokens序列拆分为UploadKvBlockMeta列表

        Args:
            tokens: 完整的token序列列表
            start: 起始索引
            end: 结束索引（不包含）
            kv_ref: 引用计数，默认为1

        Returns:
            list: UploadKvBlockMeta字典列表，每个字典包含:
                {
                    'tokens': 子序列的token列表,
                    'kv_ref': 引用计数
                }
        """
        if start < 0 or end > len(tokens) or start >= end:
            raise ValueError(f"无效的索引范围: start={start}, end={end}, tokens长度={len(tokens)}")
        
        # 提取指定范围的tokens
        sub_tokens = tokens[start:end]
        
        # 返回单个UploadKvBlockMeta
        return {
            'tokens': sub_tokens,
            'kv_ref': kv_ref
        }
        
        
