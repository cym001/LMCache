# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass
from typing import List
import os
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
        
        # 生成server_id: ip+http_port的hash值
        id_str = f"{self.data_server_ip}:{self.data_server_http_port}"
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
        except (grpc.RpcError, RuntimeError) as exc:
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
                host=self.data_server_ip,
                http_port=self.data_server_http_port,
                nixl_init_port=self.data_server_init_port,
                transfer_rpc_port=self.data_server_rpc_port,
                api_path="/v1",
            ),
            fingerprint=fingerprint,
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
        except (grpc.RpcError, RuntimeError) as exc:
            if self.protocol == "v2":
                raise
            logger.warning("GlobalKV V2 registration failed; V1 remains active: %s", exc)
            self.v2_stub = None

    def report_stored_blocks(self, blocks: list[KvBlockMetadata]) -> None:
        if not blocks or self.v2_stub is None:
            return
        proto_blocks = [
            kvcache_v2_pb2.BlockDescriptorV2(
                seq_hash=block.seq_hash,
                parent_hash=block.parent_hash or b"",
                position=block.position,
                offset=block.offset,
                token_ids=block.token_ids,
            )
            for block in blocks
        ]
        self._report_mutation(store=kvcache_v2_pb2.StoreBlocksV2(blocks=proto_blocks))

    def report_removed_blocks(self, seq_hashes: list[bytes]) -> None:
        if not seq_hashes or self.v2_stub is None:
            return
        self._report_mutation(
            remove=kvcache_v2_pb2.RemoveBlocksV2(seq_hashes=seq_hashes)
        )

    def _report_mutation(self, **payload) -> None:
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
            response = self.v2_stub.ReportCacheMutations(
                request,
                timeout=self.rpc_timeout,
            )
            accepted = {
                kvcache_v2_pb2.MUTATION_STATUS_V2_COMMITTED,
                kvcache_v2_pb2.MUTATION_STATUS_V2_DUPLICATE,
            }
            if response.status not in accepted:
                raise RuntimeError(
                    "GlobalKV V2 mutation rejected: "
                    f"status={response.status} detail={response.error_detail}"
                )
        except (grpc.RpcError, RuntimeError) as exc:
            if self.protocol == "v2":
                raise
            logger.warning("GlobalKV V2 shadow mutation failed: %s", exc)
    
    def close(self):
        """关闭连接"""
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
            ip=self.data_server_ip,
            http_port=self.data_server_http_port,
            init_port=self.data_server_init_port,
            rpc_port=self.data_server_rpc_port,
            model_name=self.model_name,
            # url=f"http://localhost:{self.data_server_http_port[0]}/v1/completions"
            url=f"http://localhost:{self.data_server_http_port}/v1"
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
        
        
