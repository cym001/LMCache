# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import List

# Third Party
import grpc

# First Party
from lmcache.v1.config import LMCacheEngineConfig

try:
    from lmcache_kv_transfer.proto import kvcache_pb2
    from lmcache_kv_transfer.proto import kvcache_pb2_grpc
except ImportError as exc:
    raise ImportError(
        "GlobalKV proto modules are missing from lmcache_kv_transfer.proto"
    ) from exc


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
        success = self.register_instance()
        print(f"注册实例结果: {success}\n")
    
    def connect(self):
        """建立连接到元数据服务器"""
        self.channel = grpc.insecure_channel(f'{self.meta_server_host}:{self.meta_server_port}')
        self.stub = kvcache_pb2_grpc.KvMeta2DataStub(self.channel)
        print(f"已连接到元数据服务器 {self.meta_server_host}:{self.meta_server_port}")
        print(f"数据服务器ID: {self.server_id} (来自 {self.data_server_ip}:{self.data_server_http_port})")
    
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
        
        
