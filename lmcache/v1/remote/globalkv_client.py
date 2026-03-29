import grpc
from typing import List, Dict, Optional, Union
import logging
import torch
import hashlib
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig

try:
    import lmcache.v1.remote.kvcache_pb2 as kvcache_pb2
    import lmcache.v1.remote.kvcache_pb2_grpc as kvcache_pb2_grpc
except ImportError:
    raise ImportError(
        "请先生成proto文件的Python代码:\n"
        "python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. kvcache.proto"
    )


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
        # self.data_server_ip = "127.0.0.1"
        # self.data_server_http_port = [8010]
        # self.data_server_init_port = [8203]
        # self.data_server_rpc_port = [8204]
        # self.model_name = "test"
        
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
    
    def register_instance(self):
        """
        注册推理实例（使用初始化时的数据服务器信息）
        
        Returns:
            bool: 是否成功
        """
        
        request = kvcache_pb2.RegisterInstanceRequest(
            worker_id=self.server_id,
            dp_rank=0,
            ip=self.data_server_ip,
            http_port=self.data_server_http_port,
            init_port=self.data_server_init_port,
            rpc_port=self.data_server_rpc_port,
            model_name=self.model_name,
            url=f"http://localhost:{self.data_server_http_port}/v1"
        )
        
        # 发送请求
        response = self.stub.RegisterInstance(request)
        return response.success

        
        
