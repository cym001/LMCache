#!/usr/bin/env python3
"""
简单的 TransferKv gRPC 客户端示例
"""

import grpc
import sys
import os

# 添加服务器目录到路径
_server_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../../lmcache/v1/remote/server"
)
sys.path.insert(0, _server_dir)

import lmcache.v1.remote.server.kvserver_pb2 as kvserver_pb2
import lmcache.v1.remote.server.kvserver_pb2_grpc as kvserver_pb2_grpc


def send_transfer_kv_request(
    server_host="localhost",
    server_port=50052,
    hashes=None,
    offsets=None,
    position="LocalCPUBackend",
    target_ip="127.0.0.1",
    target_port=50053,
    do_copy=True
):
    """
    发送 TransferKv gRPC 请求的简单函数
    
    参数:
        server_host: 服务器地址
        server_port: 服务器端口
        hashes: 哈希值列表（支持整数或十六进制字符串）
        offsets: 偏移量列表（整数）
        position: 源位置
        target_ip: 目标IP
        target_port: 目标端口
        do_copy: True=复制, False=移动
    
    返回:
        bool: 成功返回True
    """
    # 默认值
    if hashes is None:
        hashes = [12345]
    if offsets is None:
        offsets = [100]
    
    # 将哈希值转换为字节（每个32字节）
    hash_bytes = b""
    for h in hashes:
        if isinstance(h, str):
            # 十六进制字符串格式（如 "8e9e35b66155aa08..."）
            hash_bytes += bytes.fromhex(h)
        elif isinstance(h, int):
            # 整数格式（使用大端序，32字节）
            hash_bytes += h.to_bytes(32, 'big')
        else:
            raise TypeError(f"不支持的哈希类型: {type(h)}, 应为 str 或 int")
    
    # 创建连接
    server_address = f"{server_host}:{server_port}"
    with grpc.insecure_channel(server_address) as channel:
        stub = kvserver_pb2_grpc.KvServerStub(channel)
        
        # 构造请求
        request = kvserver_pb2.TransferKvRequest(
            hash=hash_bytes,
            position=position,
            offset=offsets,
            target_ip=target_ip,
            target_port=target_port,
            do_copy=do_copy
        )
        
        # 发送请求
        try:
            response = stub.TransferKv(request, timeout=30.0)
            return response.success
        except grpc.RpcError as e:
            print(f"错误: {e.code()} - {e.details()}")
            return False


# 使用示例
if __name__ == "__main__":
    # 示例：发送一个简单的请求
    success = send_transfer_kv_request(
        server_host="127.0.0.1",
        server_port=8204,
        hashes=["8e9e35b66155aa08666e0446f69d387df09cdb8d85c4db9a50dda0bbfaf42cbe"],  # 示例哈希值
        offsets=[256],       # 100个tokens
        position="LocalCPUBackend",
        target_ip="127.0.0.1",
        target_port=8207,
        do_copy=True
    )
    
    print(f"传输{'成功' if success else '失败'}")

