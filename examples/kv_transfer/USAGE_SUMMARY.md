# KV Cache Transfer 测试工具使用总结

## 📁 文件清单

已创建以下文件用于测试 KV cache 迁移效果：

### 核心文件
1. **`transfer.py`** - 主测试脚本
   - 完整的 KV cache 迁移性能测试工具
   - 基于 `benchmarks/multi_round_qa` 的测试框架
   - 测量 TTFT、吞吐量和总响应时间
   - 对比有无 KV cache 迁移的性能差异

2. **`analyze_results.py`** - 结果分析工具
   - 详细分析测试结果
   - 生成性能统计报告
   - 计算改进百分比和加速比

3. **`simple_transfer_example.py`** - 简单 gRPC 示例
   - 演示如何使用 gRPC 进行 KV cache 迁移
   - 可单独运行测试迁移功能

### 配置文件
4. **`example1.yaml`** - 源实例配置
5. **`example2.yaml`** - 目标实例配置
6. **`requirements.txt`** - Python 依赖

### 脚本文件
7. **`run_transfer_test.sh`** - 快速启动脚本
8. **`full_workflow_example.sh`** - 完整工作流示例

### 文档文件
9. **`TRANSFER_TEST_README.md`** - 详细使用文档
10. **`README.md`** - 更新后的主文档
11. **`USAGE_SUMMARY.md`** - 本文件

## 🚀 快速开始

### 1. 准备环境

```bash
# 安装依赖
cd /root/LMCache-adapter/LMCache/examples/kv_transfer
pip install -r requirements.txt
```

### 2. 启动两个 vLLM 实例

**终端 1 - 启动源实例：**
```bash
VLLM_CONFIGURE_LOGGING=0 PYTHONHASHSEED=0 UCX_TLS=tcp CUDA_VISIBLE_DEVICES=0 \
LMCACHE_CONFIG_FILE=example1.yaml \
vllm serve /path/to/model \
  --gpu-memory-utilization 0.9 \
  --port 8010 \
  --max-model-len 131072 \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'
```

**终端 2 - 启动目标实例：**
```bash
VLLM_CONFIGURE_LOGGING=0 PYTHONHASHSEED=0 UCX_TLS=tcp CUDA_VISIBLE_DEVICES=1 \
LMCACHE_CONFIG_FILE=example2.yaml \
vllm serve /path/to/model \
  --gpu-memory-utilization 0.9 \
  --port 8011 \
  --max-model-len 131072 \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'
```

### 3. 运行测试

**方法 A - 使用脚本：**
```bash
./run_transfer_test.sh
```

**方法 B - 直接运行：**
```bash
python3 transfer.py \
  --model Meta-Llama-3.1-8B-Instruct \
  --num-tests 5 \
  --shared-prompt-len 1000 \
  --unique-prompt-len 100 \
  --answer-len 100 \
  --verbose
```

### 4. 分析结果

```bash
python3 analyze_results.py transfer_results.csv
```

## 📊 测试原理

脚本会对每个测试迭代执行以下步骤：

### 测试流程图
```
Test N:
  ├─ 无迁移测试（冷启动）
  │   └─ 目标实例直接处理完整 prompt → 记录性能
  │
  └─ 有迁移测试（热启动）
      ├─ 源实例处理 prompt → 创建 KV cache
      ├─ 迁移 KV cache 到目标实例
      └─ 目标实例用已迁移的 cache 处理相同 prompt → 记录性能
```

### 性能指标
- **TTFT (Time To First Token)**: 首个 token 生成时间
- **Total Time**: 总响应时间
- **Throughput**: Token 吞吐量
- **Speedup**: 加速比（冷启动 vs 热启动）

## 📈 示例输出

### 控制台输出
```
======================================================================
PERFORMANCE SUMMARY
======================================================================

  WITHOUT KV Transfer (Target cold start):
    Average TTFT: 2.456s
    Average Total Time: 3.123s

  WITH KV Transfer (Target with cache):
    Average TTFT: 0.234s
    Average Total Time: 1.012s

  IMPROVEMENT:
    TTFT: 90.5% faster
    Total Time: 67.6% faster

======================================================================
```

### CSV 结果文件
| test_id | instance_id | with_transfer | ttft  | generation_time | total_time | prompt_tokens | generation_tokens |
|---------|-------------|---------------|-------|-----------------|------------|---------------|-------------------|
| 1       | target      | False         | 2.456 | 0.667           | 3.123      | 1100          | 100               |
| 1       | source      | True          | 2.401 | 0.651           | 3.052      | 1100          | 100               |
| 1       | target      | True          | 0.234 | 0.778           | 1.012      | 1100          | 100               |

## 🔧 高级用法

### 自定义参数

```bash
python3 transfer.py \
  --source-url http://localhost:8010/v1 \
  --source-grpc-port 8204 \
  --target-url http://localhost:8011/v1 \
  --target-grpc-port 8208 \
  --model your-model-name \
  --num-tests 10 \
  --shared-prompt-len 2000 \
  --unique-prompt-len 200 \
  --answer-len 200 \
  --output my_results.csv \
  --verbose
```

### 实际 KV Cache 迁移

在生产环境中，需要启用实际的 gRPC 迁移：

1. 从源实例日志中提取 cache hash
2. 在 `transfer.py` 的 `_test_with_transfer()` 方法中取消注释 gRPC 调用
3. 配置正确的 hash 值和端口

示例代码在 `simple_transfer_example.py` 中：
```python
success = send_transfer_kv_request(
    server_host="127.0.0.1",
    server_port=8204,
    hashes=["8e9e35b66155aa08..."],
    offsets=[256],
    position="LocalCPUBackend",
    target_ip="127.0.0.1",
    target_port=8208,
    do_copy=True
)
```

## 📚 相关文档

- `TRANSFER_TEST_README.md` - 详细文档和故障排除
- `benchmarks/multi_round_qa/README.md` - 原始测试框架文档
- `README.md` - P2P KV cache 共享说明

## 💡 使用建议

1. **首次测试**：使用较小的参数（如 `--num-tests 3`）快速验证
2. **性能测试**：使用较大的参数（如 `--num-tests 10`）获得更稳定的统计
3. **调试**：使用 `--verbose` 查看详细日志
4. **结果分析**：使用 `analyze_results.py` 获得详细的性能分析

## 🐛 常见问题

### 连接失败
```
Error: Connection refused
```
**解决**: 确保两个 vLLM 实例都在运行并监听正确的端口。

### gRPC 错误
```
Error: gRPC call failed
```
**解决**: 检查配置文件中的 gRPC 端口设置，确保防火墙允许连接。

### 性能未改进
**可能原因**:
- KV transfer 未正确配置
- 模型不一致
- NIXL 未正确安装

## 🎯 总结

该测试工具提供了一个完整的框架来评估 KV cache 迁移的性能改进。通过对比冷启动和热启动场景，可以清晰地看到 KV cache 迁移带来的性能提升。

主要优势：
- ✅ 完整的测试框架
- ✅ 详细的性能指标
- ✅ 易于使用的脚本
- ✅ 清晰的结果分析
- ✅ 基于成熟的 multi_round_qa 框架

开始测试 KV cache 迁移效果吧！🚀
