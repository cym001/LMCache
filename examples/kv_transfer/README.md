# P2P KV Cache Sharing and Transfer Testing
This directory contains examples and tools to demonstrate P2P KV cache sharing and transfer.

## Files
- `transfer.py`: Comprehensive test script for measuring KV cache transfer effectiveness
- `simple_transfer_example.py`: Simple gRPC transfer client example
- `analyze_results.py`: Script to analyze transfer test results
- `run_transfer_test.sh`: Quick start script for running tests
- `full_workflow_example.sh`: Complete workflow demonstration
- `example1.yaml`, `example2.yaml`: Configuration files for two vLLM instances
- `TRANSFER_TEST_README.md`: Detailed documentation for the transfer test script

## Prerequisites
Your server should have at least 2 GPUs.
[NIXL](https://github.com/ai-dynamo/nixl) should be installed as well. 

The LMCache controller will use port 8300 to pull messages from LMCache workers and port 8400 to reply requests to LMCache workers.

The two LMCache workers will use the port 8010 and 8011 for 2 vllms, port 8200 and 8202 for p2p initializations, and port 8201 and 8203 for p2p lookups.

## Quick Start: Testing KV Cache Transfer

To quickly test KV cache transfer effectiveness:

```bash
# 1. Start two vLLM instances (see steps below)
# 2. Run the transfer test
python3 transfer.py --model Meta-Llama-3.1-8B-Instruct --num-tests 5

# 3. Analyze results
python3 analyze_results.py transfer_results.csv
```

See `TRANSFER_TEST_README.md` for detailed documentation.

## Steps
1. Start the LMCache controller:
```bash
PYTHONHASHSEED=0 lmcache_controller --host 127.0.0.1 --port 9000 --monitor-ports '{"pull": 8300, "reply": 8400}'
``` 

2. Start two vllm engines (each with an LMCache worker):

Start vllm engine 1 at port 8010:
```bash
VLLM_CONFIGURE_LOGGING=0 PYTHONHASHSEED=0 UCX_TLS=tcp CUDA_VISIBLE_DEVICES=0 LMCACHE_CONFIG_FILE=example1.yaml vllm serve /root/autodl-tmp/model --gpu-memory-utilization 0.9 --port 8010 --max-model-len 131072 --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'
```
Start vllm engine 2 at port 8011:
```bash
VLLM_CONFIGURE_LOGGING=0 PYTHONHASHSEED=0 UCX_TLS=tcp CUDA_VISIBLE_DEVICES=1 LMCACHE_CONFIG_FILE=example2.yaml vllm serve /root/autodl-tmp/model  --gpu-memory-utilization 0.9 --port 8011 --max-model-len 131072 --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'  
```


VLLM_CONFIGURE_LOGGING=0 PYTHONHASHSEED=0 UCX_TLS=tcp CUDA_VISIBLE_DEVICES=0 LMCACHE_CONFIG_FILE=example1.yaml vllm serve /root/autodl-tmp/model --gpu-memory-utilization 0.9 --port 8010 --max-model-len 131072 --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}' --prefix-caching-hash-algo sha256_cbor

3. Send request to vllm engine 1:  
```bash
curl -X POST http://localhost:8010/v1/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"/root/autodl-tmp/model\",
    \"prompt\": \"$(printf 'you can Explain the significance of KV cache in large language models.%.0s' {1..100})\",
    \"max_tokens\": 10
  }"
```

4. Send request to vllm engine 2:  
```bash
curl -X POST http://localhost:8011/v1/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"/root/autodl-tmp/model\",
    \"prompt\": \"$(printf 'Explain the significance of KV cache in language models.%.0s' {1..100})\",
    \"max_tokens\": 10
  }"
```
The cache will be automatically retrieved from vllm engine 1.
You should be able to see logs (from vllm engine 2) like the following:
```bash
(EngineCore_DP0 pid=2577584)[2025-09-21 00:00:11,706] LMCache INFO:[0m Established connection to peer_init_url localhost:8200. The peer_lookup_url: localhost:8201 (p2p_backend.py:278:lmcache.v1.storage_backend.p2p_backend)
(EngineCore_DP0 pid=2577584)[2025-09-21 00:00:11,792] LMCache INFO: Retrieved 1002 out of total 1002 out of total 1002 tokens. size: 0.1223 gb, cost 60.3595 ms, throughput: 2.0264 GB/s; (cache_engine.py:496:lmcache.v1.cache_engine)
```


curl -X POST http://localhost:9000/Transfer \
  -H "Content-Type: application/json" \
  -d '{
        "hashes": ["-0x54109c6e7e64a629"],
        "old_position": ["lmcache_instance_1", "LocalCPUBackend"],
        "offsets": [256],
        "peer_ip": "localhost",
        "peer_init_port": 8207,
        "do_copy": true
      }'


curl -X POST http://localhost:9000/Transfer \
  -H "Content-Type: application/json" \
  -d '{
        "hashes": [-6057513497194178089, -8272346415788561506],
        "old_position": ["lmcache_instance_1", "LocalCPUBackend"],
        "offsets": [256, 256],
        "peer_ip": "localhost",
        "peer_init_port": 8207,
        "do_copy": true
      }'


curl -X POST http://localhost:9000/Transfer \
  -H "Content-Type: application/json" \
  -d '{
        "hashes": [64507957044576017690272399302942824267982388145928575038347140822825792974014, 12832674291997364242441430212481796351956584670323403819397134580660579170180],
        "old_position": ["lmcache_instance_1", "LocalCPUBackend"],
        "offsets": [256, 256],
        "peer_ip": "localhost",
        "peer_init_port": 8207,
        "do_copy": true
      }'



PYTHONHASHSEED=0 UCX_TLS=tcp CUDA_VISIBLE_DEVICES=0 LMCACHE_CONFIG_FILE=example1.yaml vllm serve /root/autodl-tmp/model --gpu-memory-utilization 0.9 --port 8010 --max-model-len 131072 --kv-cache-dtype fp8 --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'

Start vllm engine 1 at port 8010:
```bash
UCX_TLS=tcp CUDA_VISIBLE_DEVICES=0 LMCACHE_CONFIG_FILE=example1.yaml vllm serve /root/autodl-tmp/model --gpu-memory-utilization 0.5 --port 8010 --max-model-len 8192 --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'
```
Start vllm engine 2 at port 8011:
```bash
UCX_TLS=tcp CUDA_VISIBLE_DEVICES=1 LMCACHE_CONFIG_FILE=example2.yaml vllm serve /root/autodl-tmp/model  --gpu-memory-utilization 0.5 --port 8011 --max-model-len 8192 --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'  