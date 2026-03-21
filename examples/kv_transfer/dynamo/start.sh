#!/bin/bash
set -e
trap 'echo Cleaning up...; kill 0' EXIT

# -------------------------------
# ✅ 关键：关闭 etcd / NATS 依赖
# -------------------------------
export DYN_DISCOVERY_BACKEND=file
export DYN_EVENT_PLANE=zmq

# Explicitly unset PROMETHEUS_MULTIPROC_DIR
unset PROMETHEUS_MULTIPROC_DIR

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "/root/LMCache-adapter/LMCache/examples/kv_transfer/dynamo/gpu_utils.sh"
source "/root/LMCache-adapter/LMCache/examples/kv_transfer/dynamo/launch_utils.sh"

MODEL="/root/autodl-tmp/model"

# ---- Tunable ----
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_CONCURRENT_SEQS="${MAX_CONCURRENT_SEQS:-2}"

GPU_MEM_FRACTION=$(build_gpu_mem_args vllm --model "$MODEL" --max-model-len "$MAX_MODEL_LEN" --max-num-seqs "$MAX_CONCURRENT_SEQS")

HTTP_PORT="${DYN_HTTP_PORT:-8000}"
print_launch_banner "Launching Aggregated Serving + LMCache (1 GPU)" "$MODEL" "$HTTP_PORT"

# -------------------------------
# ✅ frontend（必须加 file backend）
# -------------------------------
python -m dynamo.frontend \
  --discovery-backend file &

# -------------------------------
# ✅ worker（你原来基本是对的）
# -------------------------------
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT:-8081} LMCACHE_CONFIG_FILE=/root/LMCache-adapter/LMCache/examples/kv_transfer/example1.yaml \
python -m dynamo.vllm \
  --model "$MODEL" \
  --enforce-eager \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_CONCURRENT_SEQS" \
  --discovery-backend file \
  ${GPU_MEM_FRACTION:+--gpu-memory-utilization "$GPU_MEM_FRACTION"} \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}' &

# Exit on first worker failure
wait_any_exit