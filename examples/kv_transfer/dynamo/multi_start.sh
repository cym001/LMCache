#!/bin/bash
set -e

# 为每次运行创建唯一的 Prometheus 多进程目录，避免多实例端口冲突
export PROMETHEUS_MULTIPROC_DIR=${PROMETHEUS_MULTIPROC_DIR:-/tmp/prometheus_multiproc_$$_$RANDOM}
rm -rf "$PROMETHEUS_MULTIPROC_DIR"
mkdir -p "$PROMETHEUS_MULTIPROC_DIR"

cleanup() {
    echo "Cleaning up..."
    rm -rf "$PROMETHEUS_MULTIPROC_DIR"
    kill 0
}
trap cleanup EXIT


cd /root/deploy

echo "清理旧进程和日志文件..."
pkill -u "$USER" cedfs-kv || true
rm -f *.log *.pid

sleep 1

echo "启动cedfs-kv服务..."
./cedfs-kv --path ./config.toml --log info > kv-1.log 2>&1 &
KV_PID=$!
echo $! > kv-1.pid
echo "cedfs-kv服务已启动 (PID: $KV_PID)"

# sleep 1

# -------------------------------
# ✅ 关键：关闭 etcd / NATS 依赖
# -------------------------------
export DYN_DISCOVERY_BACKEND=file
export DYN_EVENT_PLANE=zmq
# export NATS_SERVER=nats://127.0.0.1:4222
# export DYN_EVENT_PLANE=nats

# 保留唯一的 PROMETHEUS_MULTIPROC_DIR，各实例共享同一目录但不再冲突

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "/root/LMCache-adapter/LMCache/examples/kv_transfer/dynamo/gpu_utils.sh"
source "/root/LMCache-adapter/LMCache/examples/kv_transfer/dynamo/launch_utils.sh"

MODEL="/root/autodl-tmp/model"
LMCACHE_CONFIG_DIR="/root/LMCache/examples/kv_transfer"

# ---- Tunable ----
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_CONCURRENT_SEQS="${MAX_CONCURRENT_SEQS:-2}"

# 自动检测 GPU 数量，支持 NUM_GPUS 环境变量覆盖
if [ -n "$NUM_GPUS" ]; then
  GPU_COUNT=$NUM_GPUS
else
  GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l)
  if [ "$GPU_COUNT" -eq 0 ]; then
    echo "未检测到 GPU，请设置 NUM_GPUS 或检查 nvidia-smi"
    exit 1
  fi
fi

echo "检测到 GPU 数量：$GPU_COUNT"

# 清空旧日志
rm -f "${LMCACHE_CONFIG_DIR}"/log*.log
echo "已清空 ${LMCACHE_CONFIG_DIR} 下的旧日志"

GPU_MEM_FRACTION=$(build_gpu_mem_args vllm --model "$MODEL" --max-model-len "$MAX_MODEL_LEN" --max-num-seqs "$MAX_CONCURRENT_SEQS")

HTTP_PORT="${DYN_HTTP_PORT:-8000}"
print_launch_banner "Launching Aggregated Serving + LMCache (${GPU_COUNT} GPU)" "$MODEL" "$HTTP_PORT"

# -------------------------------
# ✅ frontend（必须加 file backend）
# -------------------------------
python -m dynamo.frontend --router-mode round-robin \
  --discovery-backend file &

# python -m dynamo.frontend 

# 为每个 GPU 启动一个 dynamo.vllm + LMCache 实例
BASE_SYSTEM_PORT="${DYN_SYSTEM_PORT:-8081}"
LAUNCHED=0

for ((i = 0; i < GPU_COUNT; i++)); do
  IDX=$((i + 1))
  CONFIG_FILE="${LMCACHE_CONFIG_DIR}/example${IDX}.yaml"

  if [ ! -f "$CONFIG_FILE" ]; then
    echo "警告: 配置文件不存在 $CONFIG_FILE，跳过 GPU $i"
    continue
  fi

  SYSTEM_PORT=$((BASE_SYSTEM_PORT + i))
  KV_EVENTS_PORT=$((20080 + i))
  LOG_FILE="/root/deploy/log${IDX}.log"
  echo "启动 GPU $i -> 配置 example${IDX}.yaml (DYN_SYSTEM_PORT=${SYSTEM_PORT}, KVEvents端口=${KV_EVENTS_PORT}, 日志: log${IDX}.log)"

  KV_EVENTS_CONFIG=$(printf '{"enable_kv_cache_events":true,"publisher":"zmq","endpoint":"tcp://*:%d","replay_endpoint":null,"buffer_steps":10000,"hwm":100000,"max_queue_size":100000,"topic":""}' "$KV_EVENTS_PORT")
  # KV_EVENTS_CONFIG=$(printf '{"enable_kv_cache_events":true,"publisher":"nats","topic":"kv-events"}' )

  CUDA_VISIBLE_DEVICES=$i \
  DYN_SYSTEM_PORT=${SYSTEM_PORT} \
  LMCACHE_CONFIG_FILE="${CONFIG_FILE}" \
  PROMETHEUS_MULTIPROC_DIR="$PROMETHEUS_MULTIPROC_DIR" \
  python -m dynamo.vllm \
    --model "$MODEL" \
    --enforce-eager \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_CONCURRENT_SEQS" \
    --discovery-backend file \
    ${GPU_MEM_FRACTION:+--gpu-memory-utilization "$GPU_MEM_FRACTION"} \
    --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}' \
    --kv-events-config "$KV_EVENTS_CONFIG" \
    > "${LOG_FILE}" 2>&1 &

  LAUNCHED=$((LAUNCHED + 1))
done

if [ "$LAUNCHED" -eq 0 ]; then
  echo "错误：没有可用的配置文件与 GPU 对应，退出"
  exit 1
fi

echo "已启动 ${LAUNCHED} 个 vllm+lmcache 实例"

# Exit on first worker failure
wait_any_exit