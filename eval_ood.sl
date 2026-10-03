#!/bin/bash
#SBATCH --account=als44
#SBATCH --ntasks=1 --cpus-per-task=16 --mem=220G
#SBATCH --output=/nobackup/autodelete/usr/kzhao2/relay-opd/slurm_logs/%x_%j.log
# OOD 评测。和 eval_shard.sl 的区别:允许直接给 MODEL 路径(student/teacher 没有 checkpoint 目录)。
set -uo pipefail
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES 2>/dev/null || true
source /etc/profile.d/lmod.sh 2>/dev/null || true
module load miniforge3 2>/dev/null || true
if command -v nvcc >/dev/null 2>&1; then export CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")}"; fi
eval "$(conda shell.bash hook)"; conda activate relay-opd
export OPENSSL_CONF=/dev/null HF_HUB_OFFLINE=1
export MATH_GRADER_PATH=/home/kzhao2/Relay-OPD/relay-opd/opd/reward/grader
cd /home/kzhao2/Relay-OPD/relay-opd
MODEL=${MODEL:-/home/kzhao2/Relay-OPD/outputs/checkpoints/${RUN:?}/global_step_${STEP:?}/actor/huggingface}
[ -f "$MODEL/config.json" ] || { echo "!!! 缺 $MODEL"; exit 1; }
echo "===== OOD ${RUN_NAME:?} bench=${BENCHES:?} shard ${SHARD_BASE:-0}/${NUM_SHARDS_TOTAL:-1} ====="
echo "      model=$MODEL"
# 编译缓存放节点本地:多个评测 job 同时写共享的 ~/.cache/vllm(NFS)会偶发 Errno 521/116,
# vLLM 引擎初始化直接失败(2026-10-02 一小时内 3 个 job)。每个 job 用自己的本地目录。
_LC=${TMPDIR:-/tmp}/kzhao2_compile_${SLURM_JOB_ID:-$$}
mkdir -p "$_LC"; trap 'rm -rf "$_LC"' EXIT
export VLLM_CACHE_ROOT=$_LC/vllm TORCHINDUCTOR_CACHE_DIR=$_LC/inductor TRITON_CACHE_DIR=$_LC/triton
# 抢占来的卡,上一个(被抢占的)job 的显存可能还没释放。直接起 vLLM 会拿不到 90% 显存、
# 一分钟内失败(cs-1-2 上 13947223 / 13947383 就是这样)。最多等 GPU_FREE_WAIT 秒。
_dl=$(( $(date +%s) + ${GPU_FREE_WAIT:-180} ))
while :; do
  _busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk '$1+0>=2048' | wc -l)
  [ "$_busy" -eq 0 ] && break
  [ "$(date +%s)" -ge "$_dl" ] && { echo "!!! 分到的 GPU 在 ${GPU_FREE_WAIT:-180}s 内仍未释放显存:"; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader; exit 75; }
  echo "[gpu-wait] $_busy 张卡仍有显存占用,等待释放..."; sleep 10
done
RUN_NAME=$RUN_NAME STEP=${STEP:-0} MODEL=$MODEL \
  DATA_DIR=/home/kzhao2/Relay-OPD/outputs/data/bench \
  OUT_ROOT=/home/kzhao2/Relay-OPD/outputs/eval_out \
  BENCHES=$BENCHES N_SAMPLES=${N_SAMPLES:-4} MAX_NEW=${MAX_NEW:-8192} \
  MAX_MODEL_LEN=${MAX_MODEL_LEN:-10241} DP_SIZE=1 TP=1 \
  SHARD_BASE=${SHARD_BASE:-0} NUM_SHARDS_TOTAL=${NUM_SHARDS_TOTAL:-1} \
  EVAL_STOP_TOKEN_IDS="${EVAL_STOP_TOKEN_IDS:-}" \
  bash opd/scripts/evaluation/math.sh
_rc=$?
echo "===== done rc=$_rc ====="
exit $_rc
