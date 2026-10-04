#!/bin/bash
#SBATCH --account=als44
#SBATCH --ntasks=1 --cpus-per-task=8 --mem=80G
#SBATCH --output=/nobackup/autodelete/usr/kzhao2/relay-opd/slurm_logs/%x_%j.log
# eval_base.sl 的看门狗永远盯 math500.summary.json,不管本 job 跑的是哪个 bench。
# 在一个 math500 已经跑完的目录里重跑别的 bench 时,它会在 15 秒内把 vLLM 杀掉。
# 这个 wrapper 直接调 math.sh,不带看门狗。
set -uo pipefail
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES 2>/dev/null || true
source /etc/profile.d/lmod.sh 2>/dev/null || true
module load miniforge3 2>/dev/null || true
if command -v nvcc >/dev/null 2>&1; then export CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")}"; fi
eval "$(conda shell.bash hook)"; conda activate relay-opd
export OPENSSL_CONF=/dev/null
export MATH_GRADER_PATH=/home/kzhao2/Relay-OPD/relay-opd/opd/reward/grader
cd /home/kzhao2/Relay-OPD/relay-opd
# Student / Teacher 是原始模型,没有 checkpoint 目录,用 MODEL_PATH 直接指定。
# 用独立的变量名而不是 MODEL —— --export=ALL 会把提交 shell 里残留的 MODEL 带进来。
MODEL=${MODEL_PATH:-/home/kzhao2/Relay-OPD/outputs/checkpoints/${RUN:?}/global_step_${STEP:?}/actor/huggingface}
[ -f "$MODEL/config.json" ] || { echo "!!! 缺 $MODEL"; exit 1; }
# 目标目录里若已有别的协议留下的分片,math.sh 会在**分片层**复用它们(顶层不打印 skip,
# 所以日志看不出来),然后以 dp_size=1 只聚合 shard_0 —— 产出 1/n 题数、协议错乱的数字,
# 还覆盖掉原来的完整结果。今晚这个坑连踩三次(opd_1p7b / lucid / relay+fastopd)。
# 这里直接拒绝:要么换 OUT_RUN 写新目录,要么先清理。
_OD=/home/kzhao2/Relay-OPD/outputs/eval_out/${OUT_RUN:-$RUN}/step_${STEP}
for _b in ${BENCHES//,/ }; do
  # 只查本 job 自己的分片:跨 job 分片时,别的分片先跑完是正常的,不能因此拒绝。
  if ls "$_OD"/shard_${SHARD_BASE:-0}/"$_b".summary.json >/dev/null 2>&1; then
    echo "!!! $_OD 已有 $_b 的旧分片 —— 拒绝运行,否则会复用旧协议的数据。"
    echo "    换一个 OUT_RUN(例如 ${RUN}_u),或先删除该目录。"
    exit 1
  fi
done
# 同一分片会同时向多组分区(不同 QOS)各交一个副本:先开始的拿锁并取消其余排队副本;
# 锁的持有者已不在队列里(例如被节点在 2 秒内杀掉)时,后来者接手。
mkdir -p "$_OD/shard_${SHARD_BASE:-0}"
_LOCK="$_OD/shard_${SHARD_BASE:-0}/.${BENCHES//,/_}.lock"
if ! mkdir "$_LOCK" 2>/dev/null; then
  _holder=$(cat "$_LOCK/job" 2>/dev/null)
  if [ -n "$_holder" ] && [ "$_holder" != "${SLURM_JOB_ID:-}" ] && squeue -h -j "$_holder" -t R 2>/dev/null | grep -q .; then
    echo "另一个副本 $_holder 正在跑这个分片,退出"; exit 0
  fi
  rm -rf "$_LOCK"; mkdir "$_LOCK" 2>/dev/null || { echo "抢锁失败,退出"; exit 0; }
fi
echo "${SLURM_JOB_ID:-}" > "$_LOCK/job"
for _j in $(squeue -h -u "$USER" -n "${SLURM_JOB_NAME:-none}" -t PD -o %i 2>/dev/null); do
  [ "$_j" != "${SLURM_JOB_ID:-}" ] && scancel "$_j" 2>/dev/null
done
echo "===== $RUN@$STEP shard $SHARD_BASE/${NUM_SHARDS_TOTAL} benches=${BENCHES} on $(hostname) ====="
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
RUN_NAME=${OUT_RUN:-$RUN} STEP=$STEP MODEL=$MODEL \
  DATA_DIR=/home/kzhao2/Relay-OPD/outputs/data/bench \
  OUT_ROOT=/home/kzhao2/Relay-OPD/outputs/eval_out \
  BENCHES=$BENCHES N_SAMPLES=${N_SAMPLES:-32} MAX_NEW=${MAX_NEW:-32768} \
  MAX_MODEL_LEN=${MAX_MODEL_LEN:-34817} DP_SIZE=1 TP=1 \
  SHARD_BASE=$SHARD_BASE NUM_SHARDS_TOTAL=$NUM_SHARDS_TOTAL \
  EVAL_STOP_TOKEN_IDS="${EVAL_STOP_TOKEN_IDS:-}" \
  bash opd/scripts/evaluation/math.sh
_rc=$?
echo "===== done rc=$_rc ====="
exit $_rc
