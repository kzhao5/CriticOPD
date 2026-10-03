#!/bin/bash
# MC / 通用 benchmark 评测(greedy 1-sample pass@1). 用于 mmlu_pro / gpqa_diamond 及其 _mini.
#SBATCH --job-name=eval_mc
#SBATCH --output=/home/kzhao2/Relay-OPD/logs/%x_%j.log
#SBATCH --error=/home/kzhao2/Relay-OPD/logs/%x_%j.log
#SBATCH --account=als44
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --nodes=1
#SBATCH --time=6:00:00
set -xuo pipefail
# 崩溃/结束时清掉本 job 的所有后代(vLLM EngineCore 会成僵尸占满显存)
kill_descendants(){ local pids; pids=$(pstree -p $$ 2>/dev/null | grep -oE '\([0-9]+\)' | tr -d '()' | grep -v "^$$\$"); [ -n "$pids" ] && kill -9 $pids 2>/dev/null; true; }
trap kill_descendants EXIT
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES 2>/dev/null || true
source /etc/profile.d/lmod.sh 2>/dev/null || true
module load miniforge3 2>/dev/null || true
# 可选: 额外模块(如 cuda/12.8.1 给 B200 的 flashinfer JIT 提供 nvcc)
if [ -n "${EXTRA_MODULES:-}" ]; then module load ${EXTRA_MODULES} 2>/dev/null || true; fi
if command -v nvcc >/dev/null 2>&1; then export CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")}"; echo "[cuda] nvcc=$(command -v nvcc) CUDA_HOME=$CUDA_HOME"; fi
eval "$(conda shell.bash hook)"; conda activate relay-opd
export OPENSSL_CONF=/dev/null
export HF_HUB_OFFLINE=1
export MATH_GRADER_PATH=/home/kzhao2/Relay-OPD/relay-opd/opd/reward/grader
cd /home/kzhao2/Relay-OPD/relay-opd
RUN=${RUN:?set RUN}          # tag, e.g. opd_1p7b
MODEL=${MODEL:?set MODEL}    # checkpoint huggingface dir
BENCH=${BENCH:-mmlu_pro}
STEP=${STEP:-eval}
DP=${DP:-4}
MAXNEW=${MAXNEW:-8192}
[ -f "$MODEL/config.json" ] || { echo "!!! 缺 $MODEL"; exit 1; }
RUN_NAME=${RUN}_${BENCH} STEP=${STEP} MODEL=${MODEL} \
  DATA_DIR=/home/kzhao2/Relay-OPD/outputs/data/bench \
  OUT_ROOT=/home/kzhao2/Relay-OPD/outputs/eval_out \
  BENCHES=${BENCH} \
  N_SAMPLES=1 TEMPERATURE=0.0 TOP_P=1.0 MAX_NEW=${MAXNEW} MAX_MODEL_LEN=$((MAXNEW+4096)) DP_SIZE=${DP} TP=1 SEED=42 \
  bash opd/scripts/evaluation/math.sh
echo "===== MC EVAL DONE $RUN $BENCH ====="
