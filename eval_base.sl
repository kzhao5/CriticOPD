#!/bin/bash
# entropygap_v1 训练完在同一节点(dw-2-4)无缝评 @40/60/80,一个 job 拿住节点不释放。
#SBATCH --job-name=eval_base
#SBATCH --output=/nobackup/autodelete/usr/kzhao2/relay-opd/slurm_logs/%x_%j.log
#SBATCH --error=/nobackup/autodelete/usr/kzhao2/relay-opd/slurm_logs/%x_%j.log
#SBATCH --account=als44
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --nodes=1
#SBATCH --time=8:00:00
set -xuo pipefail
# 结束/崩溃时清掉本 job 全部后代(vLLM EngineCore 退出卡 futex 会把整卡占住数小时)
kill_tree(){ local pids; pids=$(pstree -p "$1" 2>/dev/null | grep -oE '\([0-9]+\)' | tr -d '()' | grep -v "^$1\$"); [ -n "$pids" ] && kill -9 $pids 2>/dev/null; true; }
trap 'kill_tree $$' EXIT
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES 2>/dev/null || true
source /etc/profile.d/lmod.sh 2>/dev/null || true
module load miniforge3 2>/dev/null || true
if [ -n "${EXTRA_MODULES:-}" ]; then module load ${EXTRA_MODULES} 2>/dev/null || true; fi
if command -v nvcc >/dev/null 2>&1; then export CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")}"; fi
eval "$(conda shell.bash hook)"; conda activate relay-opd
export OPENSSL_CONF=/dev/null
export MATH_GRADER_PATH=/home/kzhao2/Relay-OPD/relay-opd/opd/reward/grader
cd /home/kzhao2/Relay-OPD/relay-opd
RUN=${RUN:-entropygap_v1}
EVAL_TAG=${EVAL_TAG:-}
export EVAL_STOP_TOKEN_IDS=${EVAL_STOP_TOKEN_IDS:-}
run_eval() {
  local STEP=$1
  local MODEL=/home/kzhao2/Relay-OPD/outputs/checkpoints/${RUN}/global_step_${STEP}/actor/huggingface
  [ -f "$MODEL/config.json" ] || { echo "!!! 缺 $MODEL, 跳过"; return 0; }
  echo "===== [EVAL] $RUN @ $STEP ====="
  RUN_NAME=${RUN}${EVAL_TAG} STEP=$STEP MODEL=$MODEL \
    DATA_DIR=/home/kzhao2/Relay-OPD/outputs/data/bench \
    OUT_ROOT=/home/kzhao2/Relay-OPD/outputs/eval_out \
    BENCHES=${BENCHES:-aime24,aime25,amc23,math500} \
    N_SAMPLES=${N_SAMPLES:-32} MAX_NEW=${MAX_NEW:-32768} MAX_MODEL_LEN=${MAX_MODEL_LEN:-34817} DP_SIZE=${DP:-4} TP=1 \
    bash opd/scripts/evaluation/math.sh &
  local ep=$!
  # 看门狗(仅 DP=1 单 shard 模式): 最后一个 bench(math500) 的 summary 落盘 >180s 后进程还在 = 卡在 vLLM 退出 -> 杀子树
  local last_sum=/home/kzhao2/Relay-OPD/outputs/eval_out/${RUN}${EVAL_TAG}/step_${STEP}/shard_${SHARD_BASE:-0}/math500.summary.json
  while kill -0 $ep 2>/dev/null; do
    if [ "${DP:-4}" = 1 ] && [ -f "$last_sum" ] && [ $(( $(date +%s) - $(stat -c %Y "$last_sum") )) -gt 180 ]; then
      echo "[watchdog] $last_sum 已落盘 >180s 但 math.sh 未退出 -> 强杀 vLLM 子树"; kill_tree $ep; kill -9 $ep 2>/dev/null; break
    fi
    sleep 30
  done
  wait $ep 2>/dev/null || echo "!!! eval $RUN@$STEP 非零退出(可能被看门狗结束), 继续"
}
for st in ${STEPS:-40 60 80}; do run_eval $st; done
echo "===== ALL DONE ====="
