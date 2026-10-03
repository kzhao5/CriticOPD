#!/bin/bash
# 训练结束后在同节点无缝 eval（afterany + nodelist 由 sbatch 命令行注入）
#SBATCH --job-name=ropd_eval
#SBATCH --output=/home/kzhao2/Relay-OPD/logs/%x_%j.log
#SBATCH --error=/home/kzhao2/Relay-OPD/logs/%x_%j.log
#SBATCH --account=als44
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=700G
#SBATCH --nodes=1
#SBATCH --time=12:00:00
set -xuo pipefail
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES 2>/dev/null || true
source /etc/profile.d/lmod.sh 2>/dev/null || true
module load miniforge3 2>/dev/null || true
eval "$(conda shell.bash hook)"; conda activate relay-opd
export OPENSSL_CONF=/dev/null

TRAIN_OUTPUT_DIR=${TRAIN_OUTPUT_DIR:?}
RUN_NAME=${RUN_NAME:?}
BENCH=${BENCH:-/home/kzhao2/Relay-OPD/outputs/data/bench}
BENCHES=${BENCHES:-aime24,aime25,amc23,math500,olympiad}
OUT_ROOT=${OUT_ROOT:-/home/kzhao2/Relay-OPD/outputs/eval_out}
export MATH_GRADER_PATH=/home/kzhao2/Relay-OPD/relay-opd/opd/reward/grader
cd /home/kzhao2/Relay-OPD/relay-opd

# 找最新 checkpoint 的 HF 权重（按 step 数值排序；路径多下划线，必须抽出数字排）
CKPT=$(ls -d ${TRAIN_OUTPUT_DIR}/global_step_*/actor/huggingface 2>/dev/null \
  | sed -E 's#.*/global_step_([0-9]+)/.*#\1\t&#' | sort -n | tail -1 | cut -f2-)
if [ -z "$CKPT" ] || [ ! -f "$CKPT/config.json" ]; then
  echo "!!! 找不到 HF checkpoint（试其它布局）"; find ${TRAIN_OUTPUT_DIR} -maxdepth 4 -name config.json 2>/dev/null | head; exit 3
fi
STEP=$(echo "$CKPT" | grep -oE "global_step_[0-9]+" | grep -oE "[0-9]+")
echo "[EVAL] run=$RUN_NAME step=$STEP model=$CKPT benches=$BENCHES"

RUN_NAME=$RUN_NAME STEP=$STEP MODEL=$CKPT DATA_DIR=$BENCH OUT_ROOT=$OUT_ROOT \
  BENCHES=$BENCHES N_SAMPLES=${N_SAMPLES:-32} MAX_NEW=${MAX_NEW:-32768} \
  MAX_MODEL_LEN=${MAX_MODEL_LEN:-34817} DP_SIZE=${DP_SIZE:-4} TP=${TP:-1} \
  bash opd/scripts/evaluation/math.sh && echo "=== EVAL DONE $RUN_NAME ==="
