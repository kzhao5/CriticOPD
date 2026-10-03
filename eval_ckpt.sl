#!/bin/bash
# 评指定 checkpoint（best-checkpoint 扫描用）。快集：aime24/25,amc23,math500（跳过慢的 olympiad）
#SBATCH --job-name=ropd_ec
#SBATCH --output=/home/kzhao2/Relay-OPD/logs/%x_%j.log
#SBATCH --error=/home/kzhao2/Relay-OPD/logs/%x_%j.log
#SBATCH --account=als44
#SBATCH --partition=dw
#SBATCH --qos=dw87
#SBATCH --gres=gpu:a100:8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=700G
#SBATCH --nodes=1
#SBATCH --time=6:00:00
set -xuo pipefail
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES 2>/dev/null || true
source /etc/profile.d/lmod.sh 2>/dev/null || true
module load miniforge3 2>/dev/null || true
eval "$(conda shell.bash hook)"; conda activate relay-opd
export OPENSSL_CONF=/dev/null
RUN_NAME=${RUN_NAME:?}; STEP=${STEP:?}
MODEL=/home/kzhao2/Relay-OPD/outputs/checkpoints/${RUN_NAME}/global_step_${STEP}/actor/huggingface
[ -f "$MODEL/config.json" ] || { echo "!!! 缺 $MODEL"; exit 3; }
export MATH_GRADER_PATH=/home/kzhao2/Relay-OPD/relay-opd/opd/reward/grader
cd /home/kzhao2/Relay-OPD/relay-opd
echo "[EVAL-CKPT] $RUN_NAME step=$STEP model=$MODEL"
RUN_NAME=$RUN_NAME STEP=$STEP MODEL=$MODEL \
  DATA_DIR=/home/kzhao2/Relay-OPD/outputs/data/bench \
  OUT_ROOT=/home/kzhao2/Relay-OPD/outputs/eval_out \
  BENCHES=${BENCHES:-aime24,aime25,amc23,math500} \
  N_SAMPLES=32 MAX_NEW=32768 MAX_MODEL_LEN=34817 DP_SIZE=${EVAL_DP:-4} TP=1 \
  bash opd/scripts/evaluation/math.sh && echo "=== EVAL DONE $RUN_NAME step $STEP ==="
