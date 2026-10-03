#!/bin/bash
#SBATCH --job-name=ropd
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
#SBATCH --time=24:00:00
set -xeuo pipefail
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES 2>/dev/null || true   # 集群同时设ROCR+CUDA，verl拒启
source /etc/profile.d/lmod.sh 2>/dev/null || true
module load miniforge3 2>/dev/null || true
eval "$(conda shell.bash hook)"; conda activate relay-opd
export OPENSSL_CONF=/dev/null   # FIPS 兜底（opencv 已卸）

export STUDENT_MODEL=${STUDENT_MODEL:-/home/kzhao2/OPD/model/Qwen3-1.7B}
export TEACHER_MODEL=${TEACHER_MODEL:-/home/kzhao2/OPD/model/Qwen3-4B-Instruct-2507}
export TRAIN_DATA=${TRAIN_DATA:-/home/kzhao2/OPD/datasets/dapo-math-17k.parquet}
export BENCH=${BENCH:-/home/kzhao2/Relay-OPD/outputs/data/bench}
export OUTPUT_DIR=${OUTPUT_DIR:?OUTPUT_DIR required}
export EXP_ID=${EXP_ID:-$(basename "$OUTPUT_DIR")}
# 控盘：少存优化器/降存频，避免撑爆
export SAVE_FREQ=${SAVE_FREQ:-10}
export TEST_FREQ=${TEST_FREQ:-5}
export ACTOR_GPUS_PER_NODE=${ACTOR_GPUS_PER_NODE:-4}
export TEACHER_GPUS_PER_NODE=${TEACHER_GPUS_PER_NODE:-4}
cd /home/kzhao2/Relay-OPD/relay-opd
echo "[LAUNCH] method_script=$METHOD_SCRIPT exp=$EXP_ID out=$OUTPUT_DIR extra=${EXTRA_ARGS:-}"
bash "$METHOD_SCRIPT" ${EXTRA_ARGS:-}
