#!/bin/bash
# LiveCodeBench 打分(CPU, codeeval 环境, 离线读缓存). 用法: sbatch --export=ALL,RUN=<run>,STEP=<step> score_lcb.sl
#SBATCH --job-name=lcbscore
#SBATCH --output=/home/kzhao2/Relay-OPD/logs/%x_%j.log
#SBATCH --error=/home/kzhao2/Relay-OPD/logs/%x_%j.log
#SBATCH --account=als44
#SBATCH --partition=m8
#SBATCH --qos=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=3:00:00
set -uo pipefail
module load miniforge3 2>/dev/null || true; eval "$(conda shell.bash hook)"; conda activate codeeval
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd /home/kzhao2/Relay-OPD
python newbench/score_lcb.py "${RUN:?}" "${STEP:?}" "${LCB_RELEASE:-release_v6}"
echo "===== LCB SCORE DONE $RUN@$STEP ====="
