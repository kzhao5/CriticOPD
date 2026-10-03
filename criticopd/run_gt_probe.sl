#!/bin/bash
#SBATCH --account=als44
#SBATCH --ntasks=1 --cpus-per-task=16 --mem=120G
#SBATCH --output=/home/kzhao2/Relay-OPD/logs/%x_%j.log
set -euo pipefail
source ~/.bashrc; conda activate relay-opd
cd /home/kzhao2/Relay-OPD/criticopd
export CRITIC_OPD_OUT=/home/kzhao2/Relay-OPD/criticopd/out_gt
export CRITIC_OPD_GIVE_GT=1
echo "=== GT 探针:同一批 rollout,只给 critic 正确答案 ==="
echo "    OUT=$CRITIC_OPD_OUT  GIVE_GT=$CRITIC_OPD_GIVE_GT"
python3 pipeline.py critic  --seg-mode para_merge --max-items 400
python3 pipeline.py repair  --run opd_1p7b --step 40 --repair-k 3
python3 pipeline.py report
