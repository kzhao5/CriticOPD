#!/bin/bash
#SBATCH --account=als44
#SBATCH --ntasks=1 --cpus-per-task=16 --mem=220G
#SBATCH --output=/nobackup/autodelete/usr/kzhao2/relay-opd/slurm_logs/%x_%j.log
set -uo pipefail
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES 2>/dev/null || true
source /etc/profile.d/lmod.sh 2>/dev/null || true
module load miniforge3 2>/dev/null || true
eval "$(conda shell.bash hook)"; conda activate relay-opd
export OPENSSL_CONF=/dev/null HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ALLREDUCE_USE_FLASHINFER=0
export MATH_GRADER_PATH=/home/kzhao2/Relay-OPD/relay-opd/opd/reward/grader
cd /home/kzhao2/Relay-OPD
for st in ${STAGES:?}; do
  echo "===== criticopd stage: $st ====="
  /home/kzhao2/.conda/envs/relay-opd/bin/python3 criticopd/pipeline.py $st ${EXTRA:-} || { echo "STAGE $st FAILED"; exit 1; }
done
echo "===== done ====="
