#!/bin/bash
#SBATCH --account=als44
#SBATCH --ntasks=1 --cpus-per-task=8 --mem=96G --gres=gpu:1
#SBATCH --output=/home/kzhao2/Relay-OPD/logs/%x_%j.log
set -euo pipefail
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES 2>/dev/null || true
source /etc/profile.d/lmod.sh 2>/dev/null || true
module load miniforge3 2>/dev/null || true
eval "$(conda shell.bash hook)"; conda activate relay-opd
# 计算节点没有 nvcc,FlashInfer 采样核无法现场编译(top-p 采样会触发);与 math.sh 一致关掉它
export VLLM_USE_FLASHINFER_SAMPLER=0
export OPENSSL_CONF=/dev/null HF_HUB_OFFLINE=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
_LC=${TMPDIR:-/tmp}/kzhao2_compile_${SLURM_JOB_ID:-$$}; mkdir -p "$_LC"; trap 'rm -rf "$_LC"' EXIT
export VLLM_CACHE_ROOT=$_LC/vllm TORCHINDUCTOR_CACHE_DIR=$_LC/inductor TRITON_CACHE_DIR=$_LC/triton
cd /home/kzhao2/Relay-OPD/criticopd
echo "shard $SHARD/$NSH  $(hostname)  $(nvidia-smi --query-gpu=name --format=csv,noheader)"
python3 gen_refs.py $SHARD $NSH
echo "===== done $(date +%H:%M:%S) ====="
