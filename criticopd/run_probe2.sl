#!/bin/bash
#SBATCH --account=als44
#SBATCH --ntasks=1 --cpus-per-task=8 --mem=96G --gres=gpu:1
#SBATCH --output=/home/kzhao2/Relay-OPD/logs/%x_%j.log
set -euo pipefail
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES 2>/dev/null || true
source /etc/profile.d/lmod.sh 2>/dev/null || true
module load miniforge3 2>/dev/null || true
eval "$(conda shell.bash hook)"; conda activate relay-opd
# 计算节点没有 nvcc:关掉需现场编译的 FlashInfer 采样核;主进程先 import 了训练代码,vLLM 子进程用 spawn
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_WORKER_MULTIPROC_METHOD=spawn OPENSSL_CONF=/dev/null HF_HUB_OFFLINE=1
_LC=${TMPDIR:-/tmp}/kzhao2_compile_${SLURM_JOB_ID:-$$}; mkdir -p "$_LC"; trap 'rm -rf "$_LC"' EXIT
export VLLM_CACHE_ROOT=$_LC/vllm TORCHINDUCTOR_CACHE_DIR=$_LC/inductor TRITON_CACHE_DIR=$_LC/triton
cd /home/kzhao2/Relay-OPD/criticopd
echo "CMD=$CMD $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader)"
python3 $CMD
# vLLM 偶尔在退出时卡住(结果早已写完):主动结束本进程树,避免下游依赖一直等
echo "===== done $(date +%H:%M:%S) ====="
