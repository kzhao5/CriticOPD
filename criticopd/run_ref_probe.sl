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
export OPENSSL_CONF=/dev/null HF_HUB_OFFLINE=1
# ref_probe 在主进程里 import 了训练代码(verl),会先初始化 CUDA;vLLM 默认用 fork 起引擎子进程,
# fork 出的子进程不能重新初始化 CUDA,所以改用 spawn。
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# 编译缓存放节点本地,避免多个 job 并发写 NFS 上的 ~/.cache/vllm(Errno 521/116)
_LC=${TMPDIR:-/tmp}/kzhao2_compile_${SLURM_JOB_ID:-$$}; mkdir -p "$_LC"; trap 'rm -rf "$_LC"' EXIT
export VLLM_CACHE_ROOT=$_LC/vllm TORCHINDUCTOR_CACHE_DIR=$_LC/inductor TRITON_CACHE_DIR=$_LC/triton
cd /home/kzhao2/Relay-OPD/criticopd
echo "OUT=${PROBE_OUT:-out_ref} N_PROB=${PROBE_N_PROB:-500} N_FP=${PROBE_N_FP:-150}  $(hostname)  $(nvidia-smi --query-gpu=name --format=csv,noheader)"
# 每个阶段单独一个进程:vLLM 不会把显存还干净,换模型前必须让进程退出
for S in ${STAGES:-rollout ref critic repair report}; do
  echo "===== stage $S  $(date +%H:%M:%S) ====="
  python3 ref_probe.py $S
done
echo "===== done $(date +%H:%M:%S) ====="
