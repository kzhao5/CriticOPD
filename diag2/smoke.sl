#!/bin/bash
#SBATCH --account=als44
#SBATCH --ntasks=1 --cpus-per-task=16 --mem=96G --gres=gpu:1
set -euo pipefail
source /etc/profile.d/lmod.sh 2>/dev/null || true; module load miniforge3 2>/dev/null || true
command -v conda >/dev/null 2>&1 || source /vapps/rhel9/x86_64/miniforge3/25.3.1-0/etc/profile.d/conda.sh
eval "$(conda shell.bash hook)"; conda activate relay-opd
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_WORKER_MULTIPROC_METHOD=spawn OPENSSL_CONF=/dev/null HF_HUB_OFFLINE=1 CRITIC_OPD_GRADER_PROCS=12
_LC=${TMPDIR:-/tmp}/kzhao2_smk_${SLURM_JOB_ID}; mkdir -p "$_LC"; trap 'rm -rf "$_LC"' EXIT
export VLLM_CACHE_ROOT=$_LC/vllm TORCHINDUCTOR_CACHE_DIR=$_LC/inductor TRITON_CACHE_DIR=$_LC/triton
export DIAG_OUT=/home/kzhao2/Relay-OPD/diag2/smoke_out DIAG_NPROB=40 DIAG_NC=6 DIAG_NW=12
cd /home/kzhao2/Relay-OPD/diag2; rm -rf $DIAG_OUT; mkdir -p $DIAG_OUT
P() { echo "=== $*"; python3 diag.py "$@" 2>&1 | grep -vE "Warning|it/s\]|INFO|^\(EngineCore" | tail -3; }
P rollouts 0 1; P select
P cont teacher e1 0 1; P cont student e1 0 1; P grade e1; P closing
P score teacher 0 1; P score student 0 1; P critic 0 1; P merge_critic
P cont student e6 0 1; P cont teacher e6 0 1; P grade e6
P cont teacher e7 0 1; P cont student e7 0 1; P grade e7
ls -la $DIAG_OUT
