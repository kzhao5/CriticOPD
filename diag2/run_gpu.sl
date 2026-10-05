#!/bin/bash
#SBATCH --account=als44
#SBATCH --ntasks=1 --cpus-per-task=8 --mem=96G --gres=gpu:1
#SBATCH --requeue --open-mode=append
# 诊断实验的单卡任务。同一任务会向多组分区各交一个副本:先开始的拿锁并取消其余排队副本。
set -uo pipefail
source /etc/profile.d/lmod.sh 2>/dev/null || true
module load miniforge3 2>/dev/null || true
command -v conda >/dev/null 2>&1 || source /vapps/rhel9/x86_64/miniforge3/25.3.1-0/etc/profile.d/conda.sh
eval "$(conda shell.bash hook)"; conda activate relay-opd
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_WORKER_MULTIPROC_METHOD=spawn OPENSSL_CONF=/dev/null HF_HUB_OFFLINE=1
_LC=${TMPDIR:-/tmp}/kzhao2_diag_${SLURM_JOB_ID:-$$}; mkdir -p "$_LC"; trap 'rm -rf "$_LC"' EXIT
export VLLM_CACHE_ROOT=$_LC/vllm TORCHINDUCTOR_CACHE_DIR=$_LC/inductor TRITON_CACHE_DIR=$_LC/triton
cd /home/kzhao2/Relay-OPD/diag2; mkdir -p out/locks
L=out/locks/$SLURM_JOB_NAME
if ! mkdir $L 2>/dev/null; then
  h=$(cat $L/job 2>/dev/null)
  if [ -f $L/done ]; then echo "已完成,退出"; exit 0; fi
  if [ -n "$h" ] && squeue -h -j $h -t R | grep -q .; then echo "副本 $h 在跑,退出"; exit 0; fi
  rm -rf $L; mkdir $L || exit 0
fi
echo $SLURM_JOB_ID > $L/job
# 过两分钟再取消其余排队副本:节点有时在作业启动 2 秒内把它 root 杀掉,立刻取消的话这一片就没人接了
( sleep 120; for j in $(squeue -h -u $USER -n $SLURM_JOB_NAME -t PD -o %i); do [ $j != $SLURM_JOB_ID ] && scancel $j; done ) &
echo "$(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader) :: python3 diag.py $CMD"
python3 diag.py $CMD && { touch $L/done; exit 0; }
# 失败(例如分到的卡被别的进程占满):释放锁,把自己重新排队(任务号不变,下游依赖不受影响),最多 3 次。
# 否则同名副本若已因「锁被占用」退出,这一片就没人接了(2026-10-05 d_b_cr0 在 cs-1-2 上就是这样丢的)
rm -rf $L
n=$(scontrol show job $SLURM_JOB_ID | grep -oE "Restarts=[0-9]+" | cut -d= -f2)
if [ "${n:-0}" -lt 3 ]; then echo "失败,重新排队(第 $((n + 1)) 次)"; scontrol requeue $SLURM_JOB_ID; sleep 60; fi
exit 1
