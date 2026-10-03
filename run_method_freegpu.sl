#!/bin/bash
#SBATCH --job-name=ropd_freegpu
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
# B200 (cs3, sm100): flashinfer JIT compiles kernels at runtime and needs a matching nvcc.
# EXTRA_MODULES="cuda/12.8.1" supplies it; harmless on A100 nodes where it is unset.
if [ -n "${EXTRA_MODULES:-}" ]; then module load ${EXTRA_MODULES} 2>/dev/null || true; fi
if command -v nvcc >/dev/null 2>&1; then export CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")}"; echo "[cuda] nvcc=$(command -v nvcc) CUDA_HOME=$CUDA_HOME"; fi
eval "$(conda shell.bash hook)"; conda activate relay-opd
export OPENSSL_CONF=/dev/null   # FIPS 兜底（opencv 已卸）
# H200/sm90:环境自带的 nvidia-nccl-cu13 2.28.9 在 sm90 上第一个 all_reduce 就 segfault
# (job 13937117 实测:2.28.9 rc=139,2.29.7 与 2.32.3 rc=0)。torch 2.11 依赖 2.28+ 新增的
# ncclDevCommDestroy 符号,所以只能往上换,不能退回 2.27.3。A100(sm80)不受影响,保持原装,
# 免得扰动主表那些 run 的配置。
_cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
if [ -n "${_cap:-}" ] && [ "$_cap" -ge 90 ]; then
  _nccl=/home/kzhao2/Relay-OPD/criticopd/nccl_alt/2.29.7/nvidia/nccl/lib/libnccl.so.2
  if [ -f "$_nccl" ]; then
    export LD_PRELOAD="$_nccl${LD_PRELOAD:+:$LD_PRELOAD}"
    echo "[nccl] sm$_cap -> 预加载 NCCL 2.29.7 (自带 2.28.9 在 sm90 上 segfault)"
  fi
fi
# --- no cgroup GPU isolation: verify the Slurm-assigned cards are free (<2GB used); otherwise take free ones ---
want=${NEED_GPUS:-$(( ${ACTOR_GPUS_PER_NODE:-4} + ${TEACHER_GPUS_PER_NODE:-4} ))}
assigned="${CUDA_VISIBLE_DEVICES:-}"; free=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2+0<2048{print $1}' | tr '\n' ' ')
pick=""; for g in ${assigned//,/ }; do [[ " $free " == *" $g "* ]] && pick="$pick $g"; done
# only borrow non-assigned cards when BORROW_GPUS=1 (a card that is free by memory may still belong to another job)
if [ "${BORROW_GPUS:-0}" = 1 ]; then for g in $free; do [ $(wc -w <<< "$pick") -ge $want ] && break; [[ " $pick " == *" $g "* ]] || pick="$pick $g"; done; fi
pick=$(echo $pick | tr ' ' ','); echo "[gpu] assigned=$assigned free=($free) need=$want -> use $pick"
[ $(tr ',' ' ' <<< "$pick" | wc -w) -ge $want ] || { echo "!!! not enough free GPUs on $(hostname) (need $want)"; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader; exit 3; }
export CUDA_VISIBLE_DEVICES=$pick

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
# Ray 的 raylet 注册超时是写死的 30 秒(ray/_private/node.py,没有 env 旋钮)。m13h 这类
# 86/96 核被别人占满的节点上会偶发失败并抛 "node timed out during startup" —— 偶发,不是必然
# (m13h-1-2 过了,m13h-2-2 两次没过)。这一步失败时什么都还没写,所以原地重试是安全的。
set +e
# 次数也从文件读:环境变量在 sbatch 时定死,但这个文件能随时改(只影响还没进循环的 job)。
_tries=${LAUNCH_RETRIES:-$(cat /home/kzhao2/Relay-OPD/criticopd/.launch_retries 2>/dev/null || echo 5)}
for _i in $(seq 1 $_tries); do
  bash "$METHOD_SCRIPT" ${EXTRA_ARGS:-}; _rc=$?
  [ $_rc -eq 0 ] && break
  # 只对"还没开始训练"的启动期故障重试;已经落过 checkpoint 就交给上层的链式接续
  if [ -e "$OUTPUT_DIR/latest_checkpointed_iteration.txt" ]; then
    echo "[retry] 已有 checkpoint,不重试启动 (rc=$_rc)"; break
  fi
  [ $_i -lt $_tries ] && { echo "[retry] 启动期失败 rc=$_rc,第 $_i/$_tries 次,60 秒后重试"; sleep 60; }
done
exit $_rc
