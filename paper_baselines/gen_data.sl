#!/bin/bash
#SBATCH --account=als44
#SBATCH --output=/home/kzhao2/Relay-OPD/logs/%x_%j.log
#SBATCH --error=/home/kzhao2/Relay-OPD/logs/%x_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=350G
#
# Offline data for the Relay-OPD paper baselines that train on fixed trajectories, using the paper's own
# synthesis scripts (relay-opd/opd/scripts/data/):
#   MODE=teacher  one teacher trajectory per DAPO prompt       -> SFT and SeqKD
#   MODE=trd      student trajectory + teacher rewrite          -> TRD (depends on the student, one per pair)
# Env: MODE, STUDENT_MODEL, OUT_PARQUET, optional SMOKE_ROWS / MAX_NEW for a tiny smoke dataset.
# Shards are resumable only with the same shard count, so the job insists on exactly NEED_GPUS free cards.
set -uo pipefail
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES 2>/dev/null || true
source /etc/profile.d/lmod.sh 2>/dev/null || true
module load miniforge3 2>/dev/null || true
# B200 (cs3, sm100): flashinfer JIT compiles kernels at runtime and needs a matching nvcc.
# EXTRA_MODULES="cuda/12.8.1" supplies it; harmless on A100 nodes where it is unset.
if [ -n "${EXTRA_MODULES:-}" ]; then module load ${EXTRA_MODULES} 2>/dev/null || true; fi
if command -v nvcc >/dev/null 2>&1; then export CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")}"; echo "[cuda] nvcc=$(command -v nvcc) CUDA_HOME=$CUDA_HOME"; fi
eval "$(conda shell.bash hook)"; conda activate relay-opd
export OPENSSL_CONF=/dev/null
MODE=${MODE:?MODE=teacher|trd}
STUDENT_MODEL=${STUDENT_MODEL:?}
OUT_PARQUET=${OUT_PARQUET:?}
export TEACHER_MODEL=${TEACHER_MODEL:-/home/kzhao2/OPD/model/Qwen3-4B-Instruct-2507}
SOURCE=${SOURCE_DATA:-/home/kzhao2/OPD/datasets/dapo-math-17k.parquet}
need=${NEED_GPUS:-4}

# no cgroup GPU isolation on this cluster: use only Slurm-assigned cards that are actually free
assigned="${CUDA_VISIBLE_DEVICES:-}"
free=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2+0<2048{print $1}' | tr '\n' ' ')
pick=""; for g in ${assigned//,/ }; do [[ " $free " == *" $g "* ]] && pick="$pick $g"; done
echo "[gpu] assigned=$assigned free=($free) need=$need -> ($pick)"
[ "$(wc -w <<< "$pick")" -ge "$need" ] || { echo "!!! not enough free GPUs on $(hostname)"; nvidia-smi; exit 3; }
GROUPS_STR=$(tr ' ' '\n' <<< "$pick" | grep . | head -n "$need" | paste -sd';')

mkdir -p "$(dirname "$OUT_PARQUET")"
if [ -n "${SMOKE_ROWS:-}" ]; then   # tiny prefix of the real data, for smoke tests only
  SRC_SMOKE="$(dirname "$OUT_PARQUET")/source_smoke_${SMOKE_ROWS}.parquet"
  python -c "import pandas as pd; pd.read_parquet('$SOURCE').head(${SMOKE_ROWS}).to_parquet('$SRC_SMOKE')"
  SOURCE=$SRC_SMOKE
fi
# base students declare only <|endoftext|>; register <|im_end|> too, as the rollouts and evals of this table do
export GEN_STOP_TOKEN_IDS='151643;151645'
export MAX_NEW=${MAX_NEW:-16384}
cd /home/kzhao2/Relay-OPD/relay-opd
echo "[gen] mode=$MODE student=$STUDENT_MODEL teacher=$TEACHER_MODEL source=$SOURCE out=$OUT_PARQUET groups=$GROUPS_STR max_new=$MAX_NEW"
if [ "$MODE" = teacher ]; then
  STUDENT_MODEL=$STUDENT_MODEL SOURCE_DATA=$SOURCE OUTPUT_PARQUET=$OUT_PARQUET GPU_GROUPS="$GROUPS_STR" \
    bash opd/scripts/data/teacher_trajectories.sh
else
  STUDENT_MODEL=$STUDENT_MODEL SOURCE_DATA=$SOURCE OUTPUT_PARQUET=$OUT_PARQUET \
    STUDENT_GPU_GROUPS="$GROUPS_STR" REWRITE_GPU_GROUPS="$GROUPS_STR" \
    bash opd/scripts/data/trd_trajectories.sh
fi
rc=$?
[ -s "$OUT_PARQUET" ] && python -c "import pandas as pd; d=pd.read_parquet('$OUT_PARQUET'); print('[gen] rows', len(d), 'eos_rate', round(d.eos_reached.mean(),4), 'gen_len mean', round(d.gen_len.mean(),1))"
echo "===== GEN DATA DONE mode=$MODE rc=$rc ====="
exit $rc
