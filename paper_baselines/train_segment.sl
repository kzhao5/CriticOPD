#!/bin/bash
#SBATCH --account=als44
#SBATCH --output=/home/kzhao2/Relay-OPD/logs/%x_%j.log
#SBATCH --error=/home/kzhao2/Relay-OPD/logs/%x_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#
# One training segment of a paper baseline. The same segment is submitted to several partitions (dw and
# cs/cs2/cs3); the first copy to start takes an atomic lock and cancels its queued siblings, the others exit.
# A later segment resumes from OUTPUT_DIR (verl trainer.resume_mode=auto) and exits at once if the run
# already reached TARGET_STEP. Env: everything run_method_freegpu.sl needs, plus SEG and TARGET_STEP.
set -uo pipefail
SEG=${SEG:?}; TARGET=${TARGET_STEP:-139}
mkdir -p "$OUTPUT_DIR"
latest=$(cat "$OUTPUT_DIR/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0)
if [ "${latest:-0}" -ge "$TARGET" ]; then
  echo "[segment] $EXP_ID seg$SEG: already at step $latest >= $TARGET, nothing to do"; exit 0
fi
# Check the cards BEFORE taking the lock. On 2026-09-15 cs-1-2 had a card held by a leftover process; the copy
# that landed there took the lock, cancelled its queued sibling on dw, then failed the same check inside
# run_method_freegpu.sl, and the next segment landed on the same node and did the same -- 7 runs lost in 3 min.
# A copy that cannot run now exits without the lock and leaves its siblings queued.
assigned="${CUDA_VISIBLE_DEVICES:-}"
free=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2+0<2048{print $1}' | tr '\n' ' ')
nfree=0; for g in ${assigned//,/ }; do [[ " $free " == *" $g "* ]] && nfree=$((nfree+1)); done
if [ "$nfree" -lt "${NEED_GPUS:-4}" ]; then
  echo "[segment] $EXP_ID seg$SEG: only $nfree of assigned ($assigned) free on $(hostname), exiting without the lock"
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader; exit 3
fi
if ! mkdir "$OUTPUT_DIR/.seg${SEG}.lock" 2>/dev/null; then
  if [ ! -d "$OUTPUT_DIR/.seg${SEG}.lock" ]; then   # mkdir failed with no lock present: filesystem not writable
    echo "[segment] !!! $EXP_ID seg$SEG: cannot create the lock on $(hostname) (quota / filesystem?)"; exit 2
  fi
  echo "[segment] $EXP_ID seg$SEG: lock held by $(cat "$OUTPUT_DIR/.seg${SEG}.lock/job" 2>/dev/null), exiting"; exit 0
fi
echo "$SLURM_JOB_ID $(hostname) $(date '+%F %T')" > "$OUTPUT_DIR/.seg${SEG}.lock/job"
for j in $(squeue -h -u "$USER" -n "$SLURM_JOB_NAME" -o %i); do
  [ "$j" != "$SLURM_JOB_ID" ] && scancel "$j" && echo "[segment] cancelled queued sibling $j"
done
echo "[segment] $EXP_ID seg$SEG starts on $(hostname) from step $latest"
bash /home/kzhao2/Relay-OPD/run_method_freegpu.sl; _rc=$?
# 训练已到目标步数:取消本运行还在排队的备用段,评测守护(依赖最后一段结束)就能马上开始
_latest=$(cat "$OUTPUT_DIR/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0)
if [ "${_latest:-0}" -ge "$TARGET" ]; then
  _pfx=${SLURM_JOB_NAME%_s*}_s
  for _j in $(squeue -h -u "$USER" -t PD -o "%i %j" | awk -v t="$_pfx" 'index($2,t)==1 {print $1}'); do scancel "$_j"; done
  echo "[segment] $EXP_ID 已训到 $_latest >= $TARGET,取消备用段"
fi
exit $_rc
