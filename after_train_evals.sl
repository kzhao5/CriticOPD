#!/bin/bash
# Chaser: submitted with --dependency=afterany:<train_jobid>, fires the moment a training run ends
# (success, failure or timeout) and queues that run's whole evaluation set. CPU-only and finishes in
# seconds, so it costs no GPU and -- unlike a polling watcher -- it survives client/session restarts.
#   sbatch --dependency=afterany:<jid> --export=ALL,RUN=<run> after_train_evals.sl
# Env: STEPS (default "40 80 120 139"), WHAT (math|sci|code|all, default math).
#SBATCH --job-name=chase_eval
#SBATCH --output=/home/kzhao2/Relay-OPD/logs/%x_%j.log
#SBATCH --error=/home/kzhao2/Relay-OPD/logs/%x_%j.log
#SBATCH --account=als44
#SBATCH --partition=m8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=0:20:00
set -uo pipefail
cd /home/kzhao2/Relay-OPD
RUN=${RUN:?set RUN}
# Include 20: on base students the un-fixed baselines can collapse (95-100% at cap) by step ~40,
# so their best checkpoint is an early one. Each method is reported at its own best step.
STEPS=${STEPS:-20 40 80 120 139}
WHAT=${WHAT:-math}
echo "[chase] run=$RUN steps='$STEPS' what=$WHAT"
# The final step can land anywhere near the end (139 = 1 epoch), so also take the newest checkpoint.
last=$(ls -d outputs/checkpoints/${RUN}/global_step_*/actor/huggingface 2>/dev/null \
       | sed -E 's#.*/global_step_([0-9]+)/.*#\1#' | sort -n | tail -1)
[ -n "$last" ] && STEPS="$STEPS $last"
for st in $(tr ' ' '\n' <<< "$STEPS" | sort -n -u); do
  [ -f "outputs/checkpoints/${RUN}/global_step_${st}/actor/huggingface/config.json" ] || continue
  echo "[chase] -> $RUN @ $st"
  bash eval_all_benches.sh "$RUN" "$st" "$WHAT" || echo "[chase] submit failed for $RUN@$st"
done
echo "===== CHASE DONE $RUN ====="
