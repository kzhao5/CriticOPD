#!/bin/bash
#SBATCH --account=als44
#SBATCH --partition=m8
#SBATCH --qos=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=3-00:00:00
#SBATCH --job-name=pb_reap_done
#SBATCH --output=/home/kzhao2/Relay-OPD/logs/%x_%j.log
#
# A run that reached step 139 in its first segment leaves its later segment copies queued. They would each wait for a
# 4-GPU slot only to exit immediately ("already at 139"), and the eval chaser -- which depends on the LAST segment --
# waits for them. This cancels those queued copies, which releases the chaser (afterany treats a cancelled job as
# finished) and stops the no-op jobs from taking GPU slots from real training.
# Every 10 min: for each paper-baseline run, if latest_checkpointed_iteration >= 139 and no pb_<run>_s* job is RUNNING,
# cancel its PENDING pb_<run>_s* jobs.
set -uo pipefail
cd /home/kzhao2/Relay-OPD
RUNS="fastopd1024_p1 fastopd2048_p1 fastopd4096_p1 skd_p1 seqkd_p1 trd_p1 sft_p1
      fastopd1024_p2 fastopd2048_p2 fastopd4096_p2 skd_p2 seqkd_p2 trd_p2 sft_p2"
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
while true; do
  left=0
  for r in $RUNS; do
    latest=$(cat "outputs/checkpoints/$r/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0)
    pend=$(squeue -u "$USER" -h -o "%i %j %T" | awk -v r="$r" '$2 ~ "^pb_"r"_s" && $3=="PENDING" {print $1}')
    runn=$(squeue -u "$USER" -h -o "%j %T" | awk -v r="$r" '$1 ~ "^pb_"r"_s" && $2=="RUNNING"' | wc -l)
    [ -n "$pend" ] && left=$((left+1))
    if [ "${latest:-0}" -ge 139 ] && [ "$runn" = 0 ] && [ -n "$pend" ]; then
      log "$r finished at step $latest -> cancelling queued segments: $(echo $pend | tr '\n' ' ')"
      scancel $pend
    fi
  done
  [ "$left" = 0 ] && { log "no queued segments left"; exit 0; }
  sleep 600
done
