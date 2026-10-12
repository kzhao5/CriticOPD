#!/bin/bash
# Run every stage on one GPU, in order. Usage: bash scripts/run_all.sh configs/olmo.yaml ["<stage line to resume from>"]
# Each stage reads/writes out_dir, so a failed run resumes from a stage, e.g.
#   bash scripts/run_all.sh configs/olmo.yaml "cont student e6 0 1"
set -euo pipefail
cd "$(dirname "$0")/.."
CFG=$1; START=${2:-rollouts 0 1}
PY="python -m criticopd_diag.diag --config $CFG"
STAGES=("rollouts 0 1" "select" "cont teacher e1 0 1" "cont student e1 0 1" "grade e1" "closing" "critic 0 1" "merge_critic"
        "judge 0 1" "cont student e6 0 1" "cont teacher e6 0 1" "grade e6" "cont teacher e7 0 1" "cont student e7 0 1" "grade e7"
        "plots" "pack")
printf '%s\n' "${STAGES[@]}" | grep -qxF "$START" || { echo "unknown stage line: '$START'; choose one of:"; printf '  %s\n' "${STAGES[@]}"; exit 1; }
go=0
for s in "${STAGES[@]}"; do
  [ "$s" = "$START" ] && go=1
  [ $go = 1 ] || continue
  echo "=== $s ($(date '+%F %T'))"
  $PY $s
done
