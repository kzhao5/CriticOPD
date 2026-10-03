#!/bin/bash
#SBATCH --account=als44
#SBATCH --partition=m8
#SBATCH --qos=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=3-00:00:00
#SBATCH --job-name=pb_eval_campaign
#SBATCH --output=/nobackup/autodelete/usr/kzhao2/relay-opd/slurm_logs/%x_%j.log
#
# Rate-limited math evaluation of every checkpoint of the paper baselines and the warm-up ablation arms.
# Why: on 2026-09-16 02:52 the eval chasers submitted ~50 ev16 jobs at once; all of them were killed 2 s after
# starting ("CANCELLED by 0") because the account's AssocGrpBillingRunMinutes budget counts GPUs x REQUESTED time
# over all running jobs. Submitting a few at a time stays inside the budget and retries what gets killed.
#
# Every POLL seconds: submit missing (run, step) cells while fewer than MAXQ ev16 jobs are queued/running.
# A cell is missing when its four math benchmarks are not all finished; at most TRIES attempts per cell.
# Steps that are byte copies of a fork parent are skipped (the warm-up arms' step 40 IS fastopd_p1@40).
set -uo pipefail
cd /home/kzhao2/Relay-OPD
PY=/home/kzhao2/.conda/envs/relay-opd/bin/python3
SQ=/apps/slurm/latest/bin/squeue   # not the ORC squeue wrapper: it caches into $HOME and fails when home is full
POLL=${POLL:-600}; MAXQ=${MAXQ:-4}; TRIES=${TRIES:-8}; NICE=${NICE:-200}
mkdir -p .jobids lucid_warmup_ablation/.eval_attempts
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
log "start maxq=$MAXQ poll=$POLL nice=$NICE"
while true; do
  MISSING=$($PY - <<'PY'
import glob, os, sys
os.chdir("/home/kzhao2/Relay-OPD"); sys.path.insert(0, "lucid_v4")
import full_comparison as fc
RUNS = ["fastopd1024_p1","fastopd2048_p1","fastopd4096_p1","sft_p1","seqkd_p1","trd_p1","skd_p1",
        "fastopd1024_p2","fastopd2048_p2","fastopd4096_p2","sft_p2","seqkd_p2","trd_p2","skd_p2",
        # post-trained 1.7B baselines (submit_pt17b.sh). Their per-run chasers submitted seven eval jobs
        # at once on 09-19 and four were killed 2 s in by the billing cap, so the campaign owns them now.
        # The four PT runs that already existed were evaluated with the paper's protocol (32 samples x
        # 32768). Re-evaluating them at b16k makes the whole post-trained column one protocol; listed
        # before the new baselines so the mismatch is cleared first.
        "opd_1p7b","relay_1p7b","fastopd8192_1p7b","gatedB_1p7b",
        "fastopd1024_pt17b","fastopd2048_pt17b","fastopd4096_pt17b","sft_pt17b","trd_pt17b",
        "seqkd_pt17b","skd_pt17b","grpo_pt17b",
        # our own runs in the same column: they need the same protocol to be comparable, but they are
        # last so the baselines drain first
        "lucid_L1_pt17b","lucid_L1stop_v6_pt17b","lucid_L1lex_v6_pt17b",
        "lucid_warm_A_fullLucid_fastopd40","lucid_warm_B_eosOnly_fastopd40",
        "lucid_warm_C_rbNoEos_fastopd40","lucid_warm_R_control_fastopd40"]
SKIP = {r: {40} for r in RUNS if r.startswith("lucid_warm_")}   # step 40 = the copied fastopd_p1@40 parent
for r in RUNS:
    for p in sorted(glob.glob(f"outputs/checkpoints/{r}/global_step_*")):
        s = int(p.rsplit("_", 1)[1])
        if s not in (20, 40, 60, 80, 100, 120, 139) or s in SKIP.get(r, set()):
            continue
        if not (os.path.exists(f"{p}/actor/huggingface/config.json") or os.path.exists(f"{p}/huggingface/config.json")):
            continue
        if not all(fc.complete(r, s, b) for b in fc.BENCHES):
            print(r, s)
PY
)
  [ -z "$MISSING" ] && { log "no missing cells left"; exit 0; }
  n_left=$(wc -l <<< "$MISSING")
  sub=0
  while read -r RUN STEP; do
    [ -z "$RUN" ] && continue
    # squeue lags a second behind sbatch, so count what this pass already submitted as well
    nq=$($SQ -u "$USER" -h -o "%j" | grep -c '^ev16_' || true)
    [ $((nq + sub)) -ge "$MAXQ" ] && break
    name=ev16_${RUN}_${STEP}
    Q=$($SQ -u "$USER" -h -o "%j" || true)
    grep -qx "$name" <<< "$Q" && continue
    att=lucid_warmup_ablation/.eval_attempts/$name
    t=$(cat "$att" 2>/dev/null || echo 0); [[ "$t" =~ ^[0-9]+$ ]] || t=0
    [ "$t" -ge "$TRIES" ] && continue
    # SFT checkpoints are written by verl's sft_trainer as global_step_N/huggingface; link the actor path the
    # shared eval scripts expect (no copy, nothing overwritten).
    d=outputs/checkpoints/$RUN/global_step_$STEP
    [ -f "$d/huggingface/config.json" ] && [ ! -e "$d/actor/huggingface" ] && mkdir -p "$d/actor" && ln -s ../huggingface "$d/actor/huggingface"
    # Alternate between the A100/B200 partitions and the H200 node set: m13h needs its own QOS, so the
    # two cannot be listed in one --partition. H200 rollouts run roughly twice as fast, hence the
    # shorter wall clock, which also costs less of the account's billing-minutes budget.
    if [ $((sub % 2)) -eq 0 ] && [ "${USE_H200:-1}" = 1 ]; then
      WHERE="--partition=m13h --qos=gpu --time=8:00:00"
    else
      # 8 h was not enough for the slowest cells: fastopd2048_p2 @40 and @60 hit the limit twice in a
      # row on MATH500 alone (aime/amc were already on disk, so each attempt restarted MATH500 from
      # scratch and never finished). The cs QOS allows a full day.
      WHERE="--partition=cs3,cs2,cs --qos=cs --time=14:00:00"
    fi
    J=$(RUN=$RUN STEPS=$STEP DP=2 NUM_SHARDS_TOTAL=1 N_SAMPLES=8 MAX_NEW=16384 MAX_MODEL_LEN=18433 \
        EVAL_TAG=_b16k EVAL_STOP_TOKEN_IDS='151643;151645' \
        sbatch --parsable $WHERE --gres=gpu:2 --cpus-per-task=16 --mem=200G --nice=$NICE \
          --job-name=$name --output=/nobackup/autodelete/usr/kzhao2/relay-opd/slurm_logs/%x_%j.log --export=ALL eval_base.sl)
    if [ -n "$J" ]; then
      echo $((t+1)) > "$att"; echo "$name $J" >> .jobids/evbase.txt; sub=$((sub+1)); log "submit $name -> $J (attempt $((t+1))) $WHERE"
    fi
  done <<< "$MISSING"
  log "cells left: $n_left, ev16 in queue: $($SQ -u "$USER" -h -o '%j' | grep -c '^ev16_' || true)"
  sleep "$POLL"
done
