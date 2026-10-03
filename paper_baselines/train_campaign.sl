#!/bin/bash
#SBATCH --account=als44
#SBATCH --partition=m8
#SBATCH --qos=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=3-00:00:00
#SBATCH --job-name=pb_train_campaign
#SBATCH --output=/nobackup/autodelete/usr/kzhao2/relay-opd/slurm_logs/%x_%j.log
#
# Keeps the unfinished baselines of both settings moving, one segment at a time.
#
# Why not a dependency chain: on 09-19 the account was over its billing cap when a five-segment chain was
# submitted. Slurm started segment 1, the site policy cancelled it 2 s in ("CANCELLED by 0"), afterany
# treated that as finished, segment 2 started and was cancelled, and the whole chain died in 14 seconds.
# This campaign submits ONE segment per run, only while the account is under the cap, and only when that
# run has nothing queued or running.
#
# Budget: the site counts sum(billing x REQUESTED minutes) over running jobs, not remaining minutes, so a
# job that starts while the account is over the cap is killed on the spot. BUDGET_MAX keeps a margin.
set -uo pipefail
cd /home/kzhao2/Relay-OPD
SQ=/apps/slurm/latest/bin/squeue     # not the ORC wrapper: it caches into $HOME and fails when home is full
POLL=${POLL:-600}; MAXTRAIN=${MAXTRAIN:-3}; BUDGET_MAX=${BUDGET_MAX:-75}; CAP=12441600
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

budget_pct() {
  $SQ -h -u "$USER" -t RUNNING -O "TimeLimit:12,tres-alloc:100" 2>/dev/null | awk -v cap=$CAP '
    { tl=$1; if (tl ~ /-/) { split(tl,d,"-"); split(d[2],t,":"); m=(d[1]*24+t[1])*60+t[2] }
      else { split(tl,a,":"); m=a[1]*60+a[2] }
      b=0; if (match($0,/billing=[0-9]+/)) b=substr($0,RSTART+8,RLENGTH-8); tot+=b*m }
    END { printf "%d", 100*tot/cap }'
}

# run -> student, method script, train data, extra env. SFT is deliberately absent: verl's sft_trainer runs
# with resume_mode=disable, so a resubmission would silently restart it from step 0.
P1=/home/kzhao2/OPD-yyx/model/Qwen3-1.7B-Base
P2=/home/kzhao2/OPD/model/Qwen3-0.6B-Base
PT=/home/kzhao2/OPD/model/Qwen3-1.7B
DAPO=/home/kzhao2/OPD/datasets/dapo-math-17k.parquet
DATA=$PWD/outputs/offline_data
# 2026-09-21: the post-trained arms are reported at a uniform step 120, so skd_pt17b (already
# there) and grpo_pt17b (stopped by stop_at_step.sl) are no longer driven toward 139.
RUNS=${RUNS:-"skd_p2 opd_p2 grpo_p1 seqkd_pt17b trd_pt17b fastopd4096_pt17b"}

setup_env() {   # setup_env <run> -> exports everything run_method_freegpu.sl needs, or returns 1
  local r=$1
  export TEACHER_MODEL=/home/kzhao2/OPD/model/Qwen3-4B-Instruct-2507
  export BENCH=$PWD/outputs/data/bench MAX_PROMPT_LENGTH=2048 MAX_RESPONSE_LENGTH=16384
  export SAVE_FREQ=10 TOTAL_EPOCHS=1 VAL_BEFORE_TRAIN=False
  export VERL_OPD_STOP_TOKEN_IDS='151643;151645' STOP_TOKEN_IDS='151643;151645' SEMANTIC_EOS=0
  export ACTOR_GPUS_PER_NODE=2 TEACHER_GPUS_PER_NODE=2 NEED_GPUS=4
  export OUTPUT_DIR=$PWD/outputs/checkpoints/$r EXP_ID=$r TRAIN_DATA=$DAPO
  unset NUM_GPUS N_GPUS EXTRA_ARGS
  case $r in
    *_pt17b) export STUDENT_MODEL=$PT VAL_MAX_RESPONSE_LENGTH=32768 TEST_FREQ=-1 ;;
    *_p1)    export STUDENT_MODEL=$P1 VAL_MAX_RESPONSE_LENGTH=16384 TEST_FREQ=1000 ;;
    *_p2)    export STUDENT_MODEL=$P2 VAL_MAX_RESPONSE_LENGTH=16384 TEST_FREQ=1000 ;;
    *) return 1 ;;
  esac
  case $r in
    skd_*)         export METHOD_SCRIPT=opd/scripts/baselines/skd.sh ;;
    opd_*)         export METHOD_SCRIPT=opd/scripts/baselines/opd.sh ;;
    grpo_*)        export METHOD_SCRIPT=opd/scripts/baselines/grpo.sh N_GPUS=4 ;;
    seqkd_pt17b)   export METHOD_SCRIPT=opd/scripts/baselines/seqkd.sh TRAIN_DATA=$DATA/teacher_traj_pt17b/teacher_trajectories.parquet ;;
    trd_pt17b)     export METHOD_SCRIPT=opd/scripts/baselines/trd.sh   TRAIN_DATA=$DATA/trd_pt17b/trd_trajectories.parquet ;;
    fastopd1024_*) export METHOD_SCRIPT=opd/scripts/baselines/fastopd/1024.sh ;;
    fastopd2048_*) export METHOD_SCRIPT=opd/scripts/baselines/fastopd/2048.sh ;;
    fastopd4096_*) export METHOD_SCRIPT=opd/scripts/baselines/fastopd/4096.sh ;;
    *) return 1 ;;
  esac
  [ -s "$TRAIN_DATA" ] || { log "  $r: training data missing ($TRAIN_DATA)"; return 1; }
  return 0
}

log "start maxtrain=$MAXTRAIN budget_max=${BUDGET_MAX}% poll=$POLL runs: $RUNS"
while true; do
  left=0
  # count only RUNNING segments: a segment pending on a dependency costs nothing, and counting those
  # kept the campaign permanently above MAXTRAIN
  nq=$($SQ -h -u "$USER" -t RUNNING -o "%j" 2>/dev/null | grep -c '^pb_.*_s[0-9]' || true)
  pct=$(budget_pct)
  for r in $RUNS; do
    latest=$(cat "outputs/checkpoints/$r/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0)
    [ "${latest:-0}" -ge 139 ] && continue
    left=$((left+1))
    Q=$($SQ -h -u "$USER" -o "%j" 2>/dev/null || true)
    # match ANY job of this run, not just pb_<run>_s<N>: the manually submitted tails are pb_<run>_tail,
    # and two jobs training the same OUTPUT_DIR at once would corrupt its checkpoints
    grep -qE "^pb_${r}(_s[0-9]+|_tail)$" <<< "$Q" && continue
    [ "$nq" -ge "$MAXTRAIN" ] && continue
    [ "$pct" -ge "$BUDGET_MAX" ] && continue
    setup_env "$r" || continue
    # segment numbers only ever go up: train_segment.sl takes a per-segment lock directory and a stale lock
    # from a crashed attempt would make a reused number exit immediately.
    seg=$(( $(ls -d "$OUTPUT_DIR"/.seg*.lock 2>/dev/null | sed 's/.*\.seg//; s/\.lock//' | sort -n | tail -1 || echo 0) + 1 ))
    [ -z "$seg" ] && seg=1
    J=$(sbatch --parsable --job-name=pb_${r}_s$seg --partition=dw --qos=dw87 --exclude=dw-1-5,dw-2-4 \
          --gres=gpu:4 --cpus-per-task=32 --mem=350G --time=20:00:00 \
          --export=ALL,SEG=$seg,TARGET_STEP=139 \
          --output=/nobackup/autodelete/usr/kzhao2/relay-opd/slurm_logs/%x_%j.log \
          paper_baselines/train_segment.sl 2>/dev/null)
    if [ -n "$J" ]; then
      log "submit pb_${r}_s$seg -> $J (from step $latest, budget ${pct}%)"
      echo "pb_${r}_s$seg $J" >> .jobids/paper_baselines.txt
      nq=$((nq+1)); pct=$((pct+10))     # assume the new job takes roughly a tenth of the cap
    fi
  done
  log "runs left: $left, training jobs queued/running: $nq, budget ${pct}%"
  sleep "$POLL"
done
