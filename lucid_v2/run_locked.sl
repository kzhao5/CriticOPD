#!/bin/bash
# Mutual exclusion for duplicate / chained submissions of ONE Lucid run (same OUTPUT_DIR).
#   .siblings_<stage> : other submissions of the SAME stage (stage = CHUNK_END or "full"); a new owner cancels them
#   .backups          : non-preemptible backups; never cancelled, they exit on their own when not needed
# Taking over a lock from a job that is no longer running is atomic (mkdir takeover_from_<old>), so siblings
# released at the same moment by a dependency cannot both run. A requeued owner recognises its own id.
set -uo pipefail
# Ray 的 /tmp/ray 在多人共用节点上会撞(GCS cannot find the node + startup timeout)。
export RAY_TMPDIR="${RAY_TMPDIR_BASE:-/tmp/ray_$USER}_${SLURM_JOB_ID}"
mkdir -p "$RAY_TMPDIR"
OUT=${OUTPUT_DIR:?}; mkdir -p "$OUT"; LOCK=$OUT/.run_lock; STAGE=${CHUNK_END:-full}; newly=0
# Check the cards BEFORE touching the lock (2026-09-15: a copy landing on a node with a card held by a leftover
# process took the lock, cancelled its queued siblings, then failed the same check in run_method_freegpu.sl).
# 2026-10-01:交接时(刚 scancel 掉前一个 job、新 job 立刻接上同一批卡),前一个 job 的
# CUDA 上下文要几秒到几十秒才拆完。原来这里立刻检查、立刻 exit 3,于是把 4 张正在释放的卡
# 当成坏卡直接放掉,被别人拿走(13944877 在 cs-1-2 上 5 秒就退出)。
# 改为轮询等最多 GPU_FREE_WAIT 秒;残留进程永久占卡的情况照样会在超时后退出。
_assigned="${CUDA_VISIBLE_DEVICES:-}"
_deadline=$(( $(date +%s) + ${GPU_FREE_WAIT:-180} ))
while :; do
  _free=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2+0<2048{print $1}' | tr '\n' ' ')
  _nfree=0; for _g in ${_assigned//,/ }; do [[ " $_free " == *" $_g "* ]] && _nfree=$((_nfree+1)); done
  [ "$_nfree" -ge "${NEED_GPUS:-4}" ] && break
  [ "$(date +%s)" -ge "$_deadline" ] && break
  echo "[run_locked] $_nfree/${NEED_GPUS:-4} of assigned ($_assigned) free; waiting for previous job's GPU memory to release..."
  sleep 10
done
if [ "$_nfree" -lt "${NEED_GPUS:-4}" ]; then
  echo "[run_locked] only $_nfree of assigned ($_assigned) free on $(hostname); exiting without the lock"
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader; exit 3
fi
if mkdir "$LOCK" 2>/dev/null; then echo "$SLURM_JOB_ID" > "$LOCK/owner"; newly=1
elif [ ! -d "$LOCK" ]; then
  # mkdir failed although no lock exists: the output filesystem is not writable from this node (seen on dw-1-4,
  # 2026-09-17). Exiting 0 here used to look like "lock held" and silently skipped the run.
  echo "!!! cannot create $LOCK on $(hostname) (filesystem not writable?)"; exit 2
else
  owner=$(cat "$LOCK/owner" 2>/dev/null)
  if [ "$owner" != "$SLURM_JOB_ID" ]; then
    st=$(squeue -h -j "$owner" -o %t 2>/dev/null)
    if [ "$st" = R ] || [ "$st" = CG ]; then
      oq=$(squeue -h -j "$owner" -o %q 2>/dev/null); myq=$(squeue -h -j "$SLURM_JOB_ID" -o %q 2>/dev/null)
      if [[ "$oq" == *standby* ]] && [[ "$myq" != *standby* ]] && [ -n "${LUCID_SB_ARGS:-}" ]; then
        nj=$(sbatch --parsable $LUCID_SB_ARGS --begin=now+30minutes --export=ALL /home/kzhao2/Relay-OPD/lucid_v2/run_locked.sl)
        echo "$nj" >> "$OUT/.backups"; echo "owner $owner runs on standby; queued backup $nj (+30 min)"
      fi
      echo "lock held by running job $owner; exiting"; exit 0
    fi
    if ! mkdir "$LOCK/takeover_from_$owner" 2>/dev/null; then echo "another job is taking over from $owner; exiting"; exit 0; fi
    echo "taking over lock from $owner (state '${st:-gone}')"; echo "$SLURM_JOB_ID" > "$LOCK/owner"; newly=1
    [ -n "$st" ] && scancel "$owner" 2>/dev/null
  fi
fi
if [ "$newly" = 1 ]; then
  for s in $(cat "$OUT/.siblings_$STAGE" 2>/dev/null); do [ "$s" != "$SLURM_JOB_ID" ] && scancel "$s" 2>/dev/null && echo "cancelled stage-$STAGE sibling $s"; done
fi
if [ -n "${FORK_FROM:-}" ] && [ -n "${FORK_STEP:-}" ] && [ ! -e "$OUT/latest_checkpointed_iteration.txt" ]; then
  src=$FORK_FROM/global_step_$FORK_STEP
  # data.pt(dataloader 状态)在旧 checkpoint 里没有保留(当时 20T 配额满,只存了 actor/)。
  # 没有它,续训时 dataloader 从头开始 —— 会重看已见过的题。允许这种 fork,因为 CriticOPD
  # 的所有 arm(R0..R4)都用同一个起点和同一个重启行为,组间仍然可比;不可比的只是
  # 主表里的 opd_1p7b@80(它的数据顺序是连续的),所以 R0 必须在本设置下重跑。
  [ -f "$src/actor/huggingface/config.json" ] || { echo "!!! fork source $src incomplete"; exit 1; }
  [ -f "$src/data.pt" ] || echo "[fork] 注意:$src 没有 data.pt,dataloader 将从头开始"
  cp -r "$src" "$OUT/" && echo "$FORK_STEP" > "$OUT/latest_checkpointed_iteration.txt"
  echo "[fork] copied $src -> $OUT at job start (model + optimizer + dataloader state)"
fi
echo "[run_locked] job $SLURM_JOB_ID owns $OUT (stage $STAGE, latest ckpt $(cat $OUT/latest_checkpointed_iteration.txt 2>/dev/null || echo none))"
# A chained "insurance" stage whose target is already reached must exit cleanly. Without this it trains
# ONE extra update and writes an off-target checkpoint (this is how lucid_L1_p1_fork40 got a ckpt 101),
# which then triggers a redundant ~4h eval. Only affects jobs submitted after this edit: sbatch snapshots
# the script at submission time.
if [ -n "${CHUNK_END:-}" ] && [ -e "$OUT/latest_checkpointed_iteration.txt" ]; then
  _cur=$(cat "$OUT/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0)
  case "$_cur$CHUNK_END" in *[!0-9]*) : ;; *)
    if [ "$_cur" -ge "$CHUNK_END" ]; then
      echo "[run_locked] latest ckpt $_cur already >= CHUNK_END $CHUNK_END -> nothing to train, exiting cleanly"
      exit 0
    fi ;;
  esac
fi
exec bash /home/kzhao2/Relay-OPD/run_method_freegpu.sl
