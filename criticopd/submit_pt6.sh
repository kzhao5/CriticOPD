#!/bin/bash
# 主表协议的六 benchmark 评测:32 samples x 32768(minerva 16 samples),
# 分片数按题量给,和主表当初的 4 分片保持一致,避免 [skip] 冲突。
# 用法: RUN=... STEP=... [DEPEND=jobid] bash criticopd/submit_pt6.sh
set -euo pipefail
cd /home/kzhao2/Relay-OPD
RUN=${RUN:?}; STEP=${STEP:?}
TAG=$(echo "$RUN" | sed 's/criticopd_//;s/_pt17b//')
DEP=""; [ -n "${DEPEND:-}" ] && DEP="--dependency=afterany:${DEPEND}"
D=outputs/eval_out/$RUN/step_$STEP; mkdir -p "$D"; : > "$D/.expect"
# bench:分片数:samples —— 与主表 OPD 行的 n_rows/n_problems 实测一致
for SPEC in math500:4:32 olympiad:4:32 amc23:2:32 minerva:2:16 aime24:1:32 aime25:1:32; do
  B=${SPEC%%:*}; R=${SPEC#*:}; NS=${R%%:*}; SM=${R##*:}
  echo "$B $NS" >> "$D/.expect"
  IDS=""
  for SH in $(seq 0 $((NS-1))); do
    J=$(sbatch --parsable $DEP --partition=cs,dw,m13h --exclude=dw-2-4,dw-1-3,m13h-1-1 \
      --gres=gpu:1 --time=12:00:00 --job-name=pt6_${TAG}_${STEP}_${B}_$SH \
      --output=/home/kzhao2/Relay-OPD/logs/%x_%j.log \
      --export=ALL,RUN=$RUN,STEP=$STEP,BENCHES=$B,N_SAMPLES=$SM,MAX_NEW=32768,MAX_MODEL_LEN=34817,SHARD_BASE=$SH,NUM_SHARDS_TOTAL=$NS \
      eval_shard.sl)
    IDS="$IDS $J"
  done
  echo "  $TAG@$STEP $B: $NS 片 x $SM samples ->$IDS"
done
