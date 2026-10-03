#!/bin/bash
# 补聚合(v2):顶层 summary 缺失、但各分片都已跑完的 benchmark。
# 分片数以提交时写下的 <bench>.nshards 为准(没有该文件 = 单分片),不再用「已存在的
# shard_* 目录数」—— 某个分片 job 还没启动时目录还不存在,会把残缺结果当成完整的去聚合。
# 只碰写入超过 2 分钟的分片,避免和 job 自己的收尾抢同一个文件。
cd /home/kzhao2/Relay-OPD
PY=/home/kzhao2/.conda/envs/relay-opd/bin/python3
while :; do
  for SH in outputs/eval_out/*_u32/step_*/shard_0/*.summary.json; do
    [ -f "$SH" ] || continue
    D=$(dirname "$(dirname "$SH")"); B=$(basename "$SH" .summary.json)
    [ -f "$D/$B.summary.json" ] && continue
    N=$(cat "$D/$B.nshards" 2>/dev/null || echo 1)
    ok=1
    for k in $(seq 0 $((N-1))); do
      F="$D/shard_$k/$B.summary.json"
      [ -f "$F" ] && [ $(( $(date +%s) - $(stat -c %Y "$F") )) -ge 120 ] || { ok=0; break; }
    done
    [ $ok -eq 1 ] || continue
    RUN=$(echo "$D" | cut -d/ -f3); STEP=$(basename "$D" | sed 's/step_//')
    echo "$(date +%H:%M) 聚合 $RUN@$STEP $B ($N 片)"
    $PY criticopd/agg_shards.py "$RUN" "$STEP" "$B" "$N" 2>&1 | tail -1 | sed 's/^/    /'
  done
  sleep 120
done
