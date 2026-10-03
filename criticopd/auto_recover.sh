#!/bin/bash
# 补聚合:分片(shard_0)已跑完、但 math.sh 自己的聚合失败而缺顶层 summary 的。
# 起因:17:12 改了 math.sh,bash 边执行边读脚本,正在跑的 job 回头读聚合段时字节偏移错位,
# 报 "syntax error near unexpected token `then'"。生成结果都在,只缺这一步。
# 只碰写入超过 2 分钟的分片,避免和正常 job 自己的聚合抢同一个文件。
cd /home/kzhao2/Relay-OPD
PY=/home/kzhao2/.conda/envs/relay-opd/bin/python3
while :; do
  for SH in outputs/eval_out/*_u32/step_*/shard_0/*.summary.json; do
    [ -f "$SH" ] || continue
    D=$(dirname "$(dirname "$SH")"); B=$(basename "$SH" .summary.json)
    [ -f "$D/$B.summary.json" ] && continue
    [ $(( $(date +%s) - $(stat -c %Y "$SH") )) -lt 120 ] && continue
    N=$(ls -d "$D"/shard_* | wc -l)
    RUN=$(echo "$D" | cut -d/ -f3); STEP=$(basename "$D" | sed 's/step_//')
    echo "$(date +%H:%M) 补聚合 $RUN@$STEP $B ($N 片)"
    $PY criticopd/agg_shards.py "$RUN" "$STEP" "$B" "$N" 2>&1 | tail -1 | sed 's/^/    /'
  done
  sleep 120
done
