#!/bin/bash
# 三个分片齐了就合并一次。math.sh 在跨 job 分片时会跳过聚合,所以必须有人补这一刀。
set -uo pipefail
cd /home/kzhao2/Relay-OPD
DONE=criticopd/.auto_agg_done; touch $DONE
ARMS="criticopd_R1_pt17b criticopd_R3v2_pt17b criticopd_R4_pt17b criticopd_R2v2_pt17b"
while :; do
  for RUN in $ARMS; do
    for D in outputs/eval_out/$RUN/step_*; do
      [ -d "$D" ] || continue
      S=${D##*step_}
      grep -qx "$RUN@$S" $DONE && continue
      N=$(ls -d $D/shard_* 2>/dev/null | wc -l)
      [ "$N" -ge 3 ] || continue
      ok=1
      for i in 0 1 2; do [ -f "$D/shard_$i/math500.summary.json" ] || ok=0; done
      [ "$ok" = 1 ] || continue
      /home/kzhao2/.conda/envs/relay-opd/bin/python3 criticopd/agg_shards.py "$RUN" "$S" math500 3 \
        && echo "$RUN@$S" >> $DONE
    done
  done
  sleep 180
done
