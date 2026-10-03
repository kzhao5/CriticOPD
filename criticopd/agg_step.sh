#!/bin/bash
# 评测分片跑完后,把 <bench>.nshards 标记的多分片结果合并成顶层 summary。
RUN=$1; STEP=$2; D=/home/kzhao2/Relay-OPD/outputs/eval_out/${RUN}_u32/step_$STEP
for N in $D/*.nshards; do
  [ -f "$N" ] || continue; B=$(basename $N .nshards)
  [ -f $D/$B.summary.json ] && continue
  /home/kzhao2/.conda/envs/relay-opd/bin/python3 /home/kzhao2/Relay-OPD/criticopd/agg_shards.py ${RUN}_u32 $STEP $B $(cat $N)
done
ls $D/*.summary.json
