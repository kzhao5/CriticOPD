#!/bin/bash
# checkpoint 一落地就用便宜协议(8 samples x 16384, math500)评一次,不用等 step 80。
# 正式的 PT 协议(32 x 32768, 6 benchmark)留给最终 step,由 submit_base_table.sh 走。
set -uo pipefail
cd /home/kzhao2/Relay-OPD
ARMS=${ARMS:-criticopd_R3c_pt17b criticopd_R2_pt17b}
DONE=criticopd/.auto_eval_done; touch $DONE
while :; do
  for RUN in $ARMS; do
    for D in outputs/checkpoints/$RUN/global_step_*/actor/huggingface; do
      [ -f "$D/config.json" ] || continue
      S=$(echo "$D" | sed -E 's|.*global_step_([0-9]+)/.*|\1|')
      grep -qx "$RUN@$S" $DONE && continue
      # 训练还在往这个目录写的时候别评:等 config.json 稳定 2 分钟
      [ $(( $(date +%s) - $(stat -c %Y "$D/config.json") )) -lt 120 ] && continue
      J=$(sbatch --parsable --partition=cs,dw,m13h --exclude=dw-2-4,dw-1-3 \
            --gres=gpu:1 --time=6:00:00 --job-name=ae_${RUN#criticopd_}_$S \
            --output=/home/kzhao2/Relay-OPD/logs/%x_%j.log \
            --export=ALL,RUN=$RUN,STEP=$S,BENCHES=math500,N_SAMPLES=8,MAX_NEW=16384,MAX_MODEL_LEN=18433,SHARD_BASE=0,NUM_SHARDS_TOTAL=1 \
            eval_shard.sl 2>&1 | tail -1)
      echo "$(date +%H:%M) 提交 $RUN@$S -> $J"; echo "$RUN@$S" >> $DONE
    done
  done
  sleep 300
done
