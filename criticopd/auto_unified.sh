#!/bin/bash
# checkpoint 落地即用统一协议跑全部 6 个 math benchmark(8x32768;aime/amc23 用 32 samples)。
# 用完成标记防重复提交 —— submit_unified.sh 只认 summary,在跑的 job 它看不见。
set -uo pipefail
cd /home/kzhao2/Relay-OPD
ARMS=${ARMS:-"criticopd_R4GT_pt17b criticopd_R4TR_pt17b"}
STEPS=${STEPS:-"40 60 80"}
DONE=criticopd/.auto_unified_done; touch $DONE
while :; do
  for RUN in $ARMS; do
    for S in $STEPS; do
      D=outputs/checkpoints/$RUN/global_step_$S/actor/huggingface
      [ -f "$D/model.safetensors" ] || continue
      grep -qx "$RUN@$S" $DONE && continue
      # 训练还在往这里写时别评:等权重文件稳定 3 分钟
      [ $(( $(date +%s) - $(stat -c %Y "$D/model.safetensors") )) -lt 180 ] && continue
      TAG=$(echo $RUN | sed 's/criticopd_//;s/_pt17b//')
      echo "$(date +%H:%M) 提交 $RUN@$S 全套"
      RUN=$RUN STEP=$S TAG=$TAG bash criticopd/submit_unified.sh 2>&1 | sed 's/^/    /'
      echo "$RUN@$S" >> $DONE
    done
  done
  sleep 300
done
