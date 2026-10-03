#!/bin/bash
# 用法: stop_at.sh <jobid> <run> <step>
# 等 <step> 的 checkpoint 完整写出(latest 指针=step、HF 权重存在且 3 分钟未变)后取消训练 job。
# 学习率是常数,所以「训到 step 再停」和「只训 step 步」等价。
ID=$1; RUN=$2; STEP=$3
D=/home/kzhao2/Relay-OPD/outputs/checkpoints/$RUN
until [ "$(cat $D/latest_checkpointed_iteration.txt 2>/dev/null)" = "$STEP" ] \
   && [ -f $D/global_step_$STEP/actor/huggingface/model.safetensors ] \
   && [ $(( $(date +%s) - $(stat -c %Y $D/global_step_$STEP/actor/huggingface/model.safetensors) )) -ge 180 ]; do
  squeue -j $ID -h -o %t 2>/dev/null | grep -q . || { echo "$(date +%H:%M) $RUN job $ID 已不在队列,不必停"; exit 0; }
  sleep 60
done
scancel $ID && echo "$(date +%H:%M) $RUN 的 step $STEP checkpoint 已完整,取消训练 job $ID"
