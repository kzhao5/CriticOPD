#!/bin/bash
# 训练已到目标步数、但备用分段还在排队的运行:取消这些备用段,让评测守护(依赖最后一段结束)马上开始评测
cd /home/kzhao2/Relay-OPD
# 每个运行只认最后登记的目标步数(0.6B CriticOPD 先登记 60、后改为 80;按 60 判断会把续训链当成备用段取消)
awk -F'\t' '$4 ~ /^train_seg/ {last[$2]=$3} END {for (r in last) print r, last[r]}' multiseed/jobs.tsv | while read run S; do
  CK=outputs/checkpoints/$run/global_step_$S
  [ -f $CK/actor/huggingface/config.json ] || [ -f $CK/huggingface/config.json ] || continue
  base=${run%_seed*}; seed=${run##*_seed}; m=${base%%_*}
  case $m in fastopd8192) m=fastopd;; criticopd) m=critic;; esac
  [[ $base == criticopd_prefix_* ]] && m=criticp
  [[ $base == *pt06b ]] && m=6$m
  tag=ms_${m}${seed}
  squeue -h -u kzhao2 -t R -o %j | grep -q "^${tag}_s" && continue          # 还有段在跑(比如正在存最后的 checkpoint)
  ids=$(squeue -h -u kzhao2 -t PD -o "%i %j" | awk -v t="${tag}_s" 'index($2,t)==1 {print $1}')
  [ -n "$ids" ] && { scancel $ids; echo "$run 已训到 $S,取消备用段: $(echo $ids | wc -w) 个"; }
done
