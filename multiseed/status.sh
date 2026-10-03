#!/bin/bash
# 多 seed 实验总览:每个训练的当前步数、所在节点、报错数;评测守护的最新状态
cd /home/kzhao2/Relay-OPD
printf "%-26s %-8s %-10s %-8s %s\n" 训练 步数 节点 报错 最新状态
for d in grpo_pt17b skd_pt17b seqkd_pt17b opd_1p7b trd_pt17b fastopd8192_1p7b relay_1p7b sft_pt17b; do for s in 43 44; do
  run=${d}_seed$s; L=outputs/checkpoints/$run/training.log
  st=$(grep -oE '(^| )step:[0-9]+ ' $L 2>/dev/null | tail -1 | tr -d ' ')
  [ -z "$st" ] && st=$(grep -oE '[0-9]+/139 \[' $L 2>/dev/null | tail -1 | cut -d/ -f1 | sed 's/^/step:/')
  m=$(echo $d | sed 's/_pt17b//;s/_1p7b//;s/8192//'); job=$(squeue -u kzhao2 -h -t R -o "%j %R" | awk -v t="ms_$m${s}_s" 'index($1,t)==1 {print $2}' | head -1)
  err=$(grep -c "Error executing" $L 2>/dev/null)
  last=$(grep -P "\t$run\t" multiseed/status.tsv 2>/dev/null | tail -1 | cut -f4-5 | cut -c1-60)
  printf "%-26s %-8s %-10s %-8s %s\n" $run "${st:-—}" "${job:-排队}" "${err:-0}" "$last"
done; done
