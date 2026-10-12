#!/bin/bash
# 新基线 OPSD-T 与 Critique-GRPO:两个学生(1.7B、0.6B)× 三个种子,训练与评测全自动。
#  - seed 42:主守护评测最后一步,附加守护(GUARD_NO_RESUBMIT)评测其余候选步
#  - seed 43/44:主守护只守训练(GUARD_TRAIN_ONLY),选点守护等 seed 42 评完全部候选步后
#    按前四项均值选出一步再评测(主表协议)
#  - KEEP_CKPT=6:候选步的 checkpoint 在选点之前不能被删
set -euo pipefail
cd /home/kzhao2/Relay-OPD
R=$PWD; G="-p cs --qos=cs -c 2 --mem=8G --time=0:30:00 --exclude=cs-1-1,cs-1-3,cs-1-4"
for SZ in ${SIZES:-1.7B 0.6B}; do
  TP=""; SFX=pt17b; [ "$SZ" = 0.6B ] && { TP=6; SFX=pt06b; }
  for M in ${METHODS:-opsdt cgrpo}; do
    if [ $M = opsdt ]; then STEPS="20 40 60 80"; FINAL=80; else STEPS="40 80 120"; FINAL=120; fi
    BASE=${M}_$SFX
    SIZE=$SZ KEEP_CKPT=6 NSEG_ADD=1 SEEDS=42 ONLY=$M bash multiseed/submit.sh
    for ST in $STEPS; do
      [ $ST = $FINAL ] && continue
      j=$(sbatch --parsable $G --job-name=g_${TP}${M}42_$ST --output=$R/logs/%x_%j.log --export=ALL,GUARD_NO_RESUBMIT=1 \
            --wrap="bash $R/multiseed/eval_guard.sh ${BASE}_seed42 $ST ${TP}${M}42 0 0")
      printf "%s\t%s\t%s\tguard_extra\t%s\n" "$(date '+%F %T')" ${BASE}_seed42 $ST $j >> multiseed/jobs.tsv
      echo "${BASE}_seed42 附加守护 第 $ST 步 -> $j"
    done
    GUARD_TRAIN_ONLY=1 SIZE=$SZ KEEP_CKPT=6 NSEG_ADD=1 SEEDS="43 44" ONLY=$M bash multiseed/submit.sh
    for S in 43 44; do
      j=$(sbatch --parsable $G --job-name=gs_${TP}${M}$S --output=$R/logs/%x_%j.log \
            --wrap="bash $R/multiseed/eval_select_guard.sh $BASE $S ${TP}${M}$S '$STEPS' 0")
      printf "%s\t%s\t%s\tguard_select\t%s\n" "$(date '+%F %T')" ${BASE}_seed$S "$STEPS" $j >> multiseed/jobs.tsv
      echo "${BASE}_seed$S 选点守护 -> $j"
    done
  done
done
