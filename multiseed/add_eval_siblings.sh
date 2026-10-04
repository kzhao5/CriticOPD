#!/bin/bash
# 给每个还在排队、且没有副本在运行的评测分片补齐四组分区的副本(同名;eval_shard.sl 有锁,先开始的取消其余)
cd /home/kzhao2/Relay-OPD
declare -A G=(
  [cs]="--partition=cs,cs2,cs3 --qos=cs --exclude=cs-1-2,cs-1-3 --time=6:00:00"
  [dw]="--partition=dw --qos=dw87 --exclude=dw-2-4,dw-1-4,dw-1-2 --time=6:00:00"
  [m13h]="--partition=m13h --qos=gpu --exclude=m13h-1-1 --time=6:00:00"
  [m13l]="--partition=m13l --qos=gpu --time=10:00:00" )
grp() { case $1 in *cs*) echo cs;; dw) echo dw;; m13h|m13h,m13l) echo m13h;; m13l) echo m13l;; esac; }
n=0
for name in $(squeue -u kzhao2 -h -t PD -o "%j" | grep '^u_' | sort -u); do
  squeue -u kzhao2 -h -t R -n $name | grep -q . && continue                 # 已有副本在跑
  have=" $(squeue -u kzhao2 -h -t PD -n $name -o '%P' | while read p; do grp $p; done | tr '\n' ' ') "
  id=$(squeue -u kzhao2 -h -t PD -n $name -o %i | head -1)
  exp=$(sacct -n -X -j $id -o SubmitLine%2000 | grep -oE -- '--export=[^ ]+' | head -1)
  [ -z "$exp" ] && continue
  for g in cs dw m13h m13l; do
    [[ $have == *" $g "* ]] && continue
    sbatch --parsable ${G[$g]} --cpus-per-task=4 --mem=64G --gres=gpu:1 --job-name=$name \
      --output=/home/kzhao2/Relay-OPD/logs/%x_%j.log "$exp" eval_shard.sl >/dev/null && n=$((n+1))
  done
done
echo "补交评测副本: $n 个"
