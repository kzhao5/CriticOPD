#!/bin/bash
# 统一评测协议:8 samples x 32768;aime24/aime25/amc23 用 32 samples(30/83 题,8 samples 分辨率不够)。
#   预算是上限(遇 EOS 即停),32768 消除截断偏差;不同方法生成长度差异很大,用 16384 会系统性
#   惩罚长生成的方法。
# 慢的 benchmark 按题目拆成多个单卡 job 并行跑(每个 job 写 shard_<k>),最后由
# auto_recover2.sh 按 <bench>.nshards 聚合 —— 单卡跑完一个 olympiad 要约 2.5 小时。
# 用法: RUN=... STEP=... [TAG=...] [DEPEND=...] bash criticopd/submit_unified.sh
set -euo pipefail
cd /home/kzhao2/Relay-OPD
OUT_RUN=${OUT_RUN:-${RUN}_u32}
RUN=${RUN:?}; STEP=${STEP:?}; TAG=${TAG:-$(echo "$RUN" | sed 's/_pt17b//;s/criticopd_//')}
DEP=""; [ -n "${DEPEND:-}" ] && DEP="--dependency=afterany:${DEPEND}"
OD=outputs/eval_out/${OUT_RUN}/step_$STEP; mkdir -p "$OD"
QUEUED=$(squeue -u kzhao2 -h -o "%j")
#  bench:samples:max_new:max_model_len:分片数
for SPEC in olympiad:8:32768:34817:4 amc23:32:32768:34817:4 aime24:32:32768:34817:2 \
            aime25:32:32768:34817:2 math500:8:32768:34817:2 minerva:8:32768:34817:1; do
  IFS=: read -r B NS MN ML NSH <<< "$SPEC"
  [ -f "$OD/$B.summary.json" ] && { echo "  [skip] $TAG@$STEP $B 已有"; continue; }
  [ "$NSH" -gt 1 ] && echo "$NSH" > "$OD/$B.nshards"
  for K in $(seq 0 $((NSH-1))); do
    NAME=u_${TAG}_${STEP}_$B; [ "$NSH" -gt 1 ] && NAME=${NAME}_s$K
    echo "$QUEUED" | grep -qx "$NAME" && { echo "  [skip] $NAME 已在队列"; continue; }
    [ -f "$OD/shard_$K/$B.summary.json" ] && { echo "  [skip] $NAME 分片已完成"; continue; }
    # cs QOS 能抢占 gstandby/standby;cs-1-2 疑有残留进程占卡,先排除。
    J=$(sbatch --parsable $DEP --partition=cs,cs2,cs3 --qos=cs --exclude=cs-1-2,cs-1-3 --cpus-per-task=4 --mem=64G \
      --gres=gpu:1 --time=10:00:00 --job-name=$NAME \
      --output=/home/kzhao2/Relay-OPD/logs/%x_%j.log \
      --export=ALL,RUN=$RUN,OUT_RUN=${OUT_RUN},STEP=$STEP,BENCHES=$B,N_SAMPLES=$NS,MAX_NEW=$MN,MAX_MODEL_LEN=$ML,SHARD_BASE=$K,NUM_SHARDS_TOTAL=$NSH \
      eval_shard.sl)
    echo "  $NAME (${NS}x${MN}, 分片 $K/$NSH) -> $J"
  done
done
