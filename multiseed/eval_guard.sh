#!/bin/bash
# 评测守护:等训练产出目标 checkpoint -> 提交统一协议评测 -> 汇总 -> 逐项核对题数与采样数。
# 缺分片或分片失败就只补交缺的那几片(submit_unified.sh 会跳过已完成和在队列里的),最多 4 轮;
# 每一步写进 multiseed/status.tsv。
# 用法: eval_guard.sh RUN STEP TAG [TRY] [POLL]
set -uo pipefail
RUN=$1; STEP=$2; TAG=$3; TRY=${4:-0}; POLL=${5:-0}
R=/home/kzhao2/Relay-OPD; ST=$R/multiseed/status.tsv
CK=$R/outputs/checkpoints/$RUN/global_step_$STEP
log() { printf "%s\t%s\t%s\t%s\t%s\n" "$(date '+%F %T')" "$RUN" "$STEP" "$1" "$2" >> $ST; echo "[guard] $1 $2"; }
again() {  # again <sbatch 额外参数...>:用新的 TRY/POLL 再交一次自己
  sbatch --parsable -p cs --qos=cs -c 2 --mem=8G --time=0:30:00 --exclude=cs-1-1,cs-1-3,cs-1-4 --job-name=g_$TAG \
    --output=$R/logs/%x_%j.log "$@" --wrap="bash $R/multiseed/eval_guard.sh $RUN $STEP $TAG $NT $NP"
}

# 1) 训练还没产出目标 checkpoint:训练仍在跑就 30 分钟后再看;训练已经全部结束则记失败
if [ ! -f $CK/actor/huggingface/config.json ] && [ ! -f $CK/huggingface/config.json ]; then
  if squeue -h -u kzhao2 -o "%j" | grep -q "^ms_${TAG}_s"; then
    NT=$TRY; NP=$((POLL + 1))
    [ $NP -gt 300 ] && { log TRAIN_TIMEOUT "等了 150 小时仍无 checkpoint"; exit 1; }
    again --begin=now+30minutes >/dev/null; exit 0
  fi
  # 附加守护(只负责评测另一个 checkpoint)不重提交训练,交给主守护,避免重复训练
  [ "${GUARD_NO_RESUBMIT:-0}" = 1 ] && { log TRAIN_MISSING "附加守护:训练已结束但没有 global_step_$STEP,不重提交(由主守护负责)"; exit 1; }
  # 训练链被意外打断(例如节点在启动 2 秒内把任务 root 杀掉):清掉失效的锁,重新提交整条训练链,最多 3 次
  RS=$R/outputs/checkpoints/$RUN/.resubmits; k=$(cat $RS 2>/dev/null || echo 0)
  if [ $k -lt 6 ]; then
    mkdir -p $R/outputs/checkpoints/$RUN; echo $((k + 1)) > $RS
    for L in $R/outputs/checkpoints/$RUN/.seg*.lock; do
      [ -d "$L" ] || continue
      j=$(cut -d' ' -f1 "$L/job" 2>/dev/null); squeue -h -j "${j:-0}" 2>/dev/null | grep -q . || rm -rf "$L"
    done
    seed=${TAG: -2}; m=${TAG%$seed}; size=1.7B
    [[ $m == 6* ]] && { size=0.6B; m=${m#6}; }                    # 0.6B 的任务名前缀是 6
    (cd $R && SIZE=$size NSEG_ADD=2 ONLY="$m" SEEDS="$seed" bash multiseed/submit.sh) >/dev/null 2>&1
    log TRAIN_RESUBMIT "训练链中断且没有 global_step_$STEP,第 $((k + 1)) 次重新提交"; exit 0
  fi
  log TRAIN_FAILED "训练任务都已结束,但没有 global_step_$STEP(已重新提交 6 次)"; exit 1
fi
# SFT 的模型在 global_step_N/huggingface,评测脚本默认读 actor/huggingface
if [ ! -e $CK/actor/huggingface ] && [ -d $CK/huggingface ]; then
  mkdir -p $CK/actor && ln -s ../huggingface $CK/actor/huggingface
fi

# 2) 汇总已完成的分片,再逐项核对
bash $R/criticopd/agg_step.sh $RUN $STEP >/dev/null 2>&1
CHK=$(/home/kzhao2/.conda/envs/relay-opd/bin/python3 - "$R/outputs/eval_out/${RUN}_u32/step_$STEP" <<'PY'
import json, os, sys
d = sys.argv[1]
want = {"aime24": (30, 32), "aime25": (30, 32), "amc23": (83, 32), "math500": (500, 8), "olympiad": (675, 8), "minerva": (272, 8)}
bad, acc = [], {}
for b, (n, k) in want.items():
    f = f"{d}/{b}.summary.json"
    if not os.path.exists(f):
        bad.append(f"{b}:缺"); continue
    s = json.load(open(f))
    if s["n_problems"] != n or s["n_samples"] != k:
        bad.append(f"{b}:{s['n_problems']}x{s['n_samples']}"); os.remove(f)   # 残缺汇总删掉,下一轮重汇
        continue
    # 再数一遍合并后的逐条记录:必须正好是 题数 x 采样数(防止某个分片的生成文件不完整)
    jf = f"{d}/{b}.jsonl"
    rows = sum(1 for _ in open(jf)) if os.path.exists(jf) else -1
    if rows != n * k:
        bad.append(f"{b}:记录{rows}≠{n*k}"); os.remove(f)
        continue
    acc[b] = 100 * s["avg@k"]
if bad:
    print("BAD " + " ".join(bad))
else:
    v = list(acc.values())
    print("OK avg6=%.2f avg4=%.2f " % (sum(v) / 6, sum(acc[b] for b in ("aime24", "aime25", "amc23", "math500")) / 4)
          + " ".join(f"{b}={acc[b]:.2f}" for b in want))
PY
)
if [[ $CHK == OK* ]]; then log DONE "${CHK#OK }"; exit 0; fi
if [ $TRY -ge 20 ]; then log EVAL_FAILED "${CHK#BAD }(已重试 20 轮)"; exit 1; fi   # 集群会在启动 2 秒内 root 杀掉作业,多给几轮

# 3) 补交缺的分片
# 第二轮起推迟 10 分钟再开始:节点有时会在几分钟内把新放上去的作业全部 root 杀掉(2026-10-04 1:13-1:19
# 一台节点杀了 457 个),不等的话几分钟就把重试轮数耗光
SBX=""; [ $TRY -ge 1 ] && SBX="--begin=now+10minutes"
OUT=$(cd $R && SB_EXTRA="$SBX" RUN=$RUN STEP=$STEP TAG=$TAG bash criticopd/submit_unified.sh 2>&1)
J=$(echo "$OUT" | grep -oE '[0-9]{8}$' | xargs | tr ' ' ':')
if [ -n "$J" ]; then
  log EVAL_SUBMIT "第 $((TRY + 1)) 轮 $(echo $J | tr ':' ' ' | wc -w) 片;此前 ${CHK#BAD }"
  NT=$((TRY + 1)); NP=$POLL; again --dependency=afterany:$J >/dev/null
else  # 缺的分片都已在队列里:过 30 分钟再核对
  NT=$TRY; NP=$((POLL + 1)); again --begin=now+30minutes >/dev/null
fi
