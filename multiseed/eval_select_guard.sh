#!/bin/bash
# 种子 43/44 的选点评测(主表协议:在 seed 42 上按前四项均值选 checkpoint)。
# 等 seed 42 在每个候选步都评完(status.tsv 里有 DONE,或该步已判定失败),取 avg4 最高的一步,
# 再对本种子评测这一步(交给 eval_guard.sh,附加守护模式,不重提交训练)。
# 用法: eval_select_guard.sh BASE SEED TAG "STEPS" [POLL]
#   BASE 是不带 _seedN 的运行名,例如 opsdt_pt17b;STEPS 例如 "20 40 60 80"
set -uo pipefail
BASE=$1; SEED=$2; TAG=$3; STEPS=$4; POLL=${5:-0}
R=/home/kzhao2/Relay-OPD; ST=$R/multiseed/status.tsv
log() { printf "%s\t%s\t%s\t%s\t%s\n" "$(date '+%F %T')" "${BASE}_seed$SEED" "-" "$1" "$2" >> $ST; echo "[select] $1 $2"; }
again() {
  sbatch --parsable -p cs --qos=cs -c 2 --mem=8G --time=0:30:00 --exclude=cs-1-1,cs-1-3,cs-1-4 --begin=now+30minutes \
    --job-name=gs_$TAG --output=$R/logs/%x_%j.log \
    --wrap="bash $R/multiseed/eval_select_guard.sh $BASE $SEED $TAG '$STEPS' $((POLL + 1))" >/dev/null
}
SEL=$(python3 - "$ST" "${BASE}_seed42" $STEPS <<'PY'
import sys
st, run, steps = sys.argv[1], sys.argv[2], sys.argv[3:]
last = {}
for line in open(st):
    f = line.rstrip("\n").split("\t")
    if len(f) >= 5 and f[1] == run and f[2] in steps:
        last[f[2]] = (f[3], f[4])
done = {}
for s in steps:
    state, msg = last.get(s, ("", ""))
    if state == "DONE":
        done[s] = float(msg.split("avg4=")[1].split()[0])
    elif state != "EVAL_FAILED":          # TRAIN_MISSING 等不算定论(附加守护可能误判退出后被重交)
        print("WAIT"); sys.exit()
print(max(done, key=done.get) if done else "NONE")
PY
)
if [ "$SEL" = WAIT ]; then
  [ $POLL -gt 400 ] && { log SELECT_TIMEOUT "seed 42 在 200 小时内没有评完全部候选步"; exit 1; }
  again; exit 0
fi
if [ "$SEL" = NONE ]; then log SELECT_FAILED "seed 42 的候选步都没有评测结果"; exit 1; fi
log SELECTED "按 seed 42 的前四项均值选第 $SEL 步(候选 $STEPS)"
GUARD_NO_RESUBMIT=1 bash $R/multiseed/eval_guard.sh ${BASE}_seed$SEED $SEL $TAG 0 0
