#!/bin/bash
# 离线筛选结束后:出报告 -> 读 decision.json -> 按决定提交训练(到第 40 步)+ 评测 + 汇总。
# 用干净的登录环境提交,避免把本任务的 SLURM_* 环境变量带进新训练。
set -uo pipefail
R=/home/kzhao2/Relay-OPD; O=$R/criticopd/out_kerr
cd $R/criticopd
NSH=${NSH:-10}
n=$(ls $O/repair_*.jsonl 2>/dev/null | wc -l)
if [ ! -f $O/critic.jsonl ] || [ "$n" -ne "$NSH" ]; then
  echo "[decide] 离线筛选不完整:critic.jsonl $([ -f $O/critic.jsonl ] && echo 有 || echo 缺),重写分片 $n/$NSH —— 不训练" | tee -a $O/submitted.txt
  exit 1
fi
/home/kzhao2/.conda/envs/relay-opd/bin/python3 kerr_probe.py report 2>&1 | grep -v Warning | tee $O/report.txt
ARMS=$(/home/kzhao2/.conda/envs/relay-opd/bin/python3 -c "import json;print(' '.join(json.load(open('$O/decision.json'))['train']))")
[ -z "$ARMS" ] && { echo "[decide] 不训练(见 decision.json 的 reason)" | tee -a $O/submitted.txt; exit 0; }
for ARM in $ARMS; do
  env -i HOME=/home/kzhao2 USER=kzhao2 LOGNAME=kzhao2 SHELL=/bin/bash PATH=/usr/local/bin:/usr/bin:/bin bash -lc "
    cd $R
    T=\$(PART=cs,cs2,cs3 EXCLUDE=cs-1-2,dw-2-4,dw-1-3 CPUS=32 MEM=350G TIME=14:00:00 CHUNK_END=40 TEST_FREQ=-1 ARM=$ARM bash criticopd/submit_arm.sh 2>&1 | tail -1)
    [[ \"\$T\" =~ ^[0-9]+\$ ]] || { echo \"[decide] $ARM 训练提交失败: \$T\"; exit 1; }
    E=\$(RUN=criticopd_${ARM}_pt17b STEP=40 TAG=$ARM DEPEND=\$T bash criticopd/submit_unified.sh 2>&1 | grep -oE '[0-9]{8}\$' | xargs | tr ' ' ':')
    A=\$(sbatch --parsable -p cs --qos=cs -c 2 --mem=8G --time=0:30:00 --job-name=agg_${ARM}_40 --output=$R/logs/%x_%j.log --dependency=afterany:\$E --wrap='bash $R/criticopd/agg_step.sh criticopd_${ARM}_pt17b 40')
    echo \"[decide] $ARM: 训练 \$T -> 评测 \$(echo \$E | tr ':' ' ' | wc -w) 片 -> 汇总 \$A\"
  " 2>&1 | grep -v "^\s*$" | tee -a $O/submitted.txt
done
