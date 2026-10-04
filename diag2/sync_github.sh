#!/bin/bash
# 把 Section 2 的代码(diag2/)与打包好的结果(results_final/section2_main、section2)同步到 GitHub 仓库并推送。
# 在登录节点运行(计算节点不能联网)。用法: bash sync_github.sh "<提交说明>"
set -euo pipefail
REPO=/home/kzhao2/CriticOPD-release; SRC=/home/kzhao2/Relay-OPD
mkdir -p $REPO/diag2 $REPO/results_final
for f in diag.py report.py merge_out.py analyze_loc.py package.py run_gpu.sl pipeline.sh pipeline_ext.sh pipeline_ext2.sh pipeline_e10.sh smoke.sl sync_github.sh auto_upload.sh; do
  [ -f $SRC/diag2/$f ] && cp $SRC/diag2/$f $REPO/diag2/
done
paths="diag2 README.md"
for d in section2_main section2; do
  [ -d $SRC/results_final/$d ] && { rsync -a --delete $SRC/results_final/$d/ $REPO/results_final/$d/; paths="$paths results_final/$d"; }
done
cd $REPO
# 推送前确认不含凭据
if grep -rIlE "olp_[A-Za-z0-9]{10,}|ghp_[A-Za-z0-9]{10,}|github_pat_[A-Za-z0-9_]{20,}|BEGIN (RSA|OPENSSH) PRIVATE" $paths 2>/dev/null; then echo "发现疑似凭据,停止推送"; exit 1; fi
git add $paths
git diff --cached --quiet && { echo "没有变化,不提交"; exit 0; }
git commit -q -m "$1"
GIT_SSH_COMMAND="ssh -o BatchMode=yes -o ConnectTimeout=30" git push -q git@github.com:kzhao5/CriticOPD.git main
echo "已推送 $(git rev-parse --short HEAD): $1"
