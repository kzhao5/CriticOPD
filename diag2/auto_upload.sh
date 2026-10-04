#!/bin/bash
# 结果出来就自动出图、打包并推送到 GitHub。在登录节点后台运行(计算节点不能联网):
#   nohup setsid bash auto_upload.sh > logs/auto_upload.log 2>&1 &
# 阶段 1:主批次 E10 判分完成 -> 重出主批次的数字与图,打包 section2_main,推送。
# 阶段 2:扩充批次全部完成 -> 合并两批(out_all),重出数字与图,打包 section2(最终版),推送,退出。
cd /home/kzhao2/Relay-OPD/diag2
PY=/home/kzhao2/.conda/envs/relay-opd/bin/python3
SR="srun -p cs --qos=cs -c 8 --mem=64G --time=0:45:00 --account=als44"
log() { echo "$(date '+%F %T') $*"; }
run() {  # run <DIAG_OUT> <包名>:出数字与图,再打包(都在计算节点上跑)
  $SR bash -c "cd $PWD && DIAG_OUT=$PWD/$1 $PY report.py > $1/report_stdout.txt 2>&1 && DIAG_OUT=$PWD/$1 $PY package.py $2"
}
main_ready() { [ -f out/graded_e10.jsonl ]; }
ext_ready() {
  for f in graded_e7.jsonl judge.jsonl graded_e10.jsonl critic_single_full.jsonl judge_partial_0.jsonl judge_partial_1.jsonl judge_partial_2.jsonl; do
    [ -f out_b/$f ] || return 1; done
}
fails=0
for i in $(seq 1 200); do                                    # 每 10 分钟看一次,最多约 33 小时
  if [ ! -f out/.uploaded_main ] && main_ready; then
    log "阶段 1:主批次 E10 完成,出图打包"
    if run out section2_main && bash sync_github.sh "Section 2 diagnosis: main batch with E9/E10 (independent first error, error-anchored V)"; then
      touch out/.uploaded_main; log "阶段 1 已推送"
    else fails=$((fails + 1)); log "阶段 1 失败(第 $fails 次)"; fi
  fi
  if ext_ready; then
    log "阶段 2:扩充批次完成,合并两批"
    if $SR bash -c "cd $PWD && $PY merge_out.py out out_b" && run out_all section2 && \
       bash sync_github.sh "Section 2 diagnosis: final results, main and extension batches merged"; then
      log "阶段 2 已推送,结束"; exit 0
    else fails=$((fails + 1)); log "阶段 2 失败(第 $fails 次)"; fi
  fi
  [ $fails -ge 6 ] && { log "失败次数过多,停止"; exit 1; }
  sleep 600
done
log "超时退出"
