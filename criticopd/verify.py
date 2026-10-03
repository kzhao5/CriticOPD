#!/usr/bin/env python3
"""核对每个 summary 的实际协议与题数。提交参数可能被 math.sh 的分片复用逻辑悄悄忽略,
所以判定必须读产出的 summary,不能信提交时写的 --export。"""
import json, glob, sys
FULL={"math500":500,"olympiad":675,"minerva":272,"amc23":83,"aime24":30,"aime25":30}
want_s=lambda b: 32 if b in ("aime24","aime25") else 8
bad=[]
for f in sorted(glob.glob("outputs/eval_out/*/step_*/[a-z]*.summary.json")):
    try: d=json.load(open(f))
    except Exception: continue
    b=d.get("bench") or d.get("tag")
    if b not in FULL: continue
    run=f.split("/")[2]; step=f.split("/")[3].replace("step_","")
    issues=[]
    if d.get("n_problems")!=FULL[b]: issues.append(f"题数{d.get('n_problems')}≠{FULL[b]}")
    if d.get("n_samples")!=want_s(b): issues.append(f"samples{d.get('n_samples')}≠{want_s(b)}")
    if issues: bad.append((run,step,b,"; ".join(issues)))
if not bad: print("  全部 summary 符合统一协议")
for r,s,b,i in bad: print(f"  {r:<28} @{s:<4} {b:<9} {i}")
