#!/usr/bin/env python3
"""合并 refs/refs_*.jsonl -> refs_train40.json {sha1(题目文本): 参考解答};缺分片就失败退出。"""
import glob, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
nsh = int(sys.argv[1])
files = sorted(glob.glob(f"{HERE}/refs/refs_*.jsonl"))
if len(files) != nsh:
    sys.exit(f"!!! 只有 {len(files)}/{nsh} 个分片,不合并")
rows = [json.loads(l) for f in files for l in open(f)]
refs = {r["key"]: r["ref"] for r in rows if r["ref"]}
json.dump(refs, open(f"{HERE}/refs_train40.json", "w"))
print(f"[merge] {len(rows)} 道题,有参考解答 {len(refs)} = {100*len(refs)/len(rows):.1f}%;"
      f" 第 1 次就做对 {sum(r['n_try']==1 and r['n_ok']>0 for r in rows)}")
