#!/usr/bin/env python3
"""合并旧的 refs/ 与新的 refs_v2/,输出 refs_train40_v2.json;并报告真实前 40 步题目的覆盖率。缺分片就失败退出。"""
import glob, json, os, sys
H = os.path.dirname(os.path.abspath(__file__)); nsh = int(sys.argv[1])
new = sorted(glob.glob(f"{H}/refs_v2/refs_*.jsonl"))
if len(new) != nsh: sys.exit(f"!!! refs_v2 只有 {len(new)}/{nsh} 个分片")
rows = [json.loads(l) for f in sorted(glob.glob(f"{H}/refs/refs_*.jsonl")) + new for l in open(f)]
refs = {r["key"]: r["ref"] for r in rows if r["ref"]}
json.dump(refs, open(f"{H}/refs_train40_v2.json", "w"))
keys = list(dict.fromkeys(json.load(open(f"{H}/train40_keys_nw8.json"))))
print(f"[merge v2] 参考解答 {len(refs)} 条;真实前 40 步 {len(keys)} 道题中有参考 {sum(k in refs for k in keys)} = "
      f"{100*sum(k in refs for k in keys)/len(keys):.1f}%")
