#!/usr/bin/env python3
"""把跨 job 的 shard_*/ 合并成一个 summary 并追加到 results.tsv。

math.sh 的内嵌聚合器在 NUM_SHARDS_TOTAL != DP_SIZE 时会直接 exit 0 跳过,
而且它循环的是 range(dp_size);我们是 3 个独立 job 各跑 1 个 shard,
所以没有任何现成代码会去合并 shard_0/1/2。
分片切法是 idx % num_shards == shard_id,各片题目不重叠,按题数加权即可。
"""
import json, sys
from pathlib import Path

run, step, bench, nshards = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
root = Path("/home/kzhao2/Relay-OPD/outputs/eval_out")
d = root / run / f"step_{step}"
res = root / "results.tsv"

parts = []
for i in range(nshards):
    p = d / f"shard_{i}" / f"{bench}.summary.json"
    if not p.exists():
        sys.exit(f"!!! 缺分片 {p} —— 不聚合,避免写出残缺数字")
    parts.append(json.loads(p.read_text()))

n = sum(x["n_problems"] for x in parts)
if n == 0:
    sys.exit("!!! 题数为 0")
w = lambda k: sum(x[k] * x["n_problems"] for x in parts) / n
out = {"tag": bench, "model": parts[0]["model"], "bench": bench, "n_problems": n,
       "n_samples": parts[0]["n_samples"], "avg@k": round(w("avg@k"), 6),
       "pass@1": round(w("pass@1"), 6), "pass@k": round(w("pass@k"), 6),
       "wall_seconds": max(x["wall_seconds"] for x in parts), "n_shards": nshards}
(d / f"{bench}.summary.json").write_text(json.dumps(out, indent=2))

# 合并逐题明细,便于后续复核。
# 注意:jsonl 里的 problem_idx 是**分片内**的局部下标(每片都从 0 开始),不是全局题号。
# 跨分片做配对比较时必须用 (shard_id, problem_idx) 组合键,只用 problem_idx 会把
# 不同的题错配到一起,并把题数低估为 1/n_shards。
with (d / f"{bench}.jsonl").open("w") as f:
    for i in range(nshards):
        jp = d / f"shard_{i}" / f"{bench}.jsonl"
        if jp.exists():
            f.write(jp.read_text())

if not res.exists():
    res.write_text("run\tstep\tbench\tavg@32\tpass@1\tpass@32\tn_problems\twall_seconds\tmax_new\tmax_model_len\tmodel\n")
with res.open("a") as f:
    f.write(f"{run}\t{step}\t{bench}\t{out['avg@k']:.6f}\t{out['pass@1']:.6f}\t{out['pass@k']:.6f}"
            f"\t{n}\t{out['wall_seconds']:.3f}\t32768\t34817\t{out['model']}\n")
print(f"[agg] {run}@{step} {bench}: avg@k={out['avg@k']:.4f} pass@1={out['pass@1']:.4f} "
      f"n={n} ({nshards} 片)")
