"""把主批次(out)和扩充批次(out_b,rollout_id 从 1000 起)合并成 out_all,供 report.py 使用。
批次级统计(全部 rollout 的 token 占比等)只用主批次的 raw_rollouts(900 道题全部 rollout)。"""
import glob, json, os, shutil, sys
SRC = sys.argv[1:] or ["out", "out_b"]
DST = "out_all"
os.makedirs(DST, exist_ok=True)
def jl(p): return [json.loads(l) for l in open(p)] if os.path.exists(p) else []
def cat(name, pats):
    rows = [x for s in SRC for p in pats for f in sorted(glob.glob(f"{s}/{p}")) for x in jl(f)]
    with open(f"{DST}/{name}", "w") as f:
        for x in rows: f.write(json.dumps(x, ensure_ascii=False) + "\n")
    return len(rows)
n = {}
for name in ("rollouts.jsonl", "graded_e1.jsonl", "graded_e6.jsonl", "graded_e7.jsonl", "critic_partial.jsonl", "critic_full_nogt.jsonl",
             "critic_full_gt.jsonl", "critic_single_partial.jsonl", "critic_single_full.jsonl", "judge.jsonl"):
    n[name] = cat(name, [name])
n["judge_partial.jsonl"] = cat("judge_partial.jsonl", ["judge_partial_[0-9]*.jsonl"])
n["graded_e10.jsonl"] = cat("graded_e10.jsonl", ["graded_e10.jsonl"])
n["judge_first_0.jsonl"] = cat("judge_first_0.jsonl", ["judge_first_[0-9]*.jsonl"])
for a in ("teacher", "student"): n[f"score_{a}_0.jsonl"] = cat(f"score_{a}_0.jsonl", [f"score_{a}_[0-9]*.jsonl"])
C = {"A": {}, "B": [], "C": []}; MC = {}; AN = {}
for s in SRC:
    if os.path.exists(f"{s}/closing.json"):
        c = json.load(open(f"{s}/closing.json")); C["A"].update(c["A"]); C["B"] += c["B"]; C["C"] += c["C"]
    if os.path.exists(f"{s}/method_cut.json"): MC.update(json.load(open(f"{s}/method_cut.json")))
    if os.path.exists(f"{s}/e10_anchors.json"): AN.update(json.load(open(f"{s}/e10_anchors.json")))
json.dump(C, open(f"{DST}/closing.json", "w")); json.dump(MC, open(f"{DST}/method_cut.json", "w"))
if AN: json.dump(AN, open(f"{DST}/e10_anchors.json", "w"))
for f in glob.glob(f"{SRC[0]}/raw_rollouts_*.jsonl") + [f"{SRC[0]}/raw_correct.json", f"{SRC[0]}/raw_capped.json"]:
    shutil.copy(f, DST)
print({k: v for k, v in n.items()}, "A", len(C["A"]), "B", len(C["B"]), "C", len(C["C"]), "method_cut", len(MC))
