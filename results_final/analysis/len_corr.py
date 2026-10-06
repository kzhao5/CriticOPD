import glob, json, os, re, statistics
import numpy as _np
def pearsonr(x, y): return (float(_np.corrcoef(x, y)[0, 1]),)
def _rank(v):
    o = _np.argsort(v); r = _np.empty(len(v)); r[o] = _np.arange(len(v)); return r
def spearmanr(x, y): return (float(_np.corrcoef(_rank(_np.array(x)), _rank(_np.array(y)))[0, 1]),)
EV = "/home/kzhao2/Relay-OPD/outputs/eval_out"
B = ["aime24", "aime25", "amc23", "math500", "olympiad", "minerva"]
rows = []
for d in sorted(glob.glob(f"{EV}/*_u32/step_*")):
    run = d.split("/")[-2][:-4]; st = int(d.split("_")[-1])
    if not re.match(r"(opd_1p7b|fastopd8192_1p7b|relay_1p7b|criticopd_.*pt17b)", run): continue
    s = []
    for b in B:
        f = f"{d}/{b}.summary.json"
        if not os.path.exists(f): break
        s.append(100 * json.load(open(f))["avg@k"])
    if len(s) < 6: continue
    L = [json.loads(l)["gen_len"] for b in ("aime24", "aime25") for l in open(f"{d}/{b}.jsonl")]
    rows.append((run, st, sum(s) / 6, statistics.median(L)))
for r in rows: print(f"{r[0]:40s} {r[1]:4d} avg6={r[2]:.2f} medAIME={r[3]:.0f}")
x = [r[3] for r in rows]; y = [r[2] for r in rows]
print("n", len(rows), "pearson %.3f spearman %.3f" % (pearsonr(x, y)[0], spearmanr(x, y)[0]))
import numpy as np; k = np.polyfit(x, y, 1)[0]; print("slope per 1000 tokens %.3f" % (k * 1000))
for pat in ("criticopd_final_pt17b_seed4", "opd_1p7b"):
    sel = [r for r in rows if r[0].startswith(pat) and ((pat != "opd_1p7b") or r[1] == 80)]
    print(pat, [(r[0][-6:], r[1], round(r[3])) for r in sel], "mean median len", round(np.mean([r[3] for r in sel])))
