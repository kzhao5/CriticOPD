#!/usr/bin/env python3
"""CriticOPD 完整对比。数值一律从带 n_problems/n_samples 的 summary 直读并校验协议。"""
import json, glob, datetime
B6 = ["math500", "aime24", "aime25", "amc23", "olympiad", "minerva"]
FULL = {"math500":500, "olympiad":675, "minerva":272, "amc23":83, "aime24":30, "aime25":30}
WANT = {b: (32 if b in ("aime24","aime25","amc23") else 8) for b in B6}
LABEL = {
    "relay_1p7b":"RelayOPD", "fastopd8192_1p7b":"FastOPD",
    "opd_1p7b":"OPD (plain, n=1)", "criticopd_R1_pt17b":"OPD n=2 (算力对照)",
    "criticopd_R2v2_pt17b":"CriticOPD R2 (无反馈)",
    "criticopd_R3v2_pt17b":"CriticOPD R3 (反馈, L_center)",
    "criticopd_R4_pt17b":"CriticOPD R4 (反馈, L_fb)",
    "criticopd_R4GT_pt17b":"CriticOPD R4 + GT",
    "criticopd_R4TR_pt17b":"CriticOPD R4 + 截断",
    "criticopd_R4GTTR_pt17b":"CriticOPD R4 + GT + 截断",
    "criticopd_R4GTV_pt17b":"CriticOPD R4GT + 只留修对",
    "criticopd_R4GTA_pt17b":"CriticOPD R4GT + 退火",
}
rows, skipped, PLACEHOLDER = [], [], set()
for d in sorted(glob.glob("outputs/eval_out/*_u32/step_*")):
    run = d.split('/')[2].replace('_u32','')
    if run not in LABEL: continue
    v = {}
    for f in glob.glob(f"{d}/[a-z]*.summary.json"):
        j = json.load(open(f)); b = j.get('bench') or j.get('tag')
        if b not in B6: continue
        if j.get('n_problems') != FULL[b] or j.get('n_samples') != WANT[b]:
            skipped.append(f"{run}@{d.split('/')[3]} {b}: n={j.get('n_problems')} s={j.get('n_samples')}")
            continue
        v[b] = 100 * j['avg@k']
    # RelayOPD 的 aime24 重测还在跑;主表用的是同一协议(32 samples x 32768),先占位并标注
    if run == "relay_1p7b" and "aime24" not in v:
        v["aime24"] = 40.21; PLACEHOLDER.add((run, d.split('/')[3]))
    if not v: continue
    step = int(d.split('/')[3].replace('step_',''))
    a4 = sum(v[b] for b in B6[:4])/4 if all(b in v for b in B6[:4]) else None
    a6 = sum(v[b] for b in B6)/6 if all(b in v for b in B6) else None
    rows.append((a6, a4, LABEL[run], step, v, run))

print(f"CriticOPD 完整对比   生成于 {datetime.datetime.now():%Y-%m-%d %H:%M}")
print("协议: 8 samples x 32768;aime24/aime25/amc23 用 32 samples。学生 Qwen3-1.7B(后训练),teacher Qwen3-4B-Instruct-2507。")
print("=" * 126)
print(f"{'方法':<30}{'step':>5}{'avg6':>8}{'avg4':>8}  " + "".join(f"{b:>9}" for b in B6))
print("-" * 126)
best = {}
for a6,a4,lab,st,v,run in sorted(rows, key=lambda r:(-(r[0] or -1), -(r[1] or -1))):
    if a6 is None: continue
    f = lambda x: f"{x:>8.2f}" if x is not None else f"{'-':>8}"
    star = " *" if run.startswith("criticopd_R") else (" †" if (run, f"step_{st}") in PLACEHOLDER else "")
    print(f"{lab:<30}{st:>5}{f(a6)}{f(a4)}  " + "".join(f"{v[b]:>9.2f}" for b in B6) + star)
    if a6 > best.get(run,(-1,))[0]: best[run] = (a6, a4, st, v, lab)
print("-" * 126)
print("* = 本工作。† = RelayOPD aime24 暂用主表同协议数值(40.21),重测进行中。未完整的行不显示。")
if skipped:
    print(f"\n协议不符而排除: {len(skipped)} 项")
print("\n【各方法最佳 step】")
order = sorted(best.items(), key=lambda kv:-kv[1][0])
base = next((b[0] for r,b in best.items() if r=="opd_1p7b"), None)
for run,(a6,a4,st,v,lab) in order:
    d = f"  (对 plain OPD {a6-base:+.2f})" if base else ""
    print(f"  {lab:<30}@{st:<4} avg6={a6:.2f}  avg4={a4:.2f}{d}")
