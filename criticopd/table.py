#!/usr/bin/env python3
"""汇总统一协议下的对比表。

run 名有三套后缀约定:基线用 <run>_u、plain OPD 用 opd_1p7b_b16k、我们的 arm 直接用
criticopd_*。这里统一映射成论文里的方法名。
avg4 = math500/aime24/aime25/amc23;avg6 = 再加 olympiad/minerva(与主表定义一致)。
"""
import csv, collections
B6 = ["math500", "aime24", "aime25", "amc23", "olympiad", "minerva"]
NAME = {
    # 统一协议(8x32768,aime 32x32768)的结果都在 _u32 目录。
    # 旧的 _u / _b16k 是 16384 协议,受截断偏差影响,只能做同方法的 step 间比较。
    "opd_1p7b_u32":              ("OPD (R0, n=1)",         80),
    "criticopd_R1_pt17b_u32":    ("OPD n=2 (R1)",          80),
    "fastopd8192_1p7b_u32":      ("FastOPD",               60),
    "relay_1p7b_u32":            ("RelayOPD",              40),
    "lucid_T1donly_pt17b_u32":   ("Lucid-OPD",             60),
    "criticopd_R2v2_pt17b_u32":  ("CriticOPD R2 (无反馈)",  None),
    "criticopd_R3v2_pt17b_u32":  ("CriticOPD R3 (反馈)",    None),
    "criticopd_R4_pt17b_u32":    ("CriticOPD R4 (L_fb)",    None),
}
v = collections.defaultdict(dict)
for r in csv.reader(open("outputs/eval_out/results.tsv"), delimiter="\t"):
    if len(r) < 9 or r[0] not in NAME or r[2] not in B6:
        continue
    # 统一协议:大题库 8x16384,aime 32x32768。其它组合是历史数据,排除。
    ok = r[8] == "32768"          # 统一协议:全部 32768,采样数由 bench 决定
    if ok:
        v[(r[0], int(r[1]))][r[2]] = float(r[3]) * 100

print(f"{'方法':<24}{'step':>5}" + "".join(f"{b:>10}" for b in B6) + f"{'avg4':>8}{'avg6':>8}")
for key in sorted(v, key=lambda k: (list(NAME).index(k[0]), k[1])):
    run, step = key; d = v[key]
    if NAME[run][1] is not None and step != NAME[run][1]:
        continue                                  # 基线只显示主表选定的 step
    cell = lambda b: f"{d[b]:>10.2f}" if b in d else f"{'-':>10}"
    a4 = [d[b] for b in B6[:4] if b in d]; a6 = [d[b] for b in B6 if b in d]
    f4 = f"{sum(a4)/4:>8.2f}" if len(a4) == 4 else f"{'-':>8}"
    f6 = f"{sum(a6)/6:>8.2f}" if len(a6) == 6 else f"{'-':>8}"
    print(f"{NAME[run][0]:<24}{step:>5}" + "".join(cell(b) for b in B6) + f4 + f6)
