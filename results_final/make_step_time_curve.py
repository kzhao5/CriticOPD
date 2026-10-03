#!/usr/bin/env python3
"""训练过程中每一步的 GPU 时间(只用 4xA100 上的运行,不同硬件不混)。

数据:每个方法各 seed 的训练日志
  - verl 训练器:timing_s/step(每一步墙钟秒)
  - SFT 训练器不记每步时间:用进度条 'k/139 [mm:ss<' 的累计时间求差
GPU 时间 = 每步墙钟 x 4 张卡。
输出:step_time.csv(方法, seed, step, 每步墙钟秒, 每步 GPU 秒)、step_time_summary.json、
      fig_step_time.{pdf,png}:(a) 每步 GPU 分钟,同方法各 seed 的均值 +- 标准差阴影(先做 5 步滑动平均)
                              (b) 累计 GPU 小时,均值 +- 标准差
"""
import csv, json, os, re, statistics, sys
sys.path.insert(0, "/home/kzhao2/Relay-OPD/criticopd")
import compute_cost as C   # noqa: E402

CK = "/home/kzhao2/Relay-OPD/outputs/checkpoints"
OUT = "/home/kzhao2/Relay-OPD/results_final"
G = 4
# 方法: (目录前缀, 所选步数, [在 4xA100 上的运行目录])
METHODS = {
    "SFT": ("sft_pt17b", 40, ["sft_pt17b_seed43", "sft_pt17b_seed44"]),
    "KD": ("seqkd_pt17b", 139, ["seqkd_pt17b_seed43", "seqkd_pt17b_seed44"]),
    "TRD": ("trd_pt17b", 40, ["trd_pt17b_seed43", "trd_pt17b_seed44"]),
    "GRPO": ("grpo_pt17b", 120, ["grpo_pt17b", "grpo_pt17b_seed43", "grpo_pt17b_seed44"]),          # seed 42 原本就在 4xA100
    "SKD": ("skd_pt17b", 120, ["skd_pt17b", "skd_pt17b_seed43", "skd_pt17b_seed44"]),
    "OPD": ("opd_1p7b", 80, ["opd_1p7b_seed43", "opd_1p7b_seed44"]),
    "FastOPD": ("fastopd8192_1p7b", 60, ["fastopd8192_1p7b_seed43", "fastopd8192_1p7b_seed44"]),
    "RelayOPD": ("relay_1p7b", 40, ["relay_1p7b_seed43", "relay_1p7b_seed44"]),
    "CriticOPD": ("criticopd_R4GT_pt17b", 40, ["criticopd_R4GT_pt17b"]),                           # 4xA100,目前 1 个 seed
}
COLORS = {"SFT": "#8c8c8c", "KD": "#b07aa1", "TRD": "#9c755f", "GRPO": "#e15759", "SKD": "#f28e2b",
          "OPD": "#4e79a7", "FastOPD": "#76b7b2", "RelayOPD": "#59a14f", "CriticOPD": "#1f3a93"}


def sft_steps(path):
    el = {}
    for m in re.finditer(r"(\d+)/139 \[(?:(\d+):)?(\d+):(\d+)<", open(path, errors="ignore").read()):
        k = int(m.group(1)); el[k] = int(m.group(2) or 0) * 3600 + int(m.group(3)) * 60 + int(m.group(4))
    return {k: el[k] - el[k - 1] for k in el if k - 1 in el}


def collect():
    rows = []
    for name, (_, S, runs) in METHODS.items():
        for run in runs:
            f = f"{CK}/{run}/training.log"
            if not os.path.exists(f): continue
            if name == "SFT":
                per = sft_steps(f)
            else:
                L = C.parse(f); per = {s: d["timing_s/step"] for s, d in L.items() if "timing_s/step" in d and s >= 1}
            for s in sorted(per):
                if s <= S:
                    rows.append({"method": name, "run": run, "step": s, "wall_s": per[s], "gpu_s": per[s] * G})
    return rows


def roll(v, w=5):
    return [statistics.mean(v[max(0, i - w + 1): i + 1]) for i in range(len(v))]


def main():
    rows = collect()
    with open(f"{OUT}/step_time.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "run", "step", "wall_s", "gpu_s"]); w.writeheader(); w.writerows(rows)
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
    summ = {}
    for name, (_, S, _) in METHODS.items():
        runs = sorted({r["run"] for r in rows if r["method"] == name})
        series = []
        for run in runs:
            d = {r["step"]: r["gpu_s"] for r in rows if r["run"] == run}
            if len(d) < 3: continue
            steps = list(range(1, max(d) + 1))
            v = [d.get(s, d[min(d, key=lambda x: abs(x - s))]) for s in steps]   # 个别缺步用最近一步补
            series.append((steps, v))
        if not series: continue
        n = min(len(s[0]) for s in series)        # 只画所有 seed 都跑到的步数
        steps = series[0][0][:n]
        per = [[x / 60 for x in roll(v[:n])] for _, v in series]
        cum = [[sum(v[:i + 1]) / 3600 for i in range(n)] for _, v in series]
        mean = lambda m: [statistics.mean(c) for c in zip(*m)]
        sd = lambda m: [statistics.pstdev(c) if len(m) > 1 else 0.0 for c in zip(*m)]
        lab = f"{name} ({len(series)} seed{'s' if len(series) > 1 else ''})"
        for a, m in ((ax[0], per), (ax[1], cum)):
            mu, s = mean(m), sd(m)
            a.plot(steps, mu, color=COLORS[name], lw=2 if name == "CriticOPD" else 1.4, label=lab)
            if len(series) > 1:
                a.fill_between(steps, [x - y for x, y in zip(mu, s)], [x + y for x, y in zip(mu, s)], color=COLORS[name], alpha=0.18)
        summ[name] = {"seeds": len(series), "steps": n, "gpu_h_total_mean": mean(cum)[-1],
                      "gpu_h_total_sd": sd(cum)[-1], "target_step": S}
    ax[0].set_xlabel("Training step"); ax[0].set_ylabel("GPU-minutes per step (4xA100)")
    ax[0].set_title("(a) Per-step GPU time (5-step moving average, mean $\\pm$ std over seeds)", fontsize=9)
    ax[1].set_xlabel("Training step"); ax[1].set_ylabel("Cumulative GPU-hours (4xA100)")
    ax[1].set_title("(b) Cumulative GPU time", fontsize=9)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False); a.grid(alpha=0.25)
    ax[1].legend(fontsize=8, frameon=False, loc="upper left")
    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(f"{OUT}/fig_step_time.{ext}", dpi=200, bbox_inches="tight")
    json.dump(summ, open(f"{OUT}/step_time_summary.json", "w"), indent=1)
    for k, v in summ.items():
        print(f"{k:10s} {v['seeds']} 个 seed,已有 {v['steps']}/{v['target_step']} 步,累计 {v['gpu_h_total_mean']:.1f} ± {v['gpu_h_total_sd']:.1f} GPU 小时")


if __name__ == "__main__":
    main()
