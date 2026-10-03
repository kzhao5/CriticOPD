#!/usr/bin/env python3
"""训练算力成本表 + 柱状图(Qwen3-1.7B 学生,主表所选 checkpoint)。

GPU 小时统一折算到 A100:
  - GRPO / SKD / CriticOPD(4xA100)与 OPD / FastOPD / RelayOPD(8xA100):训练日志逐步耗时之和 x GPU 数,直接用
  - SFT / SeqKD / TRD 原来在 4xH100 上训练:在 4xA100 上用相同配置、相同数据顺序重跑前 6 步
    (outputs/timing/),第 2-6 步与原日志同一批步数对比得到换算系数,再乘原日志的全部逐步耗时
  - 离线造数据原来在 B200 上:在 4xA100 上抽 4000 题重测每题 GPU 秒,乘以训练到所选步数实际用到的样本数
FLOPs 来自 criticopd/compute_cost.py(cost_table.json)。
输出:compute_cost.{md,csv,json}、fig_compute_cost.{pdf,png}
"""
import csv, json, os, re, statistics, sys
sys.path.insert(0, "/home/kzhao2/Relay-OPD/criticopd")
import compute_cost as C   # noqa: E402

R = "/home/kzhao2/Relay-OPD"
OUT = f"{R}/results_final"
TM = f"{R}/outputs/timing"
B = 128
AVG = {"SFT": None, "SeqKD": 41.73, "TRD": 41.59, "GRPO": 45.66, "SKD": 45.58, "OPD": 50.18,
       "FastOPD": 52.42, "RelayOPD": 53.24, "CriticOPD": 52.35}          # 主表 1.7B 平均分(SFT 主表仍为 XX)
LABEL = {"SeqKD": "KD"}                                                  # 论文主表里的叫法


def sft_elapsed(path):
    """SFT 训练器不记录每步时间:从进度条 'k/139 [mm:ss<' 取到第 k 步为止的累计秒数。"""
    el = {}
    for m in re.finditer(r"(\d+)/139 \[(?:(\d+):)?(\d+):(\d+)<", open(path, errors="ignore").read()):
        k = int(m.group(1)); h = int(m.group(2) or 0)
        el[k] = h * 3600 + int(m.group(3)) * 60 + int(m.group(4))
    return el


def h100_to_a100():
    """同一批步数(第 2-6 步)上 A100 / H100 的耗时比。"""
    f = {}
    for m in ("seqkd", "trd"):
        a = C.parse(f"{TM}/{m}_a100/training.log"); h = C.parse(f"{R}/outputs/checkpoints/{m}_pt17b/training.log")
        ss = [s for s in range(2, 7) if s in a and s in h]
        f[m] = sum(a[s]["timing_s/step"] for s in ss) / sum(h[s]["timing_s/step"] for s in ss)
    a, h = sft_elapsed(f"{TM}/sft_a100/training.log"), sft_elapsed(f"{R}/outputs/checkpoints/sft_pt17b/training.log")
    k = max(x for x in a if x in h and x <= 6)          # 两边都有记录的最后一步(进度条不一定每步都打印)
    f["sft"] = (a[k] - a[1]) / (h[k] - h[1])
    return f


def gen_gpu_sec_per_row(path_list):
    """造数据:每题 GPU 秒(各分片各占 1 张卡,分片墙钟相加 / 题数)。"""
    tot = rows = 0
    for p in path_list:
        d = json.load(open(p)); rows_p = sum(s["rows"] for s in d["shards"])
        tot += sum(s["wall_seconds"] for s in d["shards"])
        rows = rows_p if not rows else rows      # 多阶段(TRD)按同一批题累加时间
    return tot / rows


def main():
    rows = {r["name"]: r for r in json.load(open(f"{R}/criticopd/cost_table.json"))}
    fac = h100_to_a100()
    gt = gen_gpu_sec_per_row([f"{TM}/gen_teacher/teacher_trajectories.parquet.summary.json"])
    gd = gen_gpu_sec_per_row([f"{TM}/gen_trd/trd_student_trajectories.jsonl.summary.json",
                              f"{TM}/gen_trd/trd_trajectories.parquet.summary.json"])
    print(f"H100 -> A100 训练耗时系数: {', '.join(f'{k} {v:.2f}' for k, v in fac.items())}")
    print(f"A100 造数据每题 GPU 秒: teacher 轨迹 {gt:.2f},TRD(学生+改写) {gd:.2f}")

    out = []
    for name in ["SFT", "SeqKD", "TRD", "GRPO", "SKD", "OPD", "FastOPD", "RelayOPD", "CriticOPD"]:
        r = rows[name]; S = r["step"]
        train = r["gpu_h"] * fac.get(name.lower(), 1.0)
        data = {"SFT": gt, "SeqKD": gt, "TRD": gd}.get(name, 0.0) * S * B / 3600
        out.append({"method": LABEL.get(name, name), "step": S, "avg": AVG[name],
                    "orig_hw": f"{r['gpus']}x{r['hw']}", "train_gpu_h": train, "data_gpu_h": data,
                    "total_gpu_h": train + data, "flops_1e18": r["flops"] / 1e18,
                    "flops_student_inf": r["f_s"] / 1e18, "flops_teacher_inf": r["f_t"] / 1e18,
                    "flops_train": r["f_tr"] / 1e18})
    opd = next(o for o in out if o["method"] == "OPD")
    for o in out:
        o["gpu_h_rel_opd"] = o["total_gpu_h"] / opd["total_gpu_h"]; o["flops_rel_opd"] = o["flops_1e18"] / opd["flops_1e18"]

    json.dump({"factors_h100_to_a100": fac, "gen_gpu_sec_per_row_a100": {"teacher": gt, "trd": gd}, "rows": out},
              open(f"{OUT}/compute_cost.json", "w"), indent=1)
    with open(f"{OUT}/compute_cost.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
    with open(f"{OUT}/compute_cost.md", "w") as f:
        f.write("| Method | Step | Avg | Train (A100 GPU-h) | Offline data (A100 GPU-h) | Total (A100 GPU-h) | vs OPD | FLOPs (1e18) | vs OPD |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for o in out:
            f.write(f"| {o['method']} | {o['step']} | {o['avg'] if o['avg'] is not None else 'XX'} | {o['train_gpu_h']:.1f} | "
                    f"{o['data_gpu_h']:.1f} | {o['total_gpu_h']:.1f} | {o['gpu_h_rel_opd']:.2f}x | {o['flops_1e18']:.2f} | "
                    f"{o['flops_rel_opd']:.2f}x |\n")
    print(open(f"{OUT}/compute_cost.md").read())
    plot(out)


def plot(out):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = [o["method"] for o in out]; y = list(range(len(out)))[::-1]
    ours = ["#3b6fb6" if n == "CriticOPD" else "#9aa5b1" for n in names]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    a = ax[0]
    a.barh(y, [o["train_gpu_h"] for o in out], color=ours, label="Training")
    a.barh(y, [o["data_gpu_h"] for o in out], left=[o["train_gpu_h"] for o in out], color=ours, alpha=0.45,
           hatch="//", edgecolor="white", label="Offline data generation")
    for yi, o in zip(y, out):
        a.text(o["total_gpu_h"] + 2, yi, f"{o['total_gpu_h']:.0f}" + (f"  (avg {o['avg']:.2f})" if o["avg"] else ""),
               va="center", fontsize=8)
    a.set_yticks(y); a.set_yticklabels([f"{n} @{o['step']}" for n, o in zip(names, out)])
    a.set_xlabel("A100 GPU-hours to the reported checkpoint"); a.set_xlim(0, max(o["total_gpu_h"] for o in out) * 1.45)
    a.legend(loc="lower right", fontsize=8, frameon=False); a.set_title("(a) Measured GPU-hours", fontsize=10)
    b = ax[1]
    left = [0.0] * len(out)
    for key, col, lab in (("flops_train", "#4c72b0", "Student training"), ("flops_student_inf", "#55a868", "Student inference"),
                          ("flops_teacher_inf", "#dd8452", "Teacher inference")):
        vals = [o[key] for o in out]
        b.barh(y, vals, left=left, color=col, label=lab, edgecolor=["black" if n == "CriticOPD" else col for n in names],
               linewidth=[1.2 if n == "CriticOPD" else 0 for n in names])
        left = [l + v for l, v in zip(left, vals)]
    for yi, o in zip(y, out):
        b.text(o["flops_1e18"] + 0.06, yi, f"{o['flops_rel_opd']:.2f}x", va="center", fontsize=8)
    b.set_xlabel(r"Training FLOPs ($10^{18}$) to the reported checkpoint"); b.set_xlim(0, max(o["flops_1e18"] for o in out) * 1.18)
    b.legend(loc="lower right", fontsize=8, frameon=False); b.set_title("(b) Estimated FLOPs (ratio to OPD)", fontsize=10)
    for x in ax:
        x.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(f"{OUT}/fig_compute_cost.{ext}", dpi=200, bbox_inches="tight")
    print(f"图: {OUT}/fig_compute_cost.pdf / .png")


if __name__ == "__main__":
    main()
