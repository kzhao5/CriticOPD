#!/usr/bin/env python3
"""从原始日志里提取每次训练的实际配置(发布用,只读原始日志,不改动任何东西)。

对每个运行生成 configs/<run>/:
  launch_command.txt   实际执行的训练命令(bash -x 展开后的那一行,含全部 Hydra 覆盖),另附每个参数一行的版本
  resolved_config.json 训练器启动时打印的完整配置(verl PPO 训练器;SFT 训练器不打印,以命令为准)
  slurm_jobs.tsv       每段 Slurm 任务:作业号、状态、耗时、节点、GPU、提交命令
并生成 reference/metrics/<run>.csv(训练日志里每一步的全部指标)和 reference/training_logs/<run>.log.gz。
"""
import ast, csv, glob, gzip, json, os, re, subprocess, sys

R = "/home/kzhao2/Relay-OPD"
OUT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGDIRS = [f"{R}/logs", "/nobackup/autodelete/usr/kzhao2/relay-opd/slurm_logs"]
ANSI = re.compile(r"\x1b\[[0-9;]*m|\[3[0-9]m|\[0m|\[1m|\[22m|\[39m")
PREFIX = re.compile(r"^\(\w+ pid=\d+(?:, ip=[\d.]+)?\)\s?")

RUNS = ["sft_pt17b", "seqkd_pt17b", "trd_pt17b", "grpo_pt17b", "skd_pt17b", "opd_1p7b", "fastopd8192_1p7b",
        "relay_1p7b", "criticopd_R4GT_pt17b",
        "criticopd_R1_pt17b", "criticopd_R2v2_pt17b", "criticopd_R3v2_pt17b", "criticopd_R4_pt17b",
        "criticopd_R4TR_pt17b", "criticopd_R4GTV_pt17b", "criticopd_R4GTA_pt17b", "criticopd_R4GTB_pt17b",
        "criticopd_R4GTK2_pt17b", "criticopd_R4GTKL_pt17b", "criticopd_R4GTR2_pt17b",
        "opd_pt06b", "fastopd8192_pt06b", "relay_pt06b", "skd_pt06b", "grpo_pt06b"]
SKIP = re.compile(r"^(u_|ec_|mc_|ro_|rm_|e4_|e06_|ev|agg|g_|tm_|ms_|pb_data|cost|adv_|dep_|kerr)")


def clean(line):
    return PREFIX.sub("", ANSI.sub("", line.rstrip("\n")))


def slurm_logs(run):
    pat = re.compile(rf"(^|_){re.escape(run)}(_s\d+)?_(\d+)\.log$")
    out = {}
    for d in LOGDIRS:
        for f in glob.glob(f"{d}/*.log"):
            b = os.path.basename(f); m = pat.search(b)
            if m and not SKIP.match(b):
                out.setdefault(int(m.group(3)), []).append(f)   # 同一作业号可能两处都有
    return [out[k] for k in sorted(out)]


def launch_and_env(logs):
    for f in [x for fs in logs for x in fs]:
        lines = [clean(l) for l in open(f, errors="ignore")]
        cmd = next((l for l in lines if re.match(r"^\++ (python3? -m verl\.trainer|torchrun )", l)), None)
        if cmd:
            return f, cmd.lstrip("+ "), []
    return None, None, []


def resolved_config(train_log):
    if not os.path.exists(train_log): return None
    lines = [clean(l) for l in open(train_log, errors="ignore")]
    for i, l in enumerate(lines):
        if l.lstrip().startswith("{'actor_rollout_ref'") or l.lstrip().startswith("{'algorithm'"):
            buf = ""
            for j in range(i, min(i + 3000, len(lines))):
                buf += lines[j].strip() + " "
                if buf.count("{") == buf.count("}"):
                    try:
                        return ast.literal_eval(buf)
                    except Exception:
                        return {"_raw_printed_config": buf.strip()}   # 含非字面量的值时原样保存打印出来的配置
    return None


def sacct(jobs):
    if not jobs: return []
    o = subprocess.run(["sacct", "-n", "-X", "-P", "-j", ",".join(map(str, jobs)), "-o",
                        "JobID,JobName,State,Elapsed,NodeList,AllocTRES,SubmitLine"], capture_output=True, text=True).stdout
    return [l.split("|") for l in o.strip().splitlines() if l]


def main():
    sys.path.insert(0, f"{R}/criticopd"); import compute_cost as C
    os.makedirs(f"{OUT}/reference/metrics", exist_ok=True); os.makedirs(f"{OUT}/reference/training_logs", exist_ok=True)
    summary = []
    for run in RUNS:
        d = f"{OUT}/configs/{run}"; os.makedirs(d, exist_ok=True)
        logs = slurm_logs(run)
        src, cmd, env = launch_and_env(logs)
        tl = f"{R}/outputs/checkpoints/{run}/training.log"
        if cmd:
            with open(f"{d}/launch_command.txt", "w") as f:
                f.write(f"# 来源: {os.path.basename(src)}(bash -x 展开后的实际命令)\n{cmd}\n\n# 每个参数一行:\n")
                f.write(" \\\n    ".join(re.findall(r"(?:'[^']*'|\"[^\"]*\"|\S)+", cmd)) + "\n")
        cfg = resolved_config(tl)
        if cfg is not None:
            json.dump(cfg, open(f"{d}/resolved_config.json", "w"), indent=1, default=str)
        jobs = [int(re.search(r"_(\d+)\.log$", fs[0]).group(1)) for fs in logs]
        rows = sacct(jobs)
        with open(f"{d}/slurm_jobs.tsv", "w") as f:
            f.write("job_id\tname\tstate\telapsed\tnodes\talloc_tres\tsubmit_line\n")
            for r in rows: f.write("\t".join(r) + "\n")
        if os.path.exists(tl):
            L = C.parse(tl)
            keys = sorted({k for v in L.values() for k in v})
            with open(f"{OUT}/reference/metrics/{run}.csv", "w", newline="") as f:
                w = csv.writer(f); w.writerow(["step"] + keys)
                for s in sorted(L): w.writerow([s] + [L[s].get(k, "") for k in keys])
            with open(tl, "rb") as fi, gzip.open(f"{OUT}/reference/training_logs/{run}.log.gz", "wb") as fo:
                fo.write(fi.read())
        summary.append((run, len(logs), bool(cmd), len(env), cfg is not None, len(rows)))
    print(f"{'运行':28s} 日志 命令 完整配置 作业")
    for s in summary:
        print(f"{s[0]:28s} {s[1]:3d}  {'有' if s[2] else '缺'}   {'有' if s[4] else '缺'}   {s[5]:3d}")


if __name__ == "__main__":
    main()
