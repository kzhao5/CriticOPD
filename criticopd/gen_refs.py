#!/usr/bin/env python3
"""离线为训练题生成参考解答(teacher 独立做题,用标准答案核对,只保留做对的)。

两轮:第 1 轮每题做 1 次;没做对的题第 2 轮再做 3 次(共最多 4 次,与离线检验 ref_probe 一致)。
在做对的里面取最短的一份。键是题目文本(用户消息)的 sha1,训练时 agent loop 用同样的键查表。

用法: python3 gen_refs.py <shard> <nshards>     输出 refs/refs_<shard>.jsonl
"""
import hashlib, json, os, sys
sys.path.insert(0, "/home/kzhao2/Relay-OPD/relay-opd")
os.environ.setdefault("MATH_GRADER_PATH", "/home/kzhao2/Relay-OPD/relay-opd/opd/reward/grader")

TEACHER = "/home/kzhao2/OPD/model/Qwen3-4B-Instruct-2507"
TRAIN = "/home/kzhao2/OPD/datasets/dapo-math-17k.parquet"
HERE = os.path.dirname(os.path.abspath(__file__))
STOP_IDS = [151643, 151645]
MAXTOK, SEED = 8192, 4321


def main(shard, nsh):
    import pandas as pd
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    from opd.reward.math_reward import compute_score

    def ok(text, gt):
        try:
            v = compute_score(solution_str=text, ground_truth=gt)
            return float(v["score"] if isinstance(v, dict) else v) > 0.5
        except Exception:
            return False

    df = pd.read_parquet(TRAIN)
    # 题目清单与输出目录可配置:训练用 8 个进程加载数据,顺序与单进程不同,清单由 train40_todo_nw8.json 给出
    idx = json.load(open(os.environ.get("PROBLEMS_FILE", f"{HERE}/train40_problems.json")))["indices"][shard::nsh]
    tok = AutoTokenizer.from_pretrained(TEACHER, trust_remote_code=True)
    items = {}
    for i in idx:
        msgs = [dict(m) for m in list(df.iloc[i]["prompt"])]
        k = hashlib.sha1(msgs[-1]["content"].encode()).hexdigest()
        if k in items: continue
        items[k] = {"key": k, "row": int(i), "gt": str(df.iloc[i]["reward_model"]["ground_truth"]),
                    "ids": tok.encode(tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False),
                                      add_special_tokens=False), "cands": []}
    llm = LLM(model=TEACHER, dtype="bfloat16", gpu_memory_utilization=0.88,
              max_model_len=MAXTOK + 4096, enable_prefix_caching=True)
    sid = 0
    for rnd, n in ((1, 1), (2, 3)):
        todo = [it for it in items.values() if not any(c[0] for c in it["cands"])]
        reqs, meta, sps = [], [], []
        for it in todo:
            for _ in range(n):
                reqs.append({"prompt_token_ids": it["ids"]}); meta.append(it)
                sps.append(SamplingParams(temperature=0.7, top_p=0.95, max_tokens=MAXTOK, stop_token_ids=STOP_IDS,
                                          seed=SEED + shard * 1_000_000 + sid)); sid += 1   # 每个请求独立的种子
        if not reqs: break
        for it, o in zip(meta, llm.generate(reqs, sps)):
            txt = o.outputs[0].text
            good = o.outputs[0].finish_reason != "length" and ok(txt, it["gt"])
            it["cands"].append((good, len(o.outputs[0].token_ids), txt))
        print(f"[refs] shard {shard} 第 {rnd} 轮:{len(todo)} 题 x {n} 次,"
              f"累计做对 {sum(any(c[0] for c in it['cands']) for it in items.values())}/{len(items)}", flush=True)
    outdir = os.environ.get("REFS_DIR", f"{HERE}/refs")
    os.makedirs(outdir, exist_ok=True)
    with open(f"{outdir}/refs_{shard}.jsonl", "w") as f:
        for it in items.values():
            good = sorted((c for c in it["cands"] if c[0]), key=lambda c: c[1])
            f.write(json.dumps({"key": it["key"], "row": it["row"], "n_try": len(it["cands"]),
                                "n_ok": len(good), "ref": good[0][2] if good else None}) + "\n")


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]))
