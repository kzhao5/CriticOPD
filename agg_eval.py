import json, os, sys, glob
BENCH_ORDER=["aime24","aime25","amc23","math500","olympiad"]
def agg(step_dir):
    per={}  # bench -> [sum_correct, n]
    shards=sorted(glob.glob(os.path.join(step_dir,"shard_*")))
    for sh in shards:
        for f in glob.glob(os.path.join(sh,"*.summary.json")):
            try: d=json.load(open(f))
            except: continue
            b=d.get("bench"); n=d.get("n_problems",0); a=d.get("avg@k")
            if b is None or a is None or n==0: continue
            per.setdefault(b,[0.0,0]); per[b][0]+=a*n; per[b][1]+=n
    scores={b:(per[b][0]/per[b][1]) for b in per if per[b][1]>0}
    return scores, len(shards)
if __name__=="__main__":
    d=sys.argv[1]
    sc,nsh=agg(d)
    if not sc: print(f"{d}: (无分数)"); sys.exit()
    benches=[b for b in BENCH_ORDER if b in sc]+[b for b in sc if b not in BENCH_ORDER]
    row=" ".join(f"{b}={sc[b]*100:.1f}" for b in benches)
    avg5=sum(sc.values())/len(sc)*100
    core4=[b for b in ["aime24","aime25","amc23","math500"] if b in sc]
    avg4=sum(sc[b] for b in core4)/len(core4)*100 if core4 else float('nan')
    print(f"{os.path.basename(os.path.dirname(d))}/{os.path.basename(d)}  ({nsh}shard)")
    print(f"  {row}")
    print(f"  Avg5={avg5:.1f}  Avg4(核心)={avg4:.1f}")
