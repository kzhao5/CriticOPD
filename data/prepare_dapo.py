#!/usr/bin/env python3
"""Create the training prompts of the paper from DAPO-Math-17k.

The released DAPO-Math-17k file repeats each of its 17,917 problems 100 times. We keep the first occurrence of
each problem, which gives the training set of the paper (17,917 rows in verl format).

usage: python data/prepare_dapo.py [--source <local parquet>] [--out data/dapo-math-17k.parquet]
"""
import argparse
import os

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=None, help="local copy of the released parquet; downloaded if omitted")
    parser.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "dapo-math-17k.parquet"))
    args = parser.parse_args()
    source = args.source
    if source is None:
        from huggingface_hub import hf_hub_download
        source = hf_hub_download(repo_id="BytedTsinghua-SIA/DAPO-Math-17k", filename="data/dapo-math-17k.parquet",
                                 repo_type="dataset")
    data = pd.read_parquet(source)
    keys = data["extra_info"].apply(lambda info: info["index"])
    data = data[~keys.duplicated()].reset_index(drop=True)
    data.to_parquet(args.out)
    print(f"wrote {len(data)} problems to {args.out}")


if __name__ == "__main__":
    main()
