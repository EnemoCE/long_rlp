#!/usr/bin/env python3
"""Compute the average ground-truth answer length in tokens.

This helper is used by ``test_long_rlp.sh`` to reserve part of the response
budget for the final answer region. It reads the Omni-MATH training split and
prints ``ceil(mean_token_length)`` so the shell script can consume it directly.
"""

import argparse
import math

import pandas as pd
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to the training parquet file.")
    parser.add_argument("--tokenizer", required=True, help="Tokenizer path or HF model id.")
    return parser.parse_args()


def extract_ground_truth_lengths(frame: pd.DataFrame, tokenizer) -> list[int]:
    lengths: list[int] = []
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    for reward_model in frame["reward_model"]:
        if isinstance(reward_model, dict):
            ground_truth = str(reward_model.get("ground_truth", ""))
        elif hasattr(reward_model, "get"):
            ground_truth = str(reward_model.get("ground_truth", ""))
        else:
            ground_truth = str(reward_model)

        token_ids = tokenizer.encode(ground_truth, add_special_tokens=False)
        if not token_ids:
            token_ids = [eos_id]
        lengths.append(len(token_ids))

    return lengths


def main():
    args = parse_args()
    frame = pd.read_parquet(args.dataset, columns=["reward_model"])
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=False)

    lengths = extract_ground_truth_lengths(frame, tokenizer)
    mean_len = sum(lengths) / max(len(lengths), 1)
    print(int(math.ceil(mean_len)))


if __name__ == "__main__":
    main()
