import argparse
import os
import numpy as np
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer

def build_prompt(problem: str, sol_prefix: str, instruction: str) -> list[dict]:
    # verl expects chat-style prompt list
    content = (
        f"{instruction}\n\n"
        f"### Context\n"
        f"{problem}\n\n"
        f"{sol_prefix}"
    )
    return [{"role": "user", "content": content}]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, required=True)  # tokenizer source
    ap.add_argument("--out_dir", type=str, default='Omni-MATH')
    ap.add_argument("--offset", type=int, default=16)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train_ratio", type=float, default=0.954833)
    ap.add_argument("--max_tasks", type=int, default=100000)
    ap.add_argument("--target", choices=["token", "sentence"], default="token")
    args = ap.parse_args()
    assert args.k <= args.offset, ('offset should be notsmaller than number of predicted tokens')
    os.makedirs(args.out_dir, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    ds = load_dataset("parquet", data_files="./dataset/Omni-MATH_6/dataset_with_solution_sentences.parquet", split="train") # typically only one split
    split_name = list(ds.keys())[0]
    base = ds[split_name]

    if args.target = "sentence":
        prediction = "sentence"
    elif args.k == 1:
        prediction = "token"
    else:
        prediction = f"{args.k} tokens"
    
    instruction = (
        f"Complete the given text under ‘### Context’ by predicting the next {prediction} exactly. "
        "Please reason step by step, and put ONLY the final answer after '####'"
    )
     
    rows = []
    for idx, ex in enumerate(base):
        problem = ex["problem"]
        solution = ex["solution"]
        difficulty = float(ex.get("difficulty", np.nan))
        domain = ex.get("domain", None)
        source = ex.get("source", None)
        if args.target == "token":
            sol_ids = tok.encode(solution, add_special_tokens=False)
        else:
            
            
        # Create many (context -> next token) tasks by stepping through solution tokens
        for t in range(0, max(0, len(sol_ids) - args.k), args.offset):
            if t + args.k > len(sol_ids):
                break

            if args.target == "token":
                sol_prefix = tok.decode(sol_ids[:t], skip_special_tokens=False)
                target = tok.decode(sol_ids[t:t+args.k], skip_special_tokens=False)
            else:
                sol

            rows.append({
                "data_source": "KbsdJames/Omni-MATH",
                "prompt": build_prompt(problem, sol_prefix, instruction),
                "ability": "math",
                # store target token(s) for RLP reward computation
                "reward_model": {"style": "rule", "ground_truth": solution},
                "difficulty": difficulty,
                "extra_info": {
                    "index": idx,
                    "t": t,
                    "offset": args.offset,
                    "k": args.k,
                    "domain": domain,
                    "source": source,
                },
            })
            
    print(rows[0:5])
    task_ds = Dataset.from_list(rows[:args.max_tasks])
    task_ds = task_ds.sort("difficulty")

    # create train/test split
    task_ds = task_ds.train_test_split(test_size=1.0 - args.train_ratio, seed=args.seed)
    train_ds, test_ds = task_ds["train"], task_ds["test"]
    print(test_ds.num_rows)

    train_ds.to_parquet(os.path.join(args.out_dir, "train.parquet"))
    test_ds.to_parquet(os.path.join(args.out_dir, "test.parquet"))

if __name__ == "__main__":
    main()
