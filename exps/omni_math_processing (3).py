import argparse
import os
import numpy as np
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from pathlib import Path
import json

def build_prompt(problem: str, sol_prefix: str, instruction: str) -> list[dict]:
    # verl expects chat-style prompt list
    content = (
        f"{instruction}\n\n"
        f"### Context\n"
        f"Problem:\n\n{problem}\n\n"
        f"Solution:\n\n"
        f"{sol_prefix}"
    )
    return [{"role": "user", "content": content}]

def build_dataset_element(problem: str, sol_prefix: str, instruction: str, target: str, difficulty: float, extra_info: dict) -> dict:
    return {
            "data_source": "KbsdJames/Omni-MATH",
            "prompt": build_prompt(problem, sol_prefix, instruction),
            "ability": "math",
            # store target token(s) for RLP reward computation
            "reward_model": {"style": "rule", "ground_truth": target},
            "difficulty": difficulty,
            "extra_info": extra_info,
            }

def main():
    JSONL_PATH = os.path.join('Omni-MATH', "dataset_with_solution_sentences.jsonl")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, required=False)  # tokenizer source
    ap.add_argument("--out_dir", type=str, default='Omni-MATH')
    ap.add_argument("--offset", type=int, default=16)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train_ratio", type=float, default=0.954833)
    ap.add_argument("--max_tasks", type=int, default=1000000)
    ap.add_argument("--target", choices=["token", "sentence", "full_sentence"], default="token")
    args = ap.parse_args()
    assert args.k <= args.offset, ('offset should be notsmaller than number of predicted tokens')
    os.makedirs(args.out_dir, exist_ok=True)

    

    model_id = args.model_name
    save_dir = Path("./tokenizers") / model_id
    tokenizer_ok = (save_dir / "tokenizer.json").exists() or (save_dir / "tokenizer_config.json").exists()
    
    if not tokenizer_ok:
        save_dir.mkdir(parents=True, exist_ok=True)
        tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        tok.save_pretrained(save_dir)
        print(f"Saved tokenizer to {save_dir}")
    else:
        print(f"Tokenizer files already present in {save_dir}, skipping download.")

    tok = AutoTokenizer.from_pretrained("./tokenizers/Qwen/Qwen3-1.7b", use_fast=True)

    ds = load_dataset("parquet", data_files="./dataset/Omni-MATH_final/dataset_with_solution_sentences.parquet", split="train") # typically only one split
    if hasattr(ds,'keys'):
        split_name = list(ds.keys())[0]
        base = ds[split_name]
    else:
        base=ds

    if args.target == "sentence":
        prediction = "sentence"
    elif args.k == 1:
        prediction = "token"
    else:
        prediction = f"{args.k} tokens"
    
    instruction = (
        f"Complete the given text under ‘### Context’ by predicting next {prediction} exactly. "
        "Please reason step by step, and put ONLY the final answer after '####'."
    )
    
    

    rows = []
    for idx, ex in enumerate(base):
        problem = ex["problem"]
        solution = ex["solution"]
        difficulty = float(ex.get("difficulty", np.nan))
        domain = ex.get("domain", None)
        source = ex.get("source", None)
        extra_info = {
                    "index": idx,
                    "offset": args.offset,
                    "k": args.k,
                    "domain": domain,
                    "source": source,
                    }
        if args.target == "token":
            sol_ids = tok.encode(solution, add_special_tokens=False)
            # Create many (context -> next token) tasks by stepping through solution tokens
            for t in range(0, max(0, len(sol_ids) - args.k + 1), args.offset):
                if t + args.k > len(sol_ids):
                    break

                if args.target == "token":
                    sol_prefix = tok.decode(sol_ids[:t], skip_special_tokens=False)
                    target = tok.decode(sol_ids[t:t+args.k], skip_special_tokens=False)
                extra_info["t"] = t
                rows.append(build_dataset_element(problem, sol_prefix, instruction, target, difficulty, extra_info))
        else:
            solution_sentences = ex["solution_sentences"]
            for i in range(len(solution_sentences)):
                sol_prefix = "".join(solution_sentences[0:i])
                target = solution_sentences[i]
                if not args.target == "full_sentence":
                    pre_target_ids = tok.encode(solution_sentences[i])
                    target = tok.decode(pre_target_ids[0:args.k])
                    if not len(pre_target_ids) >= args.k:
                        print('fail!')
                        extra_info['k'] = len(pre_target_ids)
                extra_info["t"] = i
                rows.append(build_dataset_element(problem, sol_prefix, instruction, target, difficulty, extra_info))
            
    #print(rows[0:5])
    task_ds = Dataset.from_list(rows[:args.max_tasks])
    task_ds = task_ds.select(range(10))
    """
    for ele in task_ds:
        for key, value in ele.items():
            print(key, value, "\n\n\n")
            if key == "prompt":
                for key1, value1 in value[0].items():
                    print(key1, value1, "\n\n")
        print('\n\n\n\n')
    """
    # create train/test split
    # shuffled by default
    task_ds = task_ds.train_test_split(test_size=1.0 - args.train_ratio, seed=args.seed)
    train_ds, test_ds = task_ds["train"], task_ds["test"]
    train_ds = train_ds.sort("difficulty")
    print("Number of training samples: ", train_ds.num_rows, "\n", "Number of test samples: ", test_ds.num_rows)

    train_ds.to_parquet(os.path.join(args.out_dir, "train.parquet"))
    test_ds.to_parquet(os.path.join(args.out_dir, "test.parquet"))
    with open(JSONL_PATH, "w", encoding="utf-8") as f:
        for ex in train_ds:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
