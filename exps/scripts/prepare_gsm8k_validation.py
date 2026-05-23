import argparse
import os
import re

import datasets


def extract_solution(solution_str: str) -> str:
    solution = re.search(r"#### (\-?[0-9\.\,]+)", solution_str)
    if solution is None:
        raise ValueError("Could not extract GSM8K solution from answer.")
    return solution.group(1).replace(",", "")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local_save_dir",
        default="~/data/gsm8k_boxed",
        help="Directory where the validation parquet files will be written.",
    )
    parser.add_argument(
        "--local_dataset_path",
        default=None,
        help="Optional local dataset path passed to datasets.load_dataset().",
    )
    args = parser.parse_args()

    if args.local_dataset_path is not None:
        dataset = datasets.load_dataset(args.local_dataset_path, "main")
    else:
        dataset = datasets.load_dataset("openai/gsm8k", "main")

    instruction = "Let's think step by step and put the final answer within \\boxed{}."

    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example["question"]
            answer_raw = example["answer"]
            solution = extract_solution(answer_raw)
            return {
                # Route validation through boxed math scoring while keeping GSM8K metric names honest.
                "data_source": "openai/gsm8k_boxed",
                "prompt": [{"role": "user", "content": question_raw + " " + instruction}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer_raw,
                    "question": question_raw,
                    "original_data_source": "openai/gsm8k",
                },
            }

        return process_fn

    train_dataset = dataset["train"].map(function=make_map_fn("train"), with_indices=True)
    test_dataset = dataset["test"].map(function=make_map_fn("test"), with_indices=True)

    local_save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)
    train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_save_dir, "test.parquet"))


if __name__ == "__main__":
    main()
