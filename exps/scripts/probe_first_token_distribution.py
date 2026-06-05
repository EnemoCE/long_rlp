#!/usr/bin/env python3
"""Probe first-token probabilities for the base-model thought protocol.

This script checks whether ``<think>`` is a plausible first generated token for
the real training prompts. It intentionally lives outside the trainer so we can
run it cheaply on a VM before changing reward or rollout logic.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Parquet dataset with a verl-style prompt column.")
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B-Base")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--chat-template-file", default=None)
    parser.add_argument("--prompt-key", default="prompt")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-prompt-tokens", type=int, default=1024)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--dtype", default="bfloat16", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def as_messages(value: Any) -> list[dict[str, str]]:
    if isinstance(value, list):
        return [dict(item) for item in value]
    if hasattr(value, "tolist"):
        return as_messages(value.tolist())
    raise TypeError(f"Unsupported prompt value type: {type(value)!r}")


def get_dtype(name: str) -> torch.dtype | str:
    if name == "auto":
        return "auto"
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def format_prob(prob: float) -> str:
    if prob == 0.0:
        return "0"
    if prob < 1e-6:
        return f"{prob:.3e}"
    return f"{prob:.6f}"


def token_rank_and_prob(probs: torch.Tensor, token_id: int) -> tuple[int, float]:
    target_prob = float(probs[token_id].item())
    rank = int((probs > probs[token_id]).sum().item()) + 1
    return rank, target_prob


def top_tokens(tokenizer: Any, probs: torch.Tensor, top_k: int) -> list[dict[str, Any]]:
    values, ids = torch.topk(probs, k=top_k)
    rows = []
    for token_id, prob in zip(ids.tolist(), values.tolist(), strict=True):
        rows.append(
            {
                "id": token_id,
                "prob": prob,
                "text": tokenizer.decode([token_id], skip_special_tokens=False),
            }
        )
    return rows


def render_modes(tokenizer: Any, messages: list[dict[str, str]], chat_template: str | None) -> dict[str, str]:
    apply_kwargs = {}
    if chat_template is not None:
        apply_kwargs["chat_template"] = chat_template

    chat = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
        **apply_kwargs,
    )
    raw_user = messages[-1]["content"]
    return {
        "chat_template": chat,
        "raw_user": raw_user,
        "chat_template_prefill_think": chat + "<think>",
    }


def next_token_distribution(
    model: Any,
    tokenizer: Any,
    text: str,
    device: torch.device,
    max_prompt_tokens: int,
) -> torch.Tensor:
    input_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(input_ids) > max_prompt_tokens:
        input_ids = input_ids[-max_prompt_tokens:]
    tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    with torch.inference_mode():
        logits = model(input_ids=tensor).logits[0, -1].float()
    return torch.softmax(logits, dim=-1).detach().cpu()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    tokenizer_name = args.tokenizer or args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    chat_template = None
    if args.chat_template_file:
        chat_template = Path(args.chat_template_file).read_text()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    dtype = get_dtype(args.dtype)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device)
    model.eval()

    think_ids = tokenizer.encode("<think>", add_special_tokens=False)
    delimiter_ids = tokenizer.encode("####", add_special_tokens=False)
    targets = {
        "<think>": think_ids[0],
        "####": delimiter_ids[0],
    }

    dataframe = pd.read_parquet(args.dataset)
    sample_count = min(args.samples, len(dataframe))

    print(
        json.dumps(
            {
                "model": args.model,
                "dataset": args.dataset,
                "samples": sample_count,
                "top_k": args.top_k,
                "device": str(device),
                "dtype": str(dtype),
                "think_ids": think_ids,
                "delimiter_ids": delimiter_ids,
            },
            ensure_ascii=False,
        )
    )

    summary: dict[str, list[float]] = {}
    for row_idx in range(sample_count):
        messages = as_messages(dataframe.iloc[row_idx][args.prompt_key])
        rendered = render_modes(tokenizer, messages, chat_template)
        prompt_preview = messages[-1]["content"][:300].replace("\n", "\\n")
        print(json.dumps({"row": row_idx, "prompt_preview": prompt_preview}, ensure_ascii=False))

        for mode_name, text in rendered.items():
            probs = next_token_distribution(model, tokenizer, text, device, args.max_prompt_tokens)
            target_stats = {}
            for target_name, token_id in targets.items():
                rank, prob = token_rank_and_prob(probs, token_id)
                target_stats[target_name] = {"token_id": token_id, "rank": rank, "prob": prob}
                summary.setdefault(f"{mode_name}:{target_name}:rank", []).append(float(rank))
                summary.setdefault(f"{mode_name}:{target_name}:prob", []).append(float(prob))

            print(
                json.dumps(
                    {
                        "row": row_idx,
                        "mode": mode_name,
                        "target_stats": target_stats,
                        "top_tokens": top_tokens(tokenizer, probs, args.top_k),
                    },
                    ensure_ascii=False,
                )
            )

    print("SUMMARY")
    for key, values in sorted(summary.items()):
        mean = sum(values) / max(len(values), 1)
        if key.endswith(":prob"):
            print(f"{key} mean={format_prob(mean)} min={format_prob(min(values))} max={format_prob(max(values))}")
        else:
            print(f"{key} mean={mean:.1f} min={math.floor(min(values))} max={math.ceil(max(values))}")


if __name__ == "__main__":
    main()
