#!/usr/bin/env python3
# calc_nanotron_memory.py
#
# CLI version of the key formulas from nanotron/predict_memory.
# Prints shell assignments so it can be sourced from bash.

import argparse
import math
from transformers import AutoConfig


def get_num_hidden_layers_in_pp(
    hidden_size: int,
    num_layers: int,
    vocab_size: int,
    intermediate_size: int,
    num_attention_heads: int,
    pp_size: int,
) -> int:
    if pp_size == 1:
        return num_layers

    pipeline_blocks = []
    block_costs = []

    # Embedding layer: zero-cost in the original implementation
    pipeline_blocks.append("embedding")
    block_costs.append(0)

    # Decoder layers
    decoder_cost = (
        4 * num_attention_heads * (hidden_size // num_attention_heads) * hidden_size
        + 3 * intermediate_size * hidden_size
    )
    for _ in range(num_layers):
        pipeline_blocks.append("decoder")
        block_costs.append(decoder_cost)

    # LM head
    pipeline_blocks.append("lm_head")
    block_costs.append(vocab_size * hidden_size)

    total_cost = sum(block_costs)
    target_cost_per_rank = total_cost / pp_size

    blocks_in_rank0 = 0
    current_cost = 0

    for block_idx, block_cost in enumerate(block_costs):
        current_cost += block_cost
        blocks_in_rank0 += 1

        remaining_ranks = pp_size - 1
        remaining_nonzero_blocks = sum(1 for c in block_costs[block_idx + 1 :] if c > 0)

        if (
            (remaining_ranks > 0 and remaining_nonzero_blocks <= remaining_ranks)
            or (current_cost >= target_cost_per_rank)
        ):
            break

    # Exclude first rank as it's the embedding layer
    num_hidden_layers_in_pp = blocks_in_rank0 - 1
    return num_hidden_layers_in_pp


def load_model_dims(model_name: str):
    cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    text_cfg = getattr(cfg, "text_config", None)
    src = text_cfg if text_cfg is not None else cfg

    hidden_size = int(getattr(src, "hidden_size"))
    num_layers = int(getattr(src, "num_hidden_layers"))
    intermediate_size = int(getattr(src, "intermediate_size"))
    num_attention_heads = int(getattr(src, "num_attention_heads"))
    num_key_value_heads = int(getattr(src, "num_key_value_heads", num_attention_heads))

    vocab_size = getattr(cfg, "vocab_size", None)
    if vocab_size is None:
        vocab_size = getattr(src, "vocab_size", None)
    if vocab_size is None:
        # Mirrors the space's multimodal fallback
        vocab_size = 256000
    vocab_size = int(vocab_size)

    tie_word_embeddings = bool(
        getattr(cfg, "tie_word_embeddings", getattr(src, "tie_word_embeddings", True))
    )

    return {
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "vocab_size": vocab_size,
        "intermediate_size": intermediate_size,
        "num_attention_heads": num_attention_heads,
        "num_key_value_heads": num_key_value_heads,
        "tie_word_embeddings": tie_word_embeddings,
    }


def calculate_memory_components(
    hidden_size: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    num_layers: int,
    vocab_size: int,
    intermediate_size: int,
    seq_len: int,
    mbs: int,
    batch_accum: int,
    tp: int,
    pp: int,
    dp: int,
    zero_stage: int,
    tie_word_embeddings: bool,
    full_checkpointing: bool = False,
):
    if pp == 1:
        num_hidden_layers_in_pp = num_layers
    else:
        num_hidden_layers_in_pp = get_num_hidden_layers_in_pp(
            hidden_size=hidden_size,
            num_layers=num_layers,
            vocab_size=vocab_size,
            intermediate_size=intermediate_size,
            num_attention_heads=num_attention_heads,
            pp_size=pp,
        )

    vocab_embeddings = vocab_size * hidden_size * (2 if (not tie_word_embeddings and pp == 1) else 1)

    layer_params = (
        (hidden_size * hidden_size * (1 + 2 * num_key_value_heads / num_attention_heads))  # qkv
        + (hidden_size * hidden_size)  # out_proj
        + (hidden_size * 2 * intermediate_size)  # gate_up_proj
        + (intermediate_size * hidden_size)  # down_proj
    )

    model_bf16_full = (vocab_embeddings + num_hidden_layers_in_pp * layer_params) * (2 / 1024 / 1024) / tp

    if zero_stage == 3:
        model_bf16 = model_bf16_full / dp
        fp32_params = 2 * model_bf16
        fp32_grads = 2 * model_bf16
        optimstates = 4 * model_bf16
        zero3_buffers = 2 * model_bf16
        ddp_grads_buffers = 0
    else:
        dp_if_zero = 1 if zero_stage == 0 else dp
        model_bf16 = model_bf16_full
        fp32_params = 2 * model_bf16 / dp_if_zero
        fp32_grads = 2 * model_bf16
        optimstates = 4 * model_bf16 / dp_if_zero
        zero3_buffers = 0
        use_ddp = (zero_stage == 0 and dp > 1)
        ddp_grads_buffers = model_bf16 if use_ddp else 0

    overhead = 72 + 32 * mbs

    decoder_layer_mib = (
        (seq_len * mbs * hidden_size / tp)
        * (2 / 1024 / 1024)
        * (4 * intermediate_size / hidden_size + 6 + 2 * num_key_value_heads / num_attention_heads + 2)
    )

    if pp > 1:
        activs = min(pp, batch_accum) * num_hidden_layers_in_pp * decoder_layer_mib
    else:
        cast_to_fp32 = seq_len * mbs * vocab_size * (2 / 1024 / 1024) * 2 / tp
        sharded_cross_entropy = cast_to_fp32
        base_activs = num_layers * decoder_layer_mib + cast_to_fp32 + sharded_cross_entropy

        if zero_stage == 3 and full_checkpointing:
            activs = base_activs / dp
        else:
            activs = base_activs

    memory_after_optimstates = (
        model_bf16
        + fp32_params
        + fp32_grads
        + optimstates
        + ddp_grads_buffers
        + zero3_buffers
        + overhead
    )

    peak_memory_tbi = memory_after_optimstates + activs

    return {
        "num_hidden_layers_in_pp": num_hidden_layers_in_pp,
        "model_bf16_mb": model_bf16,
        "fp32_params_mb": fp32_params,
        "fp32_grads_mb": fp32_grads,
        "optimizer_states_mb": optimstates,
        "ddp_gradient_buffers_mb": ddp_grads_buffers,
        "zero3_buffers_mb": zero3_buffers,
        "overhead_mb": overhead,
        "activations_mb": activs,
        "memory_after_optimstates_mb": memory_after_optimstates,
        "peak_memory_tbi_mb": peak_memory_tbi,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--seq_len", type=int, required=True)
    ap.add_argument("--mbs", type=int, default=1)
    ap.add_argument("--batch_accum", type=int, default=1)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--pp", type=int, default=1)
    ap.add_argument("--dp", type=int, default=1)
    ap.add_argument("--zero_stage", type=int, choices=[0, 1, 2, 3], default=0)
    ap.add_argument("--full_checkpointing", action="store_true")
    ap.add_argument("--emit_json", action="store_true")
    args = ap.parse_args()

    dims = load_model_dims(args.model_name)

    result = calculate_memory_components(
        hidden_size=dims["hidden_size"],
        num_attention_heads=dims["num_attention_heads"],
        num_key_value_heads=dims["num_key_value_heads"],
        num_layers=dims["num_layers"],
        vocab_size=dims["vocab_size"],
        intermediate_size=dims["intermediate_size"],
        seq_len=args.seq_len,
        mbs=args.mbs,
        batch_accum=args.batch_accum,
        tp=args.tp,
        pp=args.pp,
        dp=args.dp,
        zero_stage=args.zero_stage,
        tie_word_embeddings=dims["tie_word_embeddings"],
        full_checkpointing=args.full_checkpointing,
    )

    if args.emit_json:
        import json
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    # Conservative integer rounding for shell usage.
    print(f"one_batch_static_memory_mb={math.ceil(result['memory_after_optimstates_mb'])}")
    print(f"one_batch_dynamic_memory_mb={math.ceil(result['activations_mb'])}")
    print(f"peak_memory_tbi_mb={math.ceil(result['peak_memory_tbi_mb'])}")
    print(f"num_hidden_layers_in_pp={int(result['num_hidden_layers_in_pp'])}")


if __name__ == "__main__":
    main()