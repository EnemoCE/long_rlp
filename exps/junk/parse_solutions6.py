# split_with_emoji_guided.py
import os
import re
import json
from typing import List, Tuple
from multiprocessing import get_context

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
from datasets import load_dataset

# =========================
# CONFIG
# =========================
version = 5
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-8B")
LOCAL_DIR = os.environ.get("LOCAL_DIR", "./models/Qwen3-8B")
OUT_DIR = f"./dataset/Omni-MATH_{version}"

DATASET_NAME = os.environ.get("DATASET_NAME", "KbsdJames/Omni-MATH")
DATASET_SPLIT = os.environ.get("DATASET_SPLIT", None)  # e.g. "test"
TAKE_N = int(os.environ.get("TAKE_N", "100"))

GPU_IDS = [0, 1, 2, 3]  # one GPU per worker process

DELIM = os.environ.get("DELIM", "🙂")  # the single separator symbol
ENABLE_THINKING = 1

GPU_MEM_UTIL = float(os.environ.get("GPU_MEM_UTIL", "0.90"))
DTYPE = os.environ.get("DTYPE", "bfloat16")
ENFORCE_EAGER = os.environ.get("ENFORCE_EAGER", "1") == "1"

# Sentence boundary candidates: after punctuation, and after newline.
END_PUNCT = {".", "!", "?", "…"}  # you can extend if needed


# =========================
# PROMPTS
# =========================
def build_system_prompt(delim: str) -> str:
    # максимально простой английский + пример (как ты просила)
    return f"""You will be given TEXT.
Output the same TEXT exactly, character-by-character, and ONLY insert the separator {delim} at ALL sentence boundaries.

Rules:
- Do NOT change, delete, or reorder any character from TEXT.
- Do NOT add any other characters (no explanations, no labels).
- Keep all spaces and newlines exactly as in TEXT.
- You need to insert only the separator {delim} at ALL sentence boundaries.

Example:
TEXT:
The box is strange. I am not. I love snorkling.
OUTPUT:
The box is strange.{delim} I am not.{delim} I love snorkling.
"""

def build_user_prompt(solution: str) -> str:
    return "TEXT:\n" + solution


# =========================
# GUIDED REGEX: exact sequence + optional DELIM only at allowed boundaries
# =========================
def find_boundary_positions(text: str) -> List[int]:
    """
    Returns positions p in [0..len(text)] meaning "DELIM may be inserted between text[p-1] and text[p]".
    We'll allow:
      - after any END_PUNCT char
      - after newline char
    """
    pos = set()
    for i, ch in enumerate(text):
        if ch in END_PUNCT:
            pos.add(i + 1)
        if ch == "\n":
            pos.add(i + 1)
    # do not allow at beginning or beyond end unless it’s also a boundary (end insertion usually useless)
    pos.discard(0)
    # allowing at end is harmless but pointless; keep it off
    pos.discard(len(text))
    return sorted(pos)

def build_exact_sequence_regex(text: str, delim: str) -> str:
    """
    Build regex matching exactly:
      chunk0 (delim)? chunk1 (delim)? ... chunkK
    where chunks come from slicing 'text' at allowed boundary positions.
    """
    boundaries = find_boundary_positions(text)
    # Split into chunks at boundaries (literal text)
    chunks = []
    start = 0
    for p in boundaries:
        chunks.append(text[start:p])
        start = p
    chunks.append(text[start:])

    # Escape literal chunks for regex
    esc_chunks = [re.escape(c) for c in chunks]
    esc_delim = re.escape(delim)

    # Optional delim ONLY between chunks (i.e., at allowed boundaries)
    # ^chunk0(?:DELIM)?chunk1(?:DELIM)?...chunkK$
    pattern = "^" + esc_chunks[0]
    for c in esc_chunks[1:]:
        pattern += f"(?:{esc_delim})?" + c
    pattern += "$"

    # NOTE: pattern can be large (≈ len(text)), but it's the correct constraint.
    return pattern


# =========================
# VALIDATION + SPLIT
# =========================
def validate_and_split(output: str, original: str, delim: str) -> Tuple[bool, List[str]]:
    """
    Accept iff removing DELIM gives exactly original text.
    Return segments split by DELIM.
    """
    if output.replace(delim, "") != original:
        return False, []
    # split keeps exact spaces/newlines because we didn't touch them
    segs = output.split(delim)
    # ensure concat back to original
    if "".join(segs) != original:
        return False, []
    return True, segs


# =========================
# vLLM worker (1 GPU per process)
# =========================
def _worker_generate_proc(
    gpu_id: int,
    local_path: str,
    indexed_prompts: List[Tuple[int, str, str, int]],  # (idx, chat_prompt, original_solution, max_tokens)
    out_q,
    llm_kwargs: dict,
    delim: str,
):
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        os.environ.setdefault("OMP_NUM_THREADS", "1")

        from vllm import LLM, SamplingParams
        from vllm.sampling_params import GuidedDecodingParams

        if not indexed_prompts:
            out_q.put(("ok", [], []))
            return

        llm = LLM(model=local_path, tensor_parallel_size=1, **llm_kwargs)

        prompts = []
        sparams = []
        metas = []  # (idx, original_solution)

        for idx, chat_prompt, sol, max_toks in indexed_prompts:
            # Important: delimiter collision makes split ambiguous; mark as bad.
            if delim in sol:
                metas.append((idx, sol))
                prompts.append("")  # dummy
                sparams.append(None)
                continue

            regex = build_exact_sequence_regex(sol, delim)
            guided = GuidedDecodingParams(regex=regex)

            sp = SamplingParams(
                temperature=0.0,
                top_p=1.0,
                max_tokens=max_toks,
                guided_decoding=guided,
            )
            prompts.append(chat_prompt)
            sparams.append(sp)
            metas.append((idx, sol))

        good = []
        bad = []

        # Generate only for valid entries (skip delimiter-collision dummies)
        real_map = []
        real_prompts = []
        real_sparams = []
        for i, sp in enumerate(sparams):
            if sp is None or prompts[i] == "":
                idx, _ = metas[i]
                bad.append(idx)
                continue
            real_map.append(i)
            real_prompts.append(prompts[i])
            real_sparams.append(sp)

        if real_prompts:
            outs = llm.generate(real_prompts, real_sparams)
            for j, out in enumerate(outs):
                meta_i = real_map[j]
                idx, sol = metas[meta_i]
                text = out.outputs[0].text
                ok, segs = validate_and_split(text, sol, delim)
                if ok:
                    good.append((idx, segs))
                else:
                    bad.append(idx)

        out_q.put(("ok", good, bad))

    except Exception:
        # по твоему требованию: без логов/эксепшенов наружу
        out_q.put(("ok", [], [idx for idx, *_ in indexed_prompts]))


def run_dp_generate(chat_prompts: List[str], originals: List[str], max_tokens_list: List[int],
                    local_path: str, gpu_ids: List[int], llm_kwargs: dict, delim: str):
    n = len(chat_prompts)

    buckets = [[] for _ in gpu_ids]
    for i in range(n):
        buckets[i % len(gpu_ids)].append((i, chat_prompts[i], originals[i], max_tokens_list[i]))

    ctx = get_context("spawn")
    out_q = ctx.Queue()

    procs = []
    active = 0
    for gpu_id, bucket in zip(gpu_ids, buckets):
        if not bucket:
            continue
        p = ctx.Process(
            target=_worker_generate_proc,
            args=(gpu_id, local_path, bucket, out_q, llm_kwargs, delim),
        )
        p.daemon = False
        p.start()
        procs.append(p)
        active += 1

    segments_col: List[List[str]] = [[] for _ in range(n)]
    bad_indices: List[int] = []

    for _ in range(active):
        _, good, bad = out_q.get()
        for idx, segs in good:
            segments_col[idx] = segs
        bad_indices.extend(bad)

    for p in procs:
        p.join()

    bad_indices = sorted(set(bad_indices))
    return segments_col, bad_indices


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    os.makedirs(LOCAL_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    hf_token = os.environ.get("HF_TOKEN")

    local_path = snapshot_download(
        repo_id=MODEL_NAME,
        local_dir=LOCAL_DIR,
        token=hf_token,
    )

    tokenizer = AutoTokenizer.from_pretrained(local_path)

    # Load dataset
    dsdict = load_dataset(DATASET_NAME, token=hf_token)
    split_name = DATASET_SPLIT if DATASET_SPLIT is not None else list(dsdict.keys())[0]
    base = dsdict[split_name].select(range(min(TAKE_N, len(dsdict[split_name]))))

    solutions = base["solution"]

    system_prompt = build_system_prompt(DELIM)

    # Build chat prompts
    chat_prompts = []
    prompt_lens = []
    sol_lens = []

    for sol in solutions:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": build_user_prompt(sol)},
        ]
        chat = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=ENABLE_THINKING,
        )
        chat_prompts.append(chat)

        prompt_ids = tokenizer.encode(chat, add_special_tokens=False)
        sol_ids = tokenizer.encode(sol, add_special_tokens=False)
        prompt_lens.append(len(prompt_ids))
        sol_lens.append(len(sol_ids))

    # Choose engine max_model_len tightly (saves VRAM)
    # Output must contain full solution + some DELIMs -> about sol_len + overhead
    desired_out = [sl + 256 for sl in sol_lens]
    max_total = max(pl + out for pl, out in zip(prompt_lens, desired_out))
    engine_max_model_len = min(32768, max_total + 64)

    # Cap per-sample max_tokens to fit engine_max_model_len
    max_tokens_list = []
    for pl, out in zip(prompt_lens, desired_out):
        room = max(1, engine_max_model_len - pl - 8)
        max_tokens_list.append(min(out, room))

    llm_kwargs = dict(
        dtype=DTYPE,
        gpu_memory_utilization=GPU_MEM_UTIL,
        max_model_len=engine_max_model_len,
        enforce_eager=ENFORCE_EAGER,
        trust_remote_code=True,
    )

    segments_col, bad_indices = run_dp_generate(
        chat_prompts=chat_prompts,
        originals=solutions,
        max_tokens_list=max_tokens_list,
        local_path=local_path,
        gpu_ids=GPU_IDS,
        llm_kwargs=llm_kwargs,
        delim=DELIM,
    )

    # Add column (bad -> [])
    out_ds = base.add_column("solution_sentences", segments_col)

    # =========================
    # SAVE (as you asked earlier)
    # =========================
    parquet_path = os.path.join(OUT_DIR, "dataset_with_solution_sentences.parquet")
    out_ds.to_parquet(parquet_path)

    # TXT (JSONL): one example per line
    txt_path = os.path.join(OUT_DIR, "dataset_with_solution_sentences.jsonl")
    with open(txt_path, "w", encoding="utf-8") as f:
        for ex in out_ds:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # bad indices
    bad_path = os.path.join(OUT_DIR, "bad_indices.txt")
    with open(bad_path, "w", encoding="utf-8") as f:
        for idx in bad_indices:
            f.write(str(idx) + "\n")
