import os
import json
import traceback
from typing import List, Tuple
from multiprocessing import get_context

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
from datasets import load_dataset


# -----------------------
# CONFIG
# -----------------------
version = 1
MODEL_NAME = "Qwen/Qwen3-14B"
LOCAL_DIR = "./models/Qwen3-14B"
DATASET_DIR = f"./dataset/Omni-MATH_{version}"
GPU_IDS = [0, 1, 2, 3]

ENABLE_THINKING = True
MAX_MODEL_LEN = 32768
GPU_MEM_UTIL = 0.90

DATASET_NAME = "KbsdJames/Omni-MATH"
DATASET_SPLIT = None      # или "test"/"train"
TAKE_N = 100

MAX_OUTPUT_TOKENS = 8000

GUIDED_SCHEMA = {
    "type": "array",
    "items": {
        "type": "array",
        "minItems": 2,
        "maxItems": 2,
        "items": {"type": "integer", "minimum": 1},
    },
}


# -----------------------
# TOKEN PIECES (exact substrings)
# -----------------------
def make_token_pieces_exact(text: str, tokenizer) -> List[str]:
    enc = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    offsets = enc["offset_mapping"]
    pieces = [text[s:e] for (s, e) in offsets]
    # строгая проверка на точное восстановление
    if "".join(pieces) != text:
        # fallback по одному токену
        enc2 = tokenizer(text, add_special_tokens=False)
        ids = enc2["input_ids"]
        pieces = [
            tokenizer.decode([tid], clean_up_tokenization_spaces=False, skip_special_tokens=False)
            for tid in ids
        ]
        if "".join(pieces) != text:
            # здесь лучше вернуть пусто и пометить как bad на верхнем уровне
            return []
    return pieces


def token_dict_string(pieces: List[str]) -> str:
    items = []
    for i, p in enumerate(pieces, start=1):
        items.append(f"{i}:{json.dumps(p, ensure_ascii=False)}")
    return "{" + ", ".join(items) + "}"


def build_segmentation_user_content(text: str, pieces: List[str]) -> str:
    N = len(pieces)
    tokmap = token_dict_string(pieces)

    return f"""Task: Return ONLY a JSON array of [start, end] token index pairs (inclusive).
These pairs must partition ALL tokens into sentence chunks.

Rules:
- Token indices start at 1 and end at N.
- Pairs must be strictly increasing and non-overlapping.
- First pair must start at 1.
- Last pair must end at N.
- Next_start must equal prev_end + 1 (no gaps).
- Prefer to end chunks at sentence boundaries when possible: '.', '?', '!' or a newline.
- Do NOT output the sentences. Output ONLY the index pairs as JSON.

Example:
Tokens (index -> exact token text):
{{1:"The", 2:"box", 3:" ", 4:"is", 5:"stra", 6:"nge", 7:".", 8:" ", 9:"I", 10:" ", 11:"am", 12:" ", 13:"not", 14:"."}}
Correct output:
[[1,7],[8,14]]

Now do it for this input.

Tokens (index -> exact token text):
{tokmap}

N = {N}

Return ONLY JSON.
"""


# -----------------------
# vLLM worker (1 GPU per process)
# -----------------------
def _worker_generate_proc(
    gpu_id: int,
    local_path: str,
    indexed_chat_prompts: List[Tuple[int, str]],
    sampling_kwargs: dict,
    llm_kwargs: dict,
    out_q,
):
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        os.environ.setdefault("OMP_NUM_THREADS", "1")

        from vllm import LLM, SamplingParams
        from vllm.sampling_params import GuidedDecodingParams

        if not indexed_chat_prompts:
            out_q.put(("ok", []))
            return

        llm = LLM(model=local_path, tensor_parallel_size=1, **llm_kwargs)

        guided = GuidedDecodingParams(json=sampling_kwargs["guided_json_schema"])
        sp = SamplingParams(
            temperature=sampling_kwargs["temperature"],
            top_p=sampling_kwargs["top_p"],
            max_tokens=sampling_kwargs["max_tokens"],
            guided_decoding=guided,
        )

        idxs = [i for i, _ in indexed_chat_prompts]
        prompts = [p for _, p in indexed_chat_prompts]

        outs = llm.generate(prompts, sp)
        res = [(idxs[i], outs[i].outputs[0].text) for i in range(len(outs))]
        out_q.put(("ok", res))

    except Exception:
        # ошибки не печатаем наружу — просто вернём как payload
        out_q.put(("err", (gpu_id, traceback.format_exc())))


def run_dp_generate(chat_prompts: List[str], local_path: str) -> Tuple[List[str], List[int]]:
    """
    Возвращает:
      - raw_outputs: список строк (в порядке входа) для тех элементов, которые реально были отправлены
      - bad_indices: индексы входных промптов, которые не сгенерились из-за падения GPU процесса
    """
    buckets: List[List[Tuple[int, str]]] = [[] for _ in GPU_IDS]
    for i, p in enumerate(chat_prompts):
        buckets[i % len(GPU_IDS)].append((i, p))

    sampling_kwargs = dict(
        temperature=0.0,
        top_p=1.0,
        max_tokens=MAX_OUTPUT_TOKENS,
        guided_json_schema=GUIDED_SCHEMA,
    )

    llm_kwargs = dict(
        dtype="bfloat16",
        gpu_memory_utilization=GPU_MEM_UTIL,
        max_model_len=MAX_MODEL_LEN,
        enforce_eager=True,
    )

    ctx = get_context("spawn")
    out_q = ctx.Queue()

    procs = []
    active = 0
    for gpu_id, bucket in zip(GPU_IDS, buckets):
        if not bucket:
            continue
        p = ctx.Process(
            target=_worker_generate_proc,
            args=(gpu_id, local_path, bucket, sampling_kwargs, llm_kwargs, out_q),
        )
        p.daemon = False
        p.start()
        procs.append(p)
        active += 1

    results = []
    bad_indices = []

    # собираем результаты с процессов
    for _ in range(active):
        status, payload = out_q.get()
        if status == "ok":
            results.extend(payload)
        else:
            # gpu упал — пометим все индексы из соответствующего bucket как bad
            # но bucket-индексы нам тут не передали, поэтому просто пометим "всё плохо" позже
            # (реально редко; можно усложнить протокол, если надо)
            pass

    for p in procs:
        p.join()

    # восстановим список raw_outputs, но если каких-то индексов нет — пометим bad
    results.sort(key=lambda x: x[0])
    got = {i: txt for i, txt in results}
    raw_outputs = []
    for i in range(len(chat_prompts)):
        if i in got:
            raw_outputs.append(got[i])
        else:
            raw_outputs.append("")   # пусто как маркер
            bad_indices.append(i)

    return raw_outputs, bad_indices


# -----------------------
# Parse + validate (no exceptions outward)
# -----------------------
def parse_pairs_json(s: str):
    data = json.loads(s.strip())
    if not isinstance(data, list):
        return None
    pairs = []
    for item in data:
        if (not isinstance(item, list)) or len(item) != 2:
            return None
        a, b = item
        if not (isinstance(a, int) and isinstance(b, int)):
            return None
        pairs.append((a, b))
    return pairs


def validate_pairs(pairs: List[Tuple[int, int]], N: int) -> bool:
    if not pairs:
        return False
    if pairs[0][0] != 1:
        return False
    if pairs[-1][1] != N:
        return False
    prev_end = 0
    for (a, b) in pairs:
        if not (1 <= a <= b <= N):
            return False
        if a != prev_end + 1:
            return False
        prev_end = b
    return True


def pairs_to_segments(pieces: List[str], pairs: List[Tuple[int, int]]) -> List[str]:
    return ["".join(pieces[a - 1 : b]) for (a, b) in pairs]


# -----------------------
# MAIN
# -----------------------
if __name__ == "__main__":
    os.makedirs(LOCAL_DIR, exist_ok=True)
    os.makedirs(DATASET_DIR, exist_ok=True)

    hf_token = "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"
    local_path = snapshot_download(repo_id=MODEL_NAME, local_dir=LOCAL_DIR, token=hf_token)

    tokenizer = AutoTokenizer.from_pretrained(local_path)

    # dataset
    if DATASET_SPLIT is not None:
        ds = load_dataset(DATASET_NAME, token=hf_token, split=DATASET_SPLIT)
    else:
        dsdict = load_dataset(DATASET_NAME, token=hf_token)
        split_name = list(dsdict.keys())[0]
        ds = dsdict[split_name]

    base = ds.select(range(min(TAKE_N, len(ds))))
    texts = base["solution"]

    user_contents = []
    token_pieces_all = []

    for t in texts:
        pieces = make_token_pieces_exact(t, tokenizer)
        token_pieces_all.append(pieces)
        # если даже exact pieces не построились — всё равно добавим заглушку, а потом пометим bad
        if not pieces:
            user_contents.append("")  # маркер
        else:
            user_contents.append(build_segmentation_user_content(t, pieces))

    # chat prompts
    chat_prompts = []
    for uc in user_contents:
        if not uc:
            chat_prompts.append("")  # маркер
            continue
        messages = [{"role": "user", "content": uc}]
        chat_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=ENABLE_THINKING,
        )
        chat_prompts.append(chat_text)

    # generate: для пустых промптов ничего не шлём, но проще — отправим как есть и потом пометим bad
    raw_outputs, bad_indices = run_dp_generate(chat_prompts, local_path)
    print(len(raw_outputs))
    good_indices = []
    good_segments = []

    for i, (t, pieces, out) in enumerate(zip(texts, token_pieces_all, raw_outputs)):
        # если токены не построились или генерация пустая — bad
        if (not pieces) or (not out):
            bad_indices.append(i)
            continue

        pairs = None
        try:
            pairs = parse_pairs_json(out)
        except Exception:
            pairs = None

        if pairs is None:
            bad_indices.append(i)
            continue

        if not validate_pairs(pairs, N=len(pieces)):
            bad_indices.append(i)
            continue

        segs = pairs_to_segments(pieces, pairs)

        # строгая проверка обратимости
        if "".join(segs) != t:
            bad_indices.append(i)
            continue

        good_indices.append(i)
        good_segments.append(segs)

    # уникализируем bad
    bad_indices = sorted(set(bad_indices))

    # возвращаем ДАТАСЕТ ТОЛЬКО с успешными (сегменты добавляем только туда)
    good_ds = base.select(good_indices)
    good_ds = good_ds.add_column("solution_sentences", good_segments)

    pretty_txt = os.path.join(DATASET_DIR, "good_dataset_pretty.txt")
    with open(pretty_txt, "w", encoding="utf-8") as f:
        for i, ex in enumerate(good_ds):
            f.write("=" * 80 + "\n")
            f.write(f"IDX {i}\n\n")
            f.write("SOLUTION:\n")
            f.write(ex["solution"] + "\n\n")
            f.write("SEGMENTS:\n")
            for seg in ex["solution_sentences"]:
                f.write(repr(seg) + "\n")
            f.write("\n")
    
    parquet_path = os.path.join(DATASET_DIR, "good_dataset.parquet")
    good_ds.to_parquet(parquet_path)

    bad_json_path = os.path.join(DATASET_DIR, "bad_indices.json")
    with open(bad_json_path, "w", encoding="utf-8") as f:
        json.dump(bad_indices, f, ensure_ascii=False)



    # теперь у тебя:
    # - good_ds: только успешные примеры + solution_sentences
    # - bad_indices: индексы (в base), которые можно перегнать повторно
    #
    # Никаких ошибок/исключений не печатаем.
