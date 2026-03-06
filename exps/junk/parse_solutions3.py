import os
import traceback
from typing import List, Tuple
from multiprocessing import get_context

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer


MODEL_NAME = "Qwen/Qwen3-8B"
LOCAL_DIR = "./models/Qwen3-8B"
GPU_IDS = [0, 1, 2, 3]

ENABLE_THINKING = True
MAX_MODEL_LEN = 10000
GPU_MEM_UTIL = 0.90


def _worker_generate_proc(
    gpu_id: int,
    local_path: str,
    indexed_prompts: List[Tuple[int, str]],
    sampling_kwargs: dict,
    llm_kwargs: dict,
    out_q,
):
    try:
        # 1 GPU per process
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        os.environ.setdefault("OMP_NUM_THREADS", "1")

        # Если хочешь A/B тест: можно выключить V1 здесь
        # os.environ["VLLM_USE_V1"] = "0"

        from vllm import LLM, SamplingParams

        if not indexed_prompts:
            out_q.put(("ok", []))
            return

        llm = LLM(
            model=local_path,
            tensor_parallel_size=1,   # одна GPU на процесс
            **llm_kwargs,
        )
        sp = SamplingParams(**sampling_kwargs)

        idxs = [i for i, _ in indexed_prompts]
        prompts = [p for _, p in indexed_prompts]

        outs = llm.generate(prompts, sp)
        res = [(idxs[i], outs[i].outputs[0].text) for i in range(len(outs))]
        out_q.put(("ok", res))

    except Exception:
        out_q.put(("err", f"GPU {gpu_id} failed:\n{traceback.format_exc()}"))


def run_dp_generate(formatted_prompts: List[str], local_path: str):
    # разбрасываем промпты round-robin по GPU
    buckets = [[] for _ in GPU_IDS]
    for i, p in enumerate(formatted_prompts):
        buckets[i % len(GPU_IDS)].append((i, p))

    # sampling
    if ENABLE_THINKING:
        sampling_kwargs = dict(
            temperature=0.6, top_p=0.95, top_k=20, min_p=0.0, max_tokens=5000
        )
    else:
        sampling_kwargs = dict(
            temperature=0.7, top_p=0.8, top_k=20, min_p=0.0, max_tokens=512
        )

    # vLLM config
    llm_kwargs = dict(
        dtype="bfloat16",
        gpu_memory_utilization=GPU_MEM_UTIL,
        max_model_len=MAX_MODEL_LEN,
        enforce_eager=True,   # как у тебя (без torch.compile)
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
        p.daemon = False  # <- критично: иначе снова нельзя будет плодить детей
        p.start()
        procs.append(p)
        active += 1

    results = []
    errors = []
    for _ in range(active):
        status, payload = out_q.get()
        if status == "ok":
            results.extend(payload)
        else:
            errors.append(payload)

    for p in procs:
        p.join()

    if errors:
        raise RuntimeError("\n\n".join(errors))

    results.sort(key=lambda x: x[0])
    return [text for _, text in results]


if __name__ == "__main__":
    os.makedirs(LOCAL_DIR, exist_ok=True)

    # НЕ хардкодь токен в коде — лучше export HF_TOKEN=...
    hf_token = "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"
    local_path = snapshot_download(
        repo_id=MODEL_NAME,
        local_dir=LOCAL_DIR,
        token=hf_token,  # или token=True если сделан huggingface-cli login
    )

    instruction = (
    "Split the solution into sentences and return ONLY a JSON array of strings.\n"
    "Example: [\"sentence1\", \"sentence2\"]\n\n"
    "Solution:\n"
)


    tokenizer = AutoTokenizer.from_pretrained(local_path)

    from datasets import load_dataset, Dataset
    ds = load_dataset("KbsdJames/Omni-MATH", token=hf_token)  # typically only one split
    split_name = list(ds.keys())[0]
    base = ds[split_name].select(range(10))
    prompts = base["solution"]



    formatted_prompts = []
    for p in prompts:
        messages = [{"role": "user", "content": instruction + str(p)}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=ENABLE_THINKING,
        )
        formatted_prompts.append(text)

    outputs = run_dp_generate(formatted_prompts, local_path)
    print(type(base))
    base = base.add_column(name="solution_sentences", column=outputs)
    
    for i, out in enumerate(base["solution_sentences"][0:10]):
        print("=" * 80)
        print(f"PROMPT #{i}: {prompts[i]}", "\n\n")
        print("-" * 80)
        print(out.strip(), "\n\n\n\n")
