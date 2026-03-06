import os
from multiprocessing import get_context
from typing import List, Tuple

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer


MODEL_NAME = "Qwen/Qwen3-8B"
LOCAL_DIR = "./models/Qwen3-8B"
GPU_IDS = [0, 1, 2, 3]

ENABLE_THINKING = False  # True -> будет <think>... блок
MAX_MODEL_LEN = 5000     # как у тебя
GPU_MEM_UTIL = 0.90

# ---------- worker: один процесс = одна GPU = одна копия модели ----------
def _worker_generate(
    gpu_id: int,
    local_path: str,
    indexed_prompts: List[Tuple[int, str]],
    sampling_kwargs: dict,
    llm_kwargs: dict,
):
    # Важно: ограничиваем видимые GPU ДО импорта vllm (и до первой CUDA-инициализации)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.setdefault("OMP_NUM_THREADS", "1")  # меньше CPU-шторма

    from vllm import LLM, SamplingParams  # импорт внутри процесса

    if not indexed_prompts:
        return []

    llm = LLM(
        model=local_path,
        tensor_parallel_size=1,   # одна GPU на процесс
        **llm_kwargs
    )
    sampling_params = SamplingParams(**sampling_kwargs)

    idxs = [i for i, _ in indexed_prompts]
    prompts = [p for _, p in indexed_prompts]

    outs = llm.generate(prompts, sampling_params)
    return [(idxs[i], outs[i].outputs[0].text) for i in range(len(outs))]


def run_dp_generate(formatted_prompts: List[str], local_path: str):
    # 1) round-robin раскладка промптов по GPU
    buckets = [[] for _ in GPU_IDS]
    for i, p in enumerate(formatted_prompts):
        buckets[i % len(GPU_IDS)].append((i, p))

    # 2) sampling params (просто как dict, чтобы легко сериализовалось между процессами)
    if ENABLE_THINKING:
        sampling_kwargs = dict(
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            min_p=0.0,
            max_tokens=512,
        )
    else:
        sampling_kwargs = dict(
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            min_p=0.0,
            max_tokens=2048,
        )

    # 3) параметры vLLM
    llm_kwargs = dict(
        dtype="bfloat16",
        gpu_memory_utilization=GPU_MEM_UTIL,
        max_model_len=MAX_MODEL_LEN,
        enforce_eager=True,   # как у тебя (без torch.compile / cudagraph — стабильнее)
    )

    # 4) запускаем только непустые задачи (если промптов меньше чем GPU)
    tasks = []
    for gpu_id, bucket in zip(GPU_IDS, buckets):
        if bucket:
            tasks.append((gpu_id, local_path, bucket, sampling_kwargs, llm_kwargs))

    # spawn — самый безопасный метод для CUDA/multiprocessing
    ctx = get_context("spawn")
    with ctx.Pool(processes=len(tasks)) as pool:
        results = pool.starmap(_worker_generate, tasks)

    # 5) склеиваем и восстанавливаем исходный порядок
    flat = [item for sub in results for item in sub]
    flat.sort(key=lambda x: x[0])
    return [text for _, text in flat]


if __name__ == "__main__":
    # ---- 0) скачивание модели локально ----
    os.makedirs(LOCAL_DIR, exist_ok=True)

    # НЕ хардкодь токен в коде: лучше export HF_TOKEN=...
    hf_token = "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"  # например: export HF_TOKEN=hf_****
    local_path = snapshot_download(
        repo_id=MODEL_NAME,
        local_dir=LOCAL_DIR,
        token=hf_token,     # None тоже ок, если модель публичная/кэш уже есть
    )

    # ---- 1) подготовка промптов ----
    user_prompts = [
        """
    Split Solution into sentences in a form of list ['sentence1', 'sentence2', ....]
    
    Solution:    
        Let \( r \) denote the largest real number satisfying \(\frac{b_n}{n^2} \geq r\) for all positive integers \( n \), where \( b_1, b_2, \dotsc \) are positive integers satisfying
    \[
    1 = \frac{b_1}{1^2} > \frac{b_2}{2^2} > \frac{b_3}{3^2} > \frac{b_4}{4^2} > \dotsb
    \]
    
    We aim to determine the possible values of \( r \).
    
    ### Claim 1: \( r = \frac{1}{2} \) works and is maximal.
    To achieve \( r = \frac{1}{2} \), consider the sequence \( b_n = \frac{n(n+1)}{2} \). Then,
    \[
    \frac{b_n}{n^2} = \frac{n(n+1)}{2n^2} = \frac{n+1}{2n} = \frac{1}{2} + \frac{1}{2n},
    \]
    which satisfies the condition \(\frac{b_n}{n^2} \geq \frac{1}{2}\).
    
    We can inductively show that \( b_n \leq \frac{n(n+1)}{2} \). The base case is given. Assuming the hypothesis holds for all integers less than \( n \), we have
    \[
    \frac{b_n}{n^2} < \frac{b_{n-1}}{(n-1)^2} \leq \frac{n}{2(n-1)} \implies b_n < \frac{n^3}{2(n-1)}.
    \]
    It is easy to verify that the largest possible \( b_n \) is \( \frac{n(n+1)}{2} \), as claimed.
    
    ### Claim 2: All \( r < \frac{1}{2} \) work.
    Consider the sequence \( a_n := \left\lceil kn^2 \right\rceil + n \) for \( k < \frac{1}{2} \). Since \( a_n \) is \( O(n^2) \), there exists \( N \) such that for all \( n \geq N \),
    \[
    \frac{a_n}{n^2} < \frac{1}{2}.
    \]
    Define the sequence \( b_n \) as follows:
    \[
    b_n := \begin{cases}
    \frac{n(n+1)}{2} & \text{for } n < N, \\
    a_n & \text{for } n \geq N.
    \end{cases}
    \]
    By definition of \( N \), \( \frac{b_n}{n^2} > \frac{b_{n+1}}{(n+1)^2} \) for \( n < N \). For \( n \geq N \), we want to show that
    \[
    \frac{\left\lceil kn^2 \right\rceil + n}{n^2} > \frac{\left\lceil k(n+1)^2 \right\rceil + n + 1}{(n+1)^2}.
    \]
    Since \( \left\lceil kn^2 \right\rceil \geq kn^2 \),
    \[
    \frac{\left\lceil kn^2 \right\rceil + n}{n^2} \geq k + \frac{1}{n},
    \]
    and since \( \left\lceil k(n+1)^2 \right\rceil < k(n+1)^2 + 1 \),
    \[
    \frac{\left\lceil k(n+1)^2 \right\rceil + n + 1}{(n+1)^2} < k + \frac{n+2}{(n+1)^2}.
    \]
    It suffices to verify that
    \[
    \frac{1}{n} \geq \frac{n+2}{(n+1)^2} \iff (n+1)^2 \geq n(n+2),
    \]
    which is true.
    
    Combining these two claims, we conclude that the possible values of \( r \) are:
    \[
    0 \leq r \leq \frac{1}{2}.
    \]
    
    The answer is: \boxed{0 \leq r \leq \frac{1}{2}}.""",
        "Сгенерируй 5 идей задач для курса по компьютерному зрению.",
    ]

    tokenizer = AutoTokenizer.from_pretrained(local_path)

    formatted_prompts = []
    for p in user_prompts:
        messages = [{"role": "user", "content": p}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,              # <-- именно строка для vLLM
            add_generation_prompt=True,
            enable_thinking=ENABLE_THINKING,
        )
        formatted_prompts.append(text)

    # ---- 2) DP генерация ----
    outputs = run_dp_generate(formatted_prompts, local_path)

    for i, out in enumerate(outputs):
        print("=" * 80)
        print(f"PROMPT #{i}: {user_prompts[i]}")
        print("-" * 80)
        print(out.strip())
