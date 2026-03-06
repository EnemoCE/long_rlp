def main():
    import os
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from huggingface_hub import snapshot_download
    
    # 4×A100 в одной ноде
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    print(5)
    MODEL_NAME = "Qwen/Qwen3-32B"
    TENSOR_PARALLEL = 4
    
    # Qwen3 умеет "thinking" (генерит блок <think>...</think>) и "non-thinking".
    # Для обычного прод-инференса часто удобнее выключить thinking (быстрее/чище ответы).
    ENABLE_THINKING = False  # True -> будет <think> блок
    
    # Список твоих промптов (обычные user-запросы)
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
    

    os.makedirs("./models/Qwen3-32B", exist_ok=True)
    local_path = snapshot_download(repo_id="Qwen/Qwen3-32B", local_dir="/models/Qwen3-32B", token="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")
    
    # 1) Собираем chat-подсказки через официальный chat template токенизатора Qwen3
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    formatted_prompts = []
    for p in user_prompts:
        messages = [{"role": "user", "content": p}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=ENABLE_THINKING,  # переключатель thinking/non-thinking
        )
        formatted_prompts.append(text)
    
    # 2) Параметры семплинга
    # В model card для thinking рекомендуют: temperature=0.6, top_p=0.95, top_k=20, min_p=0. :contentReference[oaicite:2]{index=2}
    # Для non-thinking часто берут чуть более "разговорные" параметры; ниже — безопасный дефолт.
    if ENABLE_THINKING:
        sampling_params = SamplingParams(
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            min_p=0.0,
            max_tokens=512,
        )
    else:
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            min_p=0.0,
            max_tokens=512,
        )
    
    # 3) Грузим vLLM и делаем generate на батче
    llm = LLM(
        model=MODEL_NAME,
        tensor_parallel_size=TENSOR_PARALLEL,
        dtype="bfloat16",
        gpu_memory_utilization=0.90,   # дефолт vLLM; можно поджать/поднять при нужде :contentReference[oaicite:3]{index=3}
        max_model_len=5000,           # нативный контекст Qwen3-32B :contentReference[oaicite:4]{index=4}
        enforce_eager=True,
    )
    
    outputs = llm.generate(formatted_prompts, sampling_params)
    
    # 4) Печатаем результаты
    for i, out in enumerate(outputs):
        gen = out.outputs[0].text
        print("=" * 80)
        print(f"PROMPT #{i}: {user_prompts[i]}")
        print("-" * 80)
        print(gen.strip())

if __name__ == "__main__":
    main()