# split_solutions_spacy.py
import os
import json
from typing import List, Tuple

import spacy
from datasets import load_dataset

version = 6
DATASET_NAME = os.environ.get("DATASET_NAME", "KbsdJames/Omni-MATH")
DATASET_SPLIT = os.environ.get("DATASET_SPLIT", "test")  # поменяй если нужно
TAKE_N = int(os.environ.get("TAKE_N", "100"))

OUT_DIR = OUT_DIR = f"./dataset/Omni-MATH_final"
PARQUET_PATH = os.path.join(OUT_DIR, "dataset_with_solution_sentences.parquet")
JSONL_PATH = os.path.join(OUT_DIR, "dataset_with_solution_sentences.jsonl")
BAD_PATH = os.path.join(OUT_DIR, "bad_indices.txt")

SPACY_MODEL = os.environ.get("SPACY_MODEL", "en_core_web_sm")


def load_spacy_model():
    """
    Требует заранее установленный model package.
    Если не установлен: python -m spacy download en_core_web_sm
    """
    try:
        # Оставим parser, т.к. он обычно лучше режет предложения.
        # Отключим NER (быстрее и он не нужен).
        nlp = spacy.load(SPACY_MODEL, disable=["ner"])
        return nlp
    except Exception as e:
        raise RuntimeError(
            f"Cannot load spaCy model '{SPACY_MODEL}'.\n"
            f"Install it with:\n"
            f"  python -m spacy download {SPACY_MODEL}\n"
            f"Original error: {e}"
        )


def split_sentences_preserve_exact(text: str, nlp) -> Tuple[List[str], bool]:
    """
    Возвращает список сегментов — точные срезы исходного текста.
    Ключ: включаем пробелы/переносы ПОСЛЕ последнего токена каждого предложения
    через token.whitespace_, чтобы ''.join(segs) == text.
    """
    if not text:
        return [text], True

    doc = nlp(text)

    sents = list(doc.sents)
    if not sents:
        return [text], True

    segs: List[str] = []
    for sent in sents:
        # start_char/end_char у span не включает trailing whitespace между предложениями,
        # поэтому берём end с учётом whitespace_ последнего токена.
        start = sent[0].idx
        last = sent[-1]
        end = last.idx + len(last) + len(last.whitespace_)
        segs.append(text[start:end])

    ok = ("".join(segs) == text)
    return segs, ok


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    nlp = load_spacy_model()

    ds = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
    #ds = ds.select(range(min(TAKE_N, len(ds))))

    solutions = ds["solution"]

    all_segments: List[List[str]] = []
    bad_indices: List[int] = []

    # быстрее на больших объёмах: nlp.pipe
    # но нам нужно гарантировать точность — это сохраняется
    for i, sol in enumerate(solutions):
        segs, ok = split_sentences_preserve_exact(sol, nlp)
        if not ok:
            bad_indices.append(i)
            all_segments.append([])  # как ты просила: молча не добавляем сегменты
        else:
            all_segments.append(segs)

    out_ds = ds.add_column("solution_sentences", all_segments)

    out_ds.to_parquet(PARQUET_PATH)

    with open(JSONL_PATH, "w", encoding="utf-8") as f:
        for ex in out_ds:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with open(BAD_PATH, "w", encoding="utf-8") as f:
        for idx in bad_indices:
            f.write(str(idx) + "\n")


if __name__ == "__main__":
    main()
