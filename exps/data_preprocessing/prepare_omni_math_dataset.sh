#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
DATASET_ROOT="${DATASET_ROOT:-/opt/datasets}"
DATA_DIR="${DATA_DIR:-${DATASET_ROOT}/Omni-MATH-final}"
INTERMEDIATE_DIR="${INTERMEDIATE_DIR:-${DATASET_ROOT}/Omni-MATH-sentence-splitted}"
JSONL_DIR="${JSONL_DIR:-${DATASET_ROOT}/Omni-MATH-final-json}"

#MODEL_PATH=Qwen/Qwen2.5-0.5B-Instruct
MODEL_PATH=Qwen/Qwen3-1.7B-Base

mkdir -p "${DATA_DIR}" "${INTERMEDIATE_DIR}" "${JSONL_DIR}"

export HF_HOME="${HF_HOME:-${REPO_ROOT}/.hf-cache}"

cd "${SCRIPT_DIR}"

echo "Running spaCy sentence splitting for Omni-MATH..."
python spacy_parse_solutions.py --OUT_DIR "$INTERMEDIATE_DIR"

echo "Building Omni-MATH train/test parquet files..."
python omni_math_processing.py --target full_sentence --model_name "${MODEL_PATH}" --input_dir "$INTERMEDIATE_DIR" --out_dir "$DATA_DIR" --jsonl_dir "$JSONL_DIR"
