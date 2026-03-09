#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
DATA_DIR="${SCRIPT_DIR}/dataset/Omni-MATH"
INTERMEDIATE_DIR="${SCRIPT_DIR}/dataset/Omni-MATH_final"
JSONL_DIR="${SCRIPT_DIR}/Omni-MATH"

#MODEL_PATH=Qwen/Qwen2.5-0.5B-Instruct
MODEL_PATH=Qwen/Qwen3-1.7b

mkdir -p "${DATA_DIR}" "${INTERMEDIATE_DIR}" "${JSONL_DIR}"

export HF_HOME="${HF_HOME:-${REPO_ROOT}/.hf-cache}"

cd "${SCRIPT_DIR}"

echo "Running spaCy sentence splitting for Omni-MATH..."
python spacy_parse_solutions.py

echo "Building Omni-MATH train/test parquet files..."
python omni_math_processing.py --target full_sentence --model_name "${MODEL_PATH}" --out_dir dataset/Omni-MATH
