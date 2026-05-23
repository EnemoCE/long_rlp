#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

CONDA_DIR="${CONDA_DIR:-${HOME}/miniconda3}"
ENV_NAME="${ENV_NAME:-verl}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
CUDA_VERSION_DASH="${CUDA_VERSION_DASH:-12-8}"
CUDA_VERSION_DOT="${CUDA_VERSION_DOT:-12.8}"

SUDO=()
if [[ "$(id -u)" != "0" ]]; then
  SUDO=(sudo)
fi

section() {
  echo
  echo "===== $* ====="
}

section "System packages"
"${SUDO[@]}" apt-get update
"${SUDO[@]}" apt-get install -y \
  git curl wget ca-certificates build-essential \
  libglib2.0-0 libsm6 libxext6 libxrender-dev

section "CUDA toolkit"
if command -v nvcc >/dev/null 2>&1; then
  nvcc --version
else
  . /etc/os-release
  case "${ID}:${VERSION_ID}" in
    ubuntu:22.04) CUDA_REPO_OS="ubuntu2204" ;;
    ubuntu:24.04) CUDA_REPO_OS="ubuntu2404" ;;
    *)
      echo "Unsupported CUDA toolkit bootstrap target: ${ID} ${VERSION_ID}" >&2
      echo "Install CUDA toolkit ${CUDA_VERSION_DOT} manually, then rerun this script." >&2
      exit 3
      ;;
  esac

  ARCH="$(dpkg --print-architecture)"
  CUDA_KEYRING="/tmp/cuda-keyring_1.1-1_all.deb"
  wget -qO "$CUDA_KEYRING" \
    "https://developer.download.nvidia.com/compute/cuda/repos/${CUDA_REPO_OS}/${ARCH}/cuda-keyring_1.1-1_all.deb"
  "${SUDO[@]}" dpkg -i "$CUDA_KEYRING"
  rm -f "$CUDA_KEYRING"
  "${SUDO[@]}" apt-get update
  "${SUDO[@]}" apt-get install -y "cuda-toolkit-${CUDA_VERSION_DASH}"
fi

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-${CUDA_VERSION_DOT}}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

section "Miniconda"
if [[ ! -x "${CONDA_DIR}/bin/conda" ]]; then
  wget -qO /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash /tmp/miniconda.sh -b -p "${CONDA_DIR}"
  rm /tmp/miniconda.sh
fi

source "${CONDA_DIR}/etc/profile.d/conda.sh"
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true

section "Conda environment"
if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
fi
conda activate "${ENV_NAME}"

section "Python dependencies"
cd "${REPO_ROOT}"
pip install -r requirements.txt
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
pip install "numpy<2.3" "huggingface-hub>=0.34.0,<1.0" spacy ruamel.yaml trackio
python -m spacy download en_core_web_sm
mv "${REPO_ROOT}"/flash_attn-*.whl /tmp/ 2>/dev/null || true

section "Runtime check"
export PYTHONPATH="${REPO_ROOT}"
export DATASET_ROOT="${DATASET_ROOT:-/opt/datasets}"
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu0", torch.cuda.get_device_name(0))
PY
python - <<'PY'
import ray, transformers, trackio
print("ray", ray.__version__)
print("transformers", transformers.__version__)
print("trackio", trackio.__version__)
PY

cat <<EOF

Environment ready.

Use:
  source "${CONDA_DIR}/etc/profile.d/conda.sh"
  conda activate "${ENV_NAME}"
  cd "${REPO_ROOT}"
  export PYTHONPATH="${REPO_ROOT}"
  export DATASET_ROOT="\${DATASET_ROOT:-/opt/datasets}"
  bash exps/test_long_rlp.sh
EOF
