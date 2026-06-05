#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

CONDA_DIR="${CONDA_DIR:-${HOME}/miniconda3}"
ENV_NAME="${ENV_NAME:-verl}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
GPU_STACK="${GPU_STACK:-auto}"
CUDA_VERSION_DASH="${CUDA_VERSION_DASH:-}"
CUDA_VERSION_DOT="${CUDA_VERSION_DOT:-}"
VLLM_VERSION="${VLLM_VERSION:-0.12.0}"
VLLM_CUDA_TAG="${VLLM_CUDA_TAG:-cu130}"
FLASH_ATTN_VERSION="${FLASH_ATTN_VERSION:-2.8.3}"
MAX_JOBS="${MAX_JOBS:-16}"

SUDO=()
if [[ "$(id -u)" != "0" ]]; then
  SUDO=(sudo)
fi

section() {
  echo
  echo "===== $* ====="
}

detect_gpu_stack() {
  local gpu_names
  gpu_names="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true)"
  if echo "${gpu_names}" | grep -Eiq "B200|B300|GB200|GB300|Blackwell"; then
    echo "blackwell"
  else
    echo "standard"
  fi
}

cuda_toolkit_satisfies() {
  local nvcc_bin="$1"
  local required_major="${CUDA_VERSION_DOT%%.*}"
  local required_minor="${CUDA_VERSION_DOT#*.}"
  local version major minor

  if [[ ! -x "${nvcc_bin}" ]]; then
    return 1
  fi

  version="$("${nvcc_bin}" --version 2>/dev/null | sed -nE 's/.*release ([0-9]+)\.([0-9]+).*/\1 \2/p' | head -n 1)"
  if [[ -z "${version}" ]]; then
    return 1
  fi

  read -r major minor <<<"${version}"
  (( major > required_major || (major == required_major && minor >= required_minor) ))
}

if [[ "${GPU_STACK}" == "auto" ]]; then
  GPU_STACK="$(detect_gpu_stack)"
fi

case "${GPU_STACK}" in
  blackwell)
    # B200/B300 need a Blackwell-capable PTXAS/Triton stack. CUDA 13.0
    # wheels avoid the old cu12/torch2.8 FlashAttention/vLLM path.
    CUDA_VERSION_DASH="${CUDA_VERSION_DASH:-13-0}"
    CUDA_VERSION_DOT="${CUDA_VERSION_DOT:-13.0}"
    ;;
  standard)
    CUDA_VERSION_DASH="${CUDA_VERSION_DASH:-12-8}"
    CUDA_VERSION_DOT="${CUDA_VERSION_DOT:-12.8}"
    ;;
  *)
    echo "Unsupported GPU_STACK=${GPU_STACK}. Use auto, blackwell, or standard." >&2
    exit 2
    ;;
esac

section "Selected GPU stack"
echo "GPU_STACK=${GPU_STACK}"
echo "CUDA_VERSION_DOT=${CUDA_VERSION_DOT}"
echo "CUDA_VERSION_DASH=${CUDA_VERSION_DASH}"

section "System packages"
"${SUDO[@]}" apt-get update
"${SUDO[@]}" apt-get install -y \
  git curl wget ca-certificates build-essential \
  cmake ninja-build pkg-config \
  libglib2.0-0 libsm6 libxext6 libxrender-dev

section "CUDA toolkit"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-${CUDA_VERSION_DOT}}"
if cuda_toolkit_satisfies "${CUDA_HOME}/bin/nvcc"; then
  "${CUDA_HOME}/bin/nvcc" --version
elif command -v nvcc >/dev/null 2>&1 && cuda_toolkit_satisfies "$(command -v nvcc)"; then
  CUDA_HOME="$(cd -- "$(dirname -- "$(command -v nvcc)")/.." && pwd)"
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

export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
nvcc --version
nvcc --list-gpu-code | grep -E "sm_100a|sm_103a|sm_103" || true

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
pip install -U pip setuptools wheel packaging uv

if [[ "${GPU_STACK}" == "blackwell" ]]; then
  section "Blackwell Python GPU stack"
  pip uninstall -y vllm sglang sglang-router flash-attn flashinfer-python torch torchvision torchaudio triton xformers || true
  pip install -r requirements.txt
  CPU_ARCH="$(uname -m)"
  uv pip install \
    "https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+${VLLM_CUDA_TAG}-cp38-abi3-manylinux_2_31_${CPU_ARCH}.whl" \
    --extra-index-url "https://download.pytorch.org/whl/${VLLM_CUDA_TAG}"
  MAX_JOBS="${MAX_JOBS}" pip install --no-cache-dir --no-build-isolation "flash-attn==${FLASH_ATTN_VERSION}"
else
  pip install -r requirements.txt
  USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
fi

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
    print("torch_arch_list", torch.cuda.get_arch_list())
PY
python - <<'PY'
import ray, transformers, trackio, vllm, flash_attn
print("ray", ray.__version__)
print("transformers", transformers.__version__)
print("trackio", trackio.__version__)
print("vllm", vllm.__version__)
print("flash_attn", flash_attn.__version__)
PY
python - <<'PY'
from pathlib import Path
import subprocess
import triton

ptxas = Path(triton.__file__).parent / "backends" / "nvidia" / "bin" / "ptxas"
print("triton", triton.__version__)
print("triton_ptxas", ptxas)
if ptxas.exists():
    subprocess.run([str(ptxas), "--version"], check=False)
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

GPU stack override:
  GPU_STACK=blackwell bash exps/scripts/setup_prime_env.sh  # B200/B300/GB200/GB300
  GPU_STACK=standard bash exps/scripts/setup_prime_env.sh   # A100/H100-style CUDA 12.8 path
EOF
