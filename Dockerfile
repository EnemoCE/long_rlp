FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR=/opt/conda
ENV PATH=${CONDA_DIR}/bin:${PATH}

RUN apt-get update && apt-get install -y \
    git curl wget ca-certificates build-essential \
    libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

RUN wget -qO /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash /tmp/miniconda.sh -b -p ${CONDA_DIR} && \
    rm /tmp/miniconda.sh

WORKDIR /verl

COPY . /verl
RUN mkdir -p /opt/datasets

RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda create -y -n verl python=3.12 && \
    conda run -n verl pip install -r requirements.txt && \
    conda run -n verl bash -lc 'USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh' && \
    conda run -n verl pip install "numpy<2.3" spacy ruamel.yaml trackio && \
    conda run -n verl python -m spacy download en_core_web_sm

RUN mv /verl/flash_attn-*.whl /tmp/ 2>/dev/null || true

ENV CONDA_DEFAULT_ENV=verl
ENV PATH=${CONDA_DIR}/envs/verl/bin:${CONDA_DIR}/bin:${PATH}
ENV PYTHONPATH=/verl
ENV DATASET_ROOT=/opt/datasets

WORKDIR /verl
CMD ["/bin/bash"]
