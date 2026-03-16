#!/usr/bin/env bash
#SBATCH --job-name=rl-gspo-3B
#SBATCH --partition=main
#SBATCH --nodes=1                # Number of nodes
#SBATCH --ntasks-per-node=1      # One task per node
#SBATCH --cpus-per-task=128      # cpu-cores per task
#SBATCH --gres=gpu:8
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=500:00:00
#SBATCH --output=/rl/logs/Qwen2.5-3B/gspo/math/vllm_%x_%j.out
#SBATCH --error=/rl/logs/Qwen2.5-3B/gspo/math/vllm_%x_%j.err

set -xeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

# activate the venv
#echo "Activating verl environment..."
#eval "$(conda shell.bash hook)"
#conda deactivate
#conda activate verl

# can make training faster, depends on your infrastructure
#export NCCL_IBEXT_DISABLE=1
#export NCCL_NVLS_ENABLE=1
#export NCCL_IB_HCA=mlx5
#export UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1

# Set how many GPUs we actually have on this node.
#export GPUS_PER_NODE=8
export GPUS_PER_NODE=2

#NNODES=${SLURM_JOB_NUM_NODES}
export NNODES=1

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export RAY_memory_monitor_refresh_ms=0
export RAY_LOGGING_LEVEL=DEBUG
export HYDRA_FULL_ERROR=1
#export WANDB_API_KEY=... # your wandb API key

# Let Ray know how many nodes to expect
#export RAY_NUM_NODES=$NNODES

echo "Using $NNODES nodes for training..."

# ------------------------------------- Setup xp params ---------------------------------------
project_name='Long-RLP'

adv_estimator=rloo
reward_manager=dapo
loss_mode=gspo
loss_agg_mode="seq-mean-token-mean"
#MODEL_PATH=Qwen/Qwen2.5-0.5B-Instruct
MODEL_PATH=Qwen/Qwen3-1.7b
offload=false # it's a small model, offloading will just slow-down training
rollout_engine=vllm
rollout_mode=async
gpu_memory_utilization=0.8
shuffle_dataset=true
first_time_dataset_prep=false # prepare dataset

test_freq=10
save_freq=10
total_epochs=10
total_training_steps=5
val_before_train=false

use_kl_in_reward=false
kl_coef=0.0
use_kl_loss=false
kl_loss_coef=0.0

clip_ratio_low=0.0003 # as recommended by the paper, see Sec. 5.1
clip_ratio_high=0.0004 # as recommended by the paper, see Sec. 5.1

#estimated as a real limit for omni_math prompts - NEVER CHANGE
max_prompt_length=1024
#recommended completion length from RLP NVIDIA paper - don't change
max_response_length=2048
context_len=$((max_prompt_length + max_response_length))

#always 1
TP=1
#always 1
PP=1
#num gpus available
DP=2

#zero_stage=3 because fsdp is used by default as actor_rollout_ref.actor.fsdp_config.strategy: 'fsdp' and fsdp_config.reshard_after_forward=True
ZERO_STAGE="${ZERO_STAGE:-3}"
BATCH_ACCUM="${BATCH_ACCUM:-1}"

# Это выставит:
#   one_batch_static_memory_mb
#   one_batch_dynamic_memory_mb
#   peak_memory_tbi_mb
#   num_hidden_layers_in_pp
source <(
  python "${REPO_ROOT}/exps/scripts/calc_nanotron_memory.py" \
    --model_name "${MODEL_PATH}" \
    --seq_len "${context_len}" \
    --mbs 1 \
    --batch_accum "${BATCH_ACCUM}" \
    --tp "${TP}" \
    --pp "${PP}" \
    --dp "${DP}" \
    --zero_stage "${ZERO_STAGE}"
)

echo "one_batch_static_memory_mb=${one_batch_static_memory_mb}"
echo "one_batch_dynamic_memory_mb=${one_batch_dynamic_memory_mb}"
echo "peak_memory_tbi_mb=${peak_memory_tbi_mb}"

PER_RANK_MAX_MEMORY=80000
#manually check GPU available memory and update PER_RANK_MAX_MEMORY value accordingly
#memory gpu footprint above was calculated for ppo_micro_batch_size_per_gpu=1 with 
#context_len = max_prompt_length + max_response_length and model = MODEL_PATH and other params
#non-activations example: one_batch_static_memory_mb=29533
#activations memory example: one_batch_dynamic_memory_mb=10617
#9/10 factor to protect from OOM - 88.7% utilization for H200 141 Gb, model Qwen3-1.7B, context=3072
scale_factor=$(((PER_RANK_MAX_MEMORY*11/12-one_batch_static_memory_mb)/one_batch_dynamic_memory_mb))
ppo_micro_batch_size_per_gpu=$((scale_factor)) # setup depending on your GPU memory
#distribute n_rollouts across all available GPUs and cap it with optimal value 16 from the paper
n_resp_per_prompt=$(python -c "print(min($scale_factor * $DP,16))")
ppo_mini_batch_size=$((15*n_resp_per_prompt)) # divisible by rollout.n for stable in-actor RLOO grouping
train_batch_size=$((2*ppo_mini_batch_size)) #should be divisble by ppo_mini_batch_size

scale_factor=1
ppo_micro_batch_size_per_gpu=1
n_resp_per_prompt=2
ppo_mini_batch_size=2
train_batch_size=2


# dapo reward manager params
enable_overlong_buffer=false # true
overlong_buffer_len=$((512))
overlong_penalty_factor=0

# Paths and namings
exp_number=1
SFT_MODEL=$(basename $MODEL_PATH)
exp_name="${loss_mode}-epslow-${clip_ratio_low}-epshigh-${clip_ratio_high}-${SFT_MODEL}-RL-${exp_number}"
#CKPTS_DIR=/rl/checkpoints/experimental/4b/${loss_mode}/${exp_name}
DATA_DIR=${DATA_DIR:-"${REPO_ROOT}/exps/dataset/Omni-MATH-final"}
DATA_PREP_SCRIPT="${REPO_ROOT}/exps/data_preprocessing/prepare_omni_math_dataset.sh"
CKPTS_DIR=${CKPTS_DIR:-${REPO_ROOT}/ckpts/RL-Long-RLP_${exp_number}}
WORKING_DIR=${WORKING_DIR:-"${REPO_ROOT}"}
LOGS_DIR=${REPO_ROOT}/logs_${exp_number}
RUNTIME_ENV=${RUNTIME_ENV:-"${REPO_ROOT}/exps/config/rpt_runtime_env.yaml"}
mkdir -p "$DATA_DIR" "$CKPTS_DIR" "$LOGS_DIR"
KEEP_LAST=2
PERIOD_SEC=60

## -------- persistent logging (durable across reboots) --------
# keep Ray sessions in $HOME so they don't vanish on restart
export RAY_TMPDIR="${HOME}/ray_tmp"; mkdir -p "$RAY_TMPDIR"
export TMPDIR="$RAY_TMPDIR"
# mirror ALL stdout/stderr to terminal AND a timestamped file
exec > >(tee -a "${LOGS_DIR}/train_$(date +%F_%H-%M-%S).out") 2>&1
## -------------------------------------------------------------

: "${RAY_IP:=localhost}"
export RAY_ADDRESS=${RAY_ADDRESS:-"http://${RAY_IP}:8265"}
export VERL_FILE_LOGGER_ROOT=${VERL_FILE_LOGGER_ROOT:-${LOGS_DIR}}
export GRADIO_SERVER_PORT="${GRADIO_SERVER_PORT:-7871}"
SENTINEL="$CKPTS_DIR/.cleaner.run"

# robust, nounset-proof cleaner
clean_loop() (
  set +u                    # isolate from global 'set -u' in a subshell
  shopt -s nullglob         # empty globs => empty array, not literal
  local -a cps=()
  local n=0
  local d

  while [[ -f "$SENTINEL" ]]; do
    # 1) Find checkpoints and sort them
    cps=( "$CKPTS_DIR"/global_step_* )
    if ((${#cps[@]})); then
      IFS=$'\n' cps=( $(printf '%s\n' "${cps[@]}" | sort -V) )
    fi
    n=${#cps[@]}

    # 2) Remove everything except the newest KEEP_LAST
    if (( n > KEEP_LAST )); then
      for d in "${cps[@]:0:n-KEEP_LAST}"; do
        rm -rf -- "$d"
      done
    fi

    # 3) Re-scan survivors and strip huggingface dirs in all but newest
    cps=( "$CKPTS_DIR"/global_step_* )
    if ((${#cps[@]})); then
      IFS=$'\n' cps=( $(printf '%s\n' "${cps[@]}" | sort -V) )
    fi
    n=${#cps[@]}
    if (( n > 1 )); then
      for d in "${cps[@]:0:n-1}"; do
        rm -rf -- "$d/actor/huggingface"
      done
    fi

    sleep "$PERIOD_SEC"
  done
)





# Sampling params at rollouts
temperature=0.7
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout
val_top_p=0.7

# Performance Related Parameter
sp_size=1
use_dynamic_bsz=false
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 2*5))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 3*5))
#offload=true
gen_tp=1
entropy_checkpointing=true # This enables entropy recomputation specifically for the entropy calculation, lowering memory usage during training.
async_agent_workers=1 # single rollout replica on this VM; avoid mismatched async chunking defaults

# ------------------------------------- train/val data preparation ---------------------------------------
#first_time_dataset_prep=false
if [ "$first_time_dataset_prep" = true ]; then
    echo "Preprocessing Omni-MATH dataset..."
    bash "${DATA_PREP_SCRIPT}"
fi

python "${REPO_ROOT}/exps/scripts/jinja_template_in_config.py" \
    --yaml "${REPO_ROOT}/exps/config/rpt_trainer.yaml" \
    --jinja "${REPO_ROOT}/exps/jinja/rpt_template.jinja"


omni_math_train_path="${DATA_DIR}/train.parquet"
omni_math_test_path="${DATA_DIR}/test.parquet"

#TEMPLATE_PATH="${HOME}/verl/exps/custom_template.jinja"

#TEMPLATE="$(cat "$TEMPLATE_PATH")"
#TEMPLATE="$(<"$TEMPLATE_PATH")"



# set the paths
train_files="['$omni_math_train_path']"
test_files="['$omni_math_test_path']"

: > "$SENTINEL"                 # 1) create sentinel
clean_loop & CLEAN_PID=$!    # 2) launch cleaner in background
trap 'rm -f "$SENTINEL"; kill "$CLEAN_PID" 2>/dev/null || true' EXIT INT TERM

JOB_ID="train-$(date +%s)-$RANDOM"

ray job submit --no-wait --submission-id "$JOB_ID" --runtime-env="${RUNTIME_ENV}" \
    --working-dir "${WORKING_DIR}" \
    -- python3 -m exps.main_rpt \
    --config-path "${REPO_ROOT}/exps/config" \
    --config-name "rpt_trainer" \
    algorithm.adv_estimator=${adv_estimator} \
    actor_rollout_ref.actor.policy_loss.loss_mode=${loss_mode} \
    data.train_files="${train_files}" \
    data.val_files="${test_files}" \
    data.shuffle=$shuffle_dataset \
    data.prompt_key=prompt \
    data.truncation='error' \
    data.filter_overlong_prompts=true \
    data.train_batch_size=${train_batch_size} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.name=${rollout_engine} \
    actor_rollout_ref.rollout.mode=${rollout_mode} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.05 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_memory_utilization} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=true \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.agent.num_workers=${async_agent_workers} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=true \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.entropy_checkpointing=${entropy_checkpointing} \
    reward_model.reward_manager=${reward_manager} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=false \
    +reward_model.reward_kwargs.max_resp_len=${max_response_length} \
    trainer.logger='["console", "trackio", "file"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node="${GPUS_PER_NODE}" \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=${val_before_train} \
    trainer.balance_batch=false \
    trainer.test_freq=${test_freq} \
    trainer.save_freq=${save_freq} \
    trainer.total_epochs=${total_epochs} \
    trainer.total_training_steps=${total_training_steps} \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode="disable" \
    trainer.log_val_generations=2 \
    actor_rollout_ref.rpt.ema_tau=0.999 \
    actor_rollout_ref.rpt.length_penalty=0.0 \
    $@

# blocks until the job finishes
ray job logs -f "$JOB_ID" || true

rm -f "$SENTINEL"               # cleaner stops on next tick

kill "$CLEAN_PID" 2>/dev/null || true
wait "$CLEAN_PID" 2>/dev/null || true
trap - EXIT INT TERM
