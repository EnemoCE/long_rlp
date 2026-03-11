# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Custom exps-only RPT entry point analogous to ``verl.trainer.main_ppo``.

This module is the top of the runtime path launched by ``test_long_rlp.sh`` after
the shell layer has optionally prepared Omni-MATH data and updated
``config/rpt_trainer.yaml`` with ``scripts/jinja_template_in_config.py``.

The key differences from stock PPO wiring are:

- the trainer class is ``exps.rpt_ray_trainer.RayRPTTrainer``
- the FSDP actor/rollout worker is ``exps.rpt_workers.AsyncThoughtMaskActorRolloutRefWorker``
- training reward is no longer produced in the trainer reward block
- validation still keeps the standard reward-manager path for compatibility

The config dependency that matters most here is
``trainer.use_legacy_worker_impl=enable`` together with the custom
``actor_rollout_ref.rpt`` block. Those values are consumed downstream by the
custom worker and actor so the RPT reward path can run entirely inside the
actor update.

Only syntax-level validation had been performed locally when this flow was
first assembled. Real Ray execution, GPU memory fit, and end-to-end reward
correctness still need to be checked on the remote VM.
"""

import os
import socket

import hydra
import ray
from omegaconf import OmegaConf

from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.trainer.ppo.reward import load_reward_manager
from verl.utils.device import auto_set_ascend_device_name, is_cuda_available

from exps.rpt_ray_trainer import RayRPTTrainer


@hydra.main(config_path="config", config_name="rpt_trainer", version_base=None)
def main(config):
    """Hydra entry point that normalizes device config and delegates to ``run_ppo``."""
    # Automatically set `config.trainer.device = npu` when running on Ascend NPU.
    auto_set_ascend_device_name(config)

    run_ppo(config)


def run_ppo(config) -> None:
    """Initialize Ray, choose the remote runner, and execute the custom trainer.

    ``run_ppo`` keeps the standard Ray bootstrap shape from the upstream entry
    point, but the actual training path is custom:

    1. initialize Ray with the PPO runtime environment
    2. dispatch ``TaskRunner.run`` remotely
    3. let the task runner instantiate the tokenizer, worker mapping, reward
       managers, and ``RayRPTTrainer``

    Reward managers are still created here even though training reward is
    actor-side. That split is intentional: validation remains on the standard
    reward-manager machinery while training uses the custom RPT actor update.
    """
    if not ray.is_initialized():
        # this is for local ray cluster
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    try:
        if (
            is_cuda_available
            and config.global_profiler.tool == "nsys"
            and OmegaConf.select(config.global_profiler, "steps") is not None
            and len(OmegaConf.select(config.global_profiler, "steps")) > 0
        ):
            nsight_options = OmegaConf.to_container(
                config.global_profiler.global_tool_config.nsys.controller_nsight_options
            )
            runner = TaskRunner.options(runtime_env={"nsight": nsight_options}).remote()
        else:
            runner = TaskRunner.remote()
        ray.get(runner.run.remote(config))
    finally:
        if ray.is_initialized():
            ray.shutdown()


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
class TaskRunner:
    """Remote bootstrapper that constructs workers and launches ``RayRPTTrainer``.

    This class is where the shell-script level choices become concrete Python
    objects: tokenizer / processor loading, worker-class selection, resource
    pool wiring, and reward-manager construction.

    For the FSDP path the important behavior is that ActorRollout and optional
    RefPolicy roles use ``AsyncThoughtMaskActorRolloutRefWorker`` so the stock
    PPO actor gets replaced by ``DataParallelRPTActor`` inside worker
    initialization.
    """

    def run(self, config):
        """Build the training stack and start fitting.

        The effective pipeline is:

        1. print and resolve config
        2. copy the model checkpoint locally
        3. instantiate tokenizer and optional processor
        4. map Ray roles to custom workers
        5. keep reward-function objects for validation and compatibility
        6. initialize workers and call ``trainer.fit()``

        The training reward path is not routed through ``reward_fn``. The
        actual RPT reward is computed later inside
        ``DataParallelRPTActor.update_policy`` after rollout generation and
        old/ref log-prob preprocessing have already happened.
        """
        # print initial config
        from pprint import pprint

        from omegaconf import OmegaConf

        from verl.utils.fs import copy_to_local

        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")

        pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
        OmegaConf.resolve(config)

        # download the checkpoint from hdfs
        local_path = copy_to_local(config.actor_rollout_ref.model.path)

        # instantiate tokenizer
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        # used for multimodal LLM, could be none
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        from verl.single_controller.ray import RayWorkerGroup

        # define worker classes
        if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}:
            assert config.critic.strategy in {"fsdp", "fsdp2"}

            from exps.rpt_workers import AsyncThoughtMaskActorRolloutRefWorker
            from verl.workers.fsdp_workers import CriticWorker

            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.workers.megatron_workers import AsyncActorRolloutRefWorker, CriticWorker

            ray_worker_group_cls = RayWorkerGroup

        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(AsyncThoughtMaskActorRolloutRefWorker),
            Role.Critic: ray.remote(CriticWorker),
        }

        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }

        # we should adopt a multi-source reward function here
        # - for rule-based rm, we directly call a reward score
        # - for model-based rm, we call a model
        # - for code related prompt, we send to a sandbox if there are test cases
        # - finally, we combine all the rewards together
        # - The reward type depends on the tag of the data
        if config.reward_model.enable:
            if config.reward_model.strategy in {"fsdp", "fsdp2"}:
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        # reference model
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(AsyncThoughtMaskActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        reward_fn = load_reward_manager(
            config,
            tokenizer,
            0,
            max_resp_len=config.data.max_response_length,
            overlong_buffer_cfg=config.reward_model.overlong_buffer,
        )

        # Note that we always use function-based RM for validation
        val_reward_fn = load_reward_manager(
            config,
            tokenizer,
            1,
            max_resp_len=config.data.max_response_length,
            overlong_buffer_cfg=config.reward_model.overlong_buffer,
        )
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        trainer = RayRPTTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
        )
        trainer.init_workers()
        trainer.fit()


if __name__ == "__main__":
    main()
