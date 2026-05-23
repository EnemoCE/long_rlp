"""Custom legacy FSDP worker wrapper for the RPT experiment path.

This module stays deliberately thin. It reuses the upstream async FSDP worker
for model initialization and transport, then swaps the actor implementation to
``DataParallelRPTActor`` and returns the fully updated training batch from
actor updates.

The worker is responsible for passing the runtime pieces that the actor cannot
discover by itself:

- tokenizer
- token ids for the answer delimiter
- ``actor_rollout_ref.rpt.ema_tau``
- ``actor_rollout_ref.rpt.length_penalty``
- ``actor_rollout_ref.rpt.drop_think_text``

Those values are what let the actor construct the thought mask, compute
ground-truth log-prob rewards, and update the EMA teacher after every
optimizer step.
"""

import os

import psutil
from codetiming import Timer

from verl import DataProto
from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.device import get_device_id, get_torch_device
from verl.utils.fsdp_utils import (
    load_fsdp_model_to_gpu,
    load_fsdp_optimizer,
    offload_fsdp_model_to_cpu,
    offload_fsdp_optimizer,
)
from verl.utils.profiler import DistProfiler, log_gpu_memory_usage
from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker, logger

from exps.rpt_actor import DataParallelRPTActor

logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class AsyncThoughtMaskActorRolloutRefWorker(AsyncActorRolloutRefWorker):
    """Legacy async actor/rollout/ref worker with a custom RPT actor.

    Its responsibilities differ from the stock worker in two places only:

    - ``init_model`` replaces the PPO actor with ``DataParallelRPTActor``
    - ``update_actor`` returns the updated ``DataProto`` batch, not just
      metrics, so trainer-side metric computation can still run after actor-side
      reward / advantage creation
    """

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        """Initialize the normal worker stack, then inject the RPT actor.

        After ``super().init_model()`` completes, the actor path is rewritten to
        use ``DataParallelRPTActor``. This is also where the custom ``rpt``
        config block is consumed and converted into the actor runtime state.

        The actor receives:

        - the tokenizer
        - answer delimiter token ids for thought masking
        - EMA decay ``ema_tau``
        - experimental ``length_penalty``
        - the empty-think template used for the EMA baseline prompt
        """
        super().init_model()

        if self._is_actor:
            actor_cfg = omega_conf_to_dataclass(self.config.actor)
            self.actor = DataParallelRPTActor(
                config=actor_cfg,
                actor_module=self.actor_module_fsdp,
                actor_optimizer=self.actor_optimizer,
            )

            rpt_cfg = self.config.get("rpt", {})
            reward_model_cfg = self.config.get("reward_model", {})
            reward_cfg = reward_model_cfg.get("overlong_buffer", {})
            reward_cfg = reward_model_cfg.get("reward_kwargs", {}).get("overlong_buffer_cfg", reward_cfg)
            ema_tau = float(rpt_cfg.get("ema_tau", 0.999))
            length_penalty = float(rpt_cfg.get("length_penalty", 0.0))
            answer_overlong_penalty_factor = float(rpt_cfg.get("answer_overlong_penalty_factor", 0.0))
            drop_think_text = str(rpt_cfg.get("drop_think_text", "<think>\n\n</think>\n\n####"))
            debug_train_generations = int(rpt_cfg.get("debug_train_generations", 0))
            debug_train_generation_batches = int(rpt_cfg.get("debug_train_generation_batches", 0))
            debug_train_generation_max_chars = int(rpt_cfg.get("debug_train_generation_max_chars", 1600))
            end_think_ids = self.tokenizer.encode("####", add_special_tokens=False)

            self.actor.setup_rpt(
                tokenizer=self.tokenizer,
                end_think_ids=end_think_ids,
                ema_tau=ema_tau,
                length_penalty=length_penalty,
                drop_think_text=drop_think_text,
                overlong_buffer_enable=bool(reward_cfg.get("enable", False)),
                overlong_buffer_len=int(reward_cfg.get("len", 0)),
                thought_overlong_penalty_factor=float(reward_cfg.get("penalty_factor", 0.0)),
                answer_overlong_penalty_factor=answer_overlong_penalty_factor,
                debug_train_generations=debug_train_generations,
                debug_train_generation_batches=debug_train_generation_batches,
                debug_train_generation_max_chars=debug_train_generation_max_chars,
            )

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="red", role="actor_update")
    def update_actor(self, data: DataProto):
        """Run the actor update and return the modified batch plus metrics.

        The important behavioral difference from upstream is the return value.
        ``DataParallelRPTActor.update_policy`` writes:

        - ``token_level_scores``
        - ``token_level_rewards``
        - ``advantages``
        - ``returns``

        into the batch during the first PPO epoch. Those tensors are preserved
        in the returned ``DataProto`` and metrics are attached to
        ``updated_batch.meta_info['metrics']`` so the trainer can continue using
        normal data-metric utilities after actor-side reward computation.
        """
        assert self._is_actor

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        if self._is_offload_optimizer:
            load_fsdp_optimizer(optimizer=self.actor_optimizer, device_id=get_device_id())

        with self.ulysses_sharding_manager:
            data = data.to("cpu")

            with Timer(name="update_policy", logger=None) as timer:
                updated_batch, metrics = self.actor.update_policy(data=data)

            delta_time = timer.last
            global_num_tokens = data.meta_info["global_token_num"]
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
            metrics["perf/mfu/actor"] = estimated_flops * self.config.actor.ppo_epochs / promised_flops / self.world_size
            metrics["perf/max_memory_allocated_gb"] = get_torch_device().max_memory_allocated() / (1024**3)
            metrics["perf/max_memory_reserved_gb"] = get_torch_device().max_memory_reserved() / (1024**3)
            metrics["perf/cpu_memory_used_gb"] = psutil.virtual_memory().used / (1024**3)

            lr = self.actor_lr_scheduler.get_last_lr()[0]
            metrics["actor/lr"] = lr.item() if hasattr(lr, "item") else lr
            self.actor_lr_scheduler.step()

            updated_batch.meta_info = dict(updated_batch.meta_info)
            updated_batch.meta_info["metrics"] = metrics
            output = updated_batch.to("cpu")

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            log_gpu_memory_usage("After offload actor model during update_actor", logger=logger)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.actor_optimizer)
            log_gpu_memory_usage("After offload actor optimizer during update_actor", logger=logger)

        return output
