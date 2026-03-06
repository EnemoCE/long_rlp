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
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        super().init_model()

        if self._is_actor:
            actor_cfg = omega_conf_to_dataclass(self.config.actor)
            self.actor = DataParallelRPTActor(
                config=actor_cfg,
                actor_module=self.actor_module_fsdp,
                actor_optimizer=self.actor_optimizer,
            )

            rpt_cfg = self.config.get("rpt", {})
            ema_tau = float(rpt_cfg.get("ema_tau", 0.999))
            length_penalty = float(rpt_cfg.get("length_penalty", 0.0))
            drop_think_text = str(rpt_cfg.get("drop_think_text", "<think>\n\n</think>\n\n"))
            end_think_ids = self.tokenizer.encode("</think>", add_special_tokens=False)

            self.actor.setup_rpt(
                tokenizer=self.tokenizer,
                end_think_ids=end_think_ids,
                ema_tau=ema_tau,
                length_penalty=length_penalty,
                drop_think_text=drop_think_text,
            )

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="red", role="actor_update")
    def update_actor(self, data: DataProto):
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
