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
"""Custom RPT trainer layered on top of the upstream Ray PPO trainer.

This trainer keeps the normal rollout-generation and logging pipeline, but it
changes where reward and advantage are computed:

- trainer-side reward computation is bypassed for training
- trainer-side advantage computation is bypassed for training
- old/ref log-prob bookkeeping remains trainer-side
- the actor update returns a batch that already contains rewards and advantages

The key grouping assumption is that repeated rollouts for the same prompt stay
consecutive. That is why the launch defaults disable dynamic batch balancing
and use a PPO mini-batch size divisible by the rollout count. Breaking that
assumption can silently corrupt both RLOO grouping and EMA deduplication.
"""

import os
import uuid
from collections import defaultdict
from copy import deepcopy
from pprint import pprint

import numpy as np
import torch
from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import compute_data_metrics, compute_throughout_metrics, compute_timing_metrics
from verl.trainer.ppo.ray_trainer import (
    AdvantageEstimator,
    RayPPOTrainer,
)
from verl.trainer.ppo.reward import compute_reward
from verl.utils.metric import reduce_metrics
from verl.utils.profiler import marked_timer
from verl.utils.rollout_skip import RolloutSkip






class RayRPTTrainer(RayPPOTrainer):
    """
    Driver-side trainer that delegates the RPT objective to the actor worker.

    The trainer still owns rollout generation, UID assignment, old/ref log-prob
    computation, optional critic values, checkpointing, validation, and metric
    aggregation. What moved out of the trainer is the training reward path:
    reward and advantages are produced inside ``DataParallelRPTActor`` instead
    of the standard trainer reward block.
    """

    #diff
    def compute_rpt_response_mask(self, data: DataProto, think: bool = True):
        """Return either a full-response or thought-only response mask.

        When ``think=True`` this helper truncates the response at the first
        ``</think>`` token span. When ``think=False`` it simply returns the full
        response-valid portion of ``attention_mask``.

        The current trainer deliberately calls this with ``think=False`` inside
        ``compute_kl_related_metrics`` so old/ref log-probs are computed over the
        whole generated response, not just the thought region. That behavior is
        intentionally different from the actor-side reward mask.
        """
        responses = data.batch["responses"]
        end_think_ids = self.actor_rollout_wg.tokenizer.encode("</think>", add_special_tokens=False)
        device = responses.device
        needle = torch.tensor(end_think_ids, device=device, dtype=responses.dtype)
        tel = len(end_think_ids)
        attention_mask = data.batch["attention_mask"]
        response_length = responses.size(1)
        response_mask = attention_mask[:, -response_length:]
        if think:
            for i in range(responses.size(0)):
                found = False
                for j in range(len(responses[i]) - tel + 1):
                    if (responses[i, j:j+tel] == needle).all():
                        found = True
                        break
                if found:
                    response_mask[i, j+tel:] = 0
                    found = False
        return response_mask


    def compute_kl_related_metrics(self, batch: DataProto, metrics: dict, timing_raw: dict):
        """Compute trainer-side old/ref log-prob bookkeeping over full responses.

        This step remains trainer-side because PPO still needs old log-probs,
        entropy metrics, and optional reference log-probs for the policy update.
        The response mask here is built with ``think=False`` on purpose.
        """
        #diff
        batch.batch["response_mask"] = self.compute_rpt_response_mask(batch, think=False)

        # recompute old_log_probs
        with marked_timer("old_log_prob", timing_raw, "blue"):
            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
            entropys = old_log_prob.batch["entropys"]
            response_masks = batch.batch["response_mask"]
            loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
            entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
            old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
            metrics.update(old_log_prob_metrics)
            old_log_prob.batch.pop("entropys")
            batch = batch.union(old_log_prob)

        if self.use_reference_policy:
            # compute reference log_prob
            with marked_timer("ref", timing_raw, "olive"):
                if not self.ref_in_actor:
                    ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                else:
                    ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                batch = batch.union(ref_log_prob)

        return batch

    def _update_actor(self, batch: DataProto):
        """Attach actor-side metadata and request the custom worker update.

        The metadata added here is what lets ``DataParallelRPTActor`` compute
        advantages consistently with the rollout configuration:

        - ``temperature``
        - ``adv_estimator``
        - ``gamma``
        - ``lam``
        - ``num_repeat``
        - ``norm_adv_by_std_in_grpo``

        The worker returns the updated batch itself. This wrapper keeps API
        compatibility by also producing a metrics-only ``DataProto``.
        """
        if self.use_legacy_worker_impl == "disable":
            raise NotImplementedError("RPT exps path supports only legacy FSDP worker implementation")

        rollout_config = self.config.actor_rollout_ref.rollout
        batch.meta_info["multi_turn"] = rollout_config.multi_turn.enable
        batch.meta_info["temperature"] = rollout_config.temperature
        batch.meta_info["adv_estimator"] = str(self.config.algorithm.adv_estimator)
        batch.meta_info["gamma"] = self.config.algorithm.gamma
        batch.meta_info["lam"] = self.config.algorithm.lam
        batch.meta_info["num_repeat"] = self.config.actor_rollout_ref.rollout.n
        batch.meta_info["norm_adv_by_std_in_grpo"] = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)

        worker_output = self.actor_rollout_wg.update_actor(batch)
        metrics_only = DataProto.from_single_dict(data={}, meta_info={"metrics": worker_output.meta_info["metrics"]})
        return worker_output, metrics_only

    def fit(self):
        """
        Run the PPO training loop with actor-side RPT reward computation.

        The flow differs from stock PPO at the update stage but otherwise stays
        familiar:

        1. load checkpoints and optionally validate
        2. build ``new_batch`` from the dataloader output
        3. generate repeated rollouts
        4. assign one UUID per original prompt before repetition
        5. repeat the prompt batch so repeated trajectories share the same UID
        6. merge rollout outputs into the repeated batch
        7. compute trainer-side old/ref log-probs over the full response
        8. call ``_update_actor`` so the actor computes rewards, advantages,
           PPO gradients, and EMA updates
        9. compute the usual data, timing, and throughput metrics

        Two assumptions are especially important for debugging:

        - repeated UID groups should remain consecutive and intact
        - ``trainer.balance_batch`` is normally kept disabled because reordering
          can break the actor-side grouping assumptions
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self.gen_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        self.gen_steps += 1
        last_val_metrics = None

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        timing_raw = defaultdict(float)
        batch = None
        num_prompt_in_batch = 0
        num_gen_batches = 0
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)
                metrics = {}
                reward_extra_infos_dict = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )

                new_batch: DataProto = DataProto.from_single_dict(batch_dict)

                num_gen_batches += 1
                gen_batch = self._get_gen_batch(new_batch)
                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )

                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, "red"):
                        gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with marked_timer("gen_max", timing_raw, "red"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)

                            new_batch = new_batch.union(gen_baseline_output)
                            # compute reward model score on new_batch
                            rm_scores = None
                            if self.use_rm and "rm_scores" not in new_batch.batch.keys():
                                rm_scores = self.rm_wg.compute_rm_score(new_batch)
                                new_batch = new_batch.union(rm_scores)
                            reward_baseline_tensor, _ = compute_reward(new_batch, self.reward_fn)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            keys_to_pop = set(gen_baseline_output.batch.keys())
                            if rm_scores is not None:
                                keys_to_pop.update(rm_scores.batch.keys())
                            new_batch.pop(batch_keys=list(keys_to_pop))

                            new_batch.batch["reward_baselines"] = reward_baseline_tensor

                            del rm_scores, gen_baseline_batch, gen_baseline_output

                    new_batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
                    )
                    # repeat to align with repeated responses in rollout
                    new_batch = new_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    new_batch = new_batch.union(gen_batch_output)
                    batch = new_batch

                    if self.config.algorithm.use_kl_in_reward:
                        # We need these metrics for apply_kl_penalty if using kl in reward
                        new_batch = self.compute_kl_related_metrics(new_batch, metrics, timing_raw)
                        # otherwise, we will compute those after dynamic sampling
                    

                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    if not self.config.algorithm.use_kl_in_reward:
                        batch = self.compute_kl_related_metrics(batch, metrics, timing_raw)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, "cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    # Compute rollout correction weights and off-policy metrics (inherited from RayPPOTrainer)
                    from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    if rollout_corr_config is not None and "rollout_log_probs" in batch.batch:
                        batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                        # IS and off-policy metrics already have rollout_corr/ prefix
                        metrics.update(is_metrics)

                    # update critic
                    # if self.use_critic:
                    #     with marked_timer("update_critic", timing_raw, "pink"):
                    #         critic_output = self.critic_wg.update_critic(batch)
                    #     critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                    #     metrics.update(critic_output_metrics)
                    
                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, "red"):
                            #actor_output = self.actor_rollout_wg.update_actor(batch)
                            batch, actor_output = self._update_actor(batch) #new
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, "green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                ):
                    with marked_timer("save_checkpoint", timing_raw, "green"):
                        self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                timing_raw = defaultdict(float)  # clear timing

                metrics["train/num_gen_batches"] = num_gen_batches
                batch = None
                num_prompt_in_batch = 0
                num_gen_batches = 0

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                if is_last_step:
                    if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                        self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                progress_bar.update(1)
                self.global_steps += 1
                self.gen_steps += 1
        # check if last step checkpint exists
        checkpoint_dir = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")
        if not os.path.exists(checkpoint_dir):
            # save last step checkpoint
            timing_raw = defaultdict(float)
            with marked_timer("save_checkpoint", timing_raw, "green"):
                self._save_checkpoint()
            metrics = {f"timing/{k}": v for k, v in timing_raw.items()}
            logger.log(data=metrics, step=self.global_steps)
