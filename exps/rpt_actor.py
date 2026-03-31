"""Custom PPO actor implementing the RPT reward path inside actor updates.

This file contains the main deviation from stock PPO. Reward and advantage are
not produced in the trainer before ``update_actor``. Instead, the actor:

1. builds a thought-only mask ending at the first ``</think>``
2. computes ``logp_actor(ground_truth | prompt + generated_thought)``
3. computes ``logp_ema(ground_truth | prompt + empty_think)``
4. places the scalar reward on the last thought token
5. computes grouped advantages inside the actor update
6. performs PPO optimization
7. updates the EMA teacher immediately after every optimizer step

This design exists because the EMA baseline should track the current actor at
mini-batch granularity. If reward were computed once in the trainer before the
policy update, the EMA teacher would be stale for later mini-batches.

There is also an intentional mask split in this project:

- trainer-side old/ref log-prob bookkeeping uses the full generated response
- actor-side RPT reward uses only the thought region through ``</think>``
"""

import contextlib
import copy
import logging
import os
from typing import Tuple

import numpy as np
import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.trainer.ppo.ray_trainer import AdvantageEstimator, compute_advantage
from verl.utils.device import get_device_id
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch
from verl.workers.actor.dp_actor import DataParallelPPOActor

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def make_thought_mask(
    responses: torch.Tensor,
    end_think_ids: list[int],
    response_attn: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return a thought-only response mask ending at the first ``</think>``.

    The mask is ``1`` from the start of the generated response through the
    closing ``</think>`` token span and ``0`` after that. If the delimiter is
    not found, the mask falls back to the valid response length from
    ``response_attn`` instead of zeroing the sample out.

    This mask is used only for the RPT reward path. Trainer-side KL / old/ref
    log-prob computation intentionally keeps the full generated response.
    """
    bs, rlen = responses.shape
    m = len(end_think_ids)
    device = responses.device

    valid_len = response_attn.sum(dim=-1).long()
    if m == 0 or rlen < m:
        pos = torch.arange(rlen, device=device)[None, :]
        return (pos < valid_len[:, None]).to(dtype)

    needle = torch.tensor(end_think_ids, device=device, dtype=responses.dtype)
    windows = responses.unfold(dimension=1, size=m, step=1)
    matches = (windows == needle).all(dim=-1)

    starts = torch.arange(rlen - m + 1, device=device)[None, :]
    max_start = (valid_len - m).clamp(min=-1)
    matches = matches & (starts <= max_start[:, None])

    has = matches.any(dim=-1)
    first = matches.int().argmax(dim=-1)
    end_pos = torch.where(has, first + m + 1, valid_len)

    pos = torch.arange(rlen, device=device)[None, :]
    return (pos < end_pos[:, None]).to(dtype)


class DataParallelRPTActor(DataParallelPPOActor):
    """Stock PPO actor plus EMA-backed RPT reward computation.

    The class preserves normal PPO loss code but adds the state and helpers
    needed to score ground-truth continuation likelihood under:

    - the actor conditioned on prompt plus generated thought
    - the EMA teacher conditioned on prompt plus empty-think template

    The final scalar reward is the difference between those two quantities and
    is attached to the last thought token before grouped advantage computation.
    """

    def __init__(self, config, actor_module, actor_optimizer=None):
        super().__init__(config=config, actor_module=actor_module, actor_optimizer=actor_optimizer)
        self.tokenizer = None
        self.end_think_ids: list[int] = []
        self.drop_think_text = "<think>\n\n</think>\n\n"
        self.ema_tau = 0.999
        self.length_penalty = 0.0
        self.ema_module = None

    def setup_rpt(self, tokenizer, end_think_ids: list[int], ema_tau: float, length_penalty: float, drop_think_text: str):
        """Store RPT runtime state and initialize the EMA teacher copy."""
        self.tokenizer = tokenizer
        self.end_think_ids = list(end_think_ids)
        self.ema_tau = float(ema_tau)
        self.length_penalty = float(length_penalty)
        if drop_think_text:
            self.drop_think_text = drop_think_text
        self.init_ema_from_actor()

    def _get_ema_source_module(self):
        """Return the underlying user model instead of the FSDP wrapper shell."""
        module = self.actor_module
        seen: set[int] = set()
        while id(module) not in seen:
            seen.add(id(module))
            next_module = getattr(module, "_fsdp_wrapped_module", None)
            if next_module is not None:
                module = next_module
                continue
            next_module = getattr(module, "module", None)
            if next_module is not None:
                module = next_module
                continue
            break
        return module

    def _ema_actor_param_ctx(self):
        """Expose full actor params when the policy lives under an FSDP shell."""
        if isinstance(self.actor_module, FSDP):
            return FSDP.summon_full_params(self.actor_module, writeback=False, recurse=False)
        return contextlib.nullcontext()

    @staticmethod
    def _materialize_tensor(tensor: torch.Tensor) -> torch.Tensor:
        """Convert DTensor-like values into plain local tensors for EMA copies."""
        if hasattr(tensor, "full_tensor"):
            tensor = tensor.full_tensor()
        return tensor.detach().cpu().clone()

    @staticmethod
    def _detach_tensor_view(tensor: torch.Tensor) -> torch.Tensor:
        """Convert DTensor-like values into detached dense tensors for EMA math."""
        if hasattr(tensor, "full_tensor"):
            tensor = tensor.full_tensor()
        return tensor.detach()

    @torch.no_grad()
    def init_ema_from_actor(self):
        """Initialize the EMA module from the unwrapped actor state dict.

        ``copy.deepcopy`` on the FSDP-wrapped actor fails on the current
        PyTorch stack because the wrapper contains non-leaf tensors. Build a
        plain model instance from the wrapped module's config instead and load a
        detached copy of its state.
        """
        with self._ema_actor_param_ctx():
            source_module = self._get_ema_source_module()
            ema_config = copy.deepcopy(source_module.config)
            self.ema_module = source_module.__class__(ema_config)
            ema_state = {
                name: self._materialize_tensor(tensor)
                for name, tensor in source_module.state_dict().items()
                if "_flat_param" not in name
            }
            self.ema_module.load_state_dict(ema_state)
            first_param = next(source_module.parameters(), None)
            if first_param is not None:
                self.ema_module.to(device=first_param.device, dtype=self.param_dtype)
            else:
                self.ema_module.to(dtype=self.param_dtype)
        del ema_state
        self.ema_module.eval()
        for p in self.ema_module.parameters():
            p.requires_grad_(False)

    def _named_actor_params(self) -> dict[str, torch.Tensor]:
        """Collect actor params in the same key space used by EMA init."""
        source_module = self._get_ema_source_module()
        actor_state = source_module.state_dict()
        ema_param_names = {name for name, _ in self.ema_module.named_parameters()}
        return {
            name: self._detach_tensor_view(tensor)
            for name, tensor in actor_state.items()
            if name in ema_param_names and "_flat_param" not in name
        }

    @torch.no_grad()
    def ema_update(self, tau: float):
        """Update the EMA teacher after each optimizer step.

        The update is ``ema = tau * ema + (1 - tau) * actor``. Performing this
        step inside the actor update is what keeps the reward baseline aligned
        with the current mini-batch training state.
        """
        if self.ema_module is None:
            self.init_ema_from_actor()
        with self._ema_actor_param_ctx():
            actor_params = self._named_actor_params()
            for name, ep in self.ema_module.named_parameters():
                actor_param = actor_params[name]
                if ep.shape != actor_param.shape:
                    raise RuntimeError(
                        f"EMA parameter shape mismatch for {name}: ema={tuple(ep.shape)} actor={tuple(actor_param.shape)}"
                    )
                ep.data.mul_(tau).add_(actor_param.to(device=ep.device, dtype=ep.dtype), alpha=(1.0 - tau))

    def _forward_micro_batch_with_module(
        self,
        actor_module,
        micro_batch,
        temperature,
        calculate_entropy=False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reuse the stock micro-batch forward with either actor or EMA module.

        The helper temporarily swaps ``self.actor_module`` so the upstream
        forward implementation can be reused for both reward branches without
        rewriting the parallel log-prob path from scratch.
        """
        original_actor_module = self.actor_module
        self.actor_module = actor_module
        try:
            return self._forward_micro_batch(
                micro_batch=micro_batch,
                temperature=temperature,
                calculate_entropy=calculate_entropy,
            )
        finally:
            self.actor_module = original_actor_module

    def _extract_ground_truth(self, batch: DataProto) -> list[str]:
        """Read target continuations from the dataset-specific non-tensor fields.

        The intended source is ``non_tensor_batch['reward_model']['ground_truth']``
        as produced by ``omni_math_processing.py``. ``gt_text`` is kept only as a
        fallback.
        """
        if "reward_model" in batch.non_tensor_batch:
            gt_texts: list[str] = []
            for rm_obj in batch.non_tensor_batch["reward_model"]:
                if isinstance(rm_obj, dict):
                    gt_texts.append(str(rm_obj.get("ground_truth", "")))
                elif hasattr(rm_obj, "get"):
                    gt_texts.append(str(rm_obj.get("ground_truth", "")))
                else:
                    gt_texts.append(str(rm_obj))
            return gt_texts

        if "gt_text" in batch.non_tensor_batch:
            return [str(x) for x in batch.non_tensor_batch["gt_text"]]

        raise KeyError("ground truth not found in non_tensor_batch['reward_model'] nor ['gt_text']")

    @torch.no_grad()
    def compute_gt_logprob(self, batch: DataProto, which: str = "actor") -> DataProto:
        """Compute summed log-prob of the known target continuation.

        For each sample this method reconstructs a synthetic sequence:

        - prefix = prompt tokens + generated thought tokens
        - suffix = ground-truth target tokens

        The suffix log-probs are then summed into one scalar per sample and
        returned as ``gt_logprob_sum``. The same logic is used for both the
        actor and EMA modules.

        The implementation still uses explicit per-sample Python reconstruction
        loops. That is not the most elegant form, but it makes prompt boundary,
        thought boundary, and suffix tokenization behavior explicit for
        debugging.
        """
        if self.tokenizer is None:
            raise RuntimeError("RPT tokenizer is not initialized. Call setup_rpt() first.")

        module = self.actor_module if which == "actor" else self.ema_module
        if module is None:
            raise RuntimeError("EMA module is not initialized.")

        input_ids = batch.batch["input_ids"]
        attn = batch.batch["attention_mask"]
        responses = batch.batch["responses"]
        resp_mask = batch.batch["response_mask"]
        position_ids = batch.batch["position_ids"]

        bs, seq_len = input_ids.shape
        resp_len = responses.size(1)
        prompt_cut = seq_len - resp_len

        gt_texts = self._extract_ground_truth(batch)

        prefix_list = []
        suffix_list = []
        for i in range(bs):
            prompt_len_i = int(attn[i, :prompt_cut].sum().item())
            thought_len_i = int(resp_mask[i].sum().item())

            prompt_start = max(0, prompt_cut - prompt_len_i)
            prompt_ids = input_ids[i, prompt_start:prompt_cut]
            thought_ids = input_ids[i, prompt_cut : prompt_cut + thought_len_i]
            prefix_ids = torch.cat([prompt_ids, thought_ids], dim=0)

            suffix_token_ids = self.tokenizer.encode(gt_texts[i], add_special_tokens=False)
            if len(suffix_token_ids) == 0:
                eos_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 0
                suffix_token_ids = [eos_id]
            suffix_ids = torch.tensor(suffix_token_ids, device=input_ids.device, dtype=input_ids.dtype)

            prefix_list.append(prefix_ids)
            suffix_list.append(suffix_ids)

        max_pref = max(x.numel() for x in prefix_list)
        max_suf = max(x.numel() for x in suffix_list)
        seq_len_new = max_pref + max_suf

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 0

        new_input = torch.full((bs, seq_len_new), pad_id, device=input_ids.device, dtype=input_ids.dtype)
        new_attn = torch.zeros((bs, seq_len_new), device=input_ids.device, dtype=attn.dtype)
        new_pos = torch.zeros((bs, seq_len_new), device=input_ids.device, dtype=position_ids.dtype)
        new_resp = torch.full((bs, max_suf), pad_id, device=input_ids.device, dtype=input_ids.dtype)
        new_rmask = torch.zeros((bs, max_suf), device=input_ids.device, dtype=attn.dtype)

        for i in range(bs):
            pref = prefix_list[i]
            suf = suffix_list[i]
            pref_len = pref.numel()
            suf_len = suf.numel()

            new_input[i, :pref_len] = pref
            new_input[i, pref_len : pref_len + suf_len] = suf
            new_attn[i, : pref_len + suf_len] = 1
            new_pos[i, : pref_len + suf_len] = torch.arange(
                0,
                pref_len + suf_len,
                device=input_ids.device,
                dtype=position_ids.dtype,
            )

            new_resp[i, :suf_len] = suf
            new_rmask[i, :suf_len] = 1

        micro = {
            "input_ids": new_input,
            "attention_mask": new_attn,
            "position_ids": new_pos,
            "responses": new_resp,
            "response_mask": new_rmask,
        }

        _, token_lp = self._forward_micro_batch_with_module(
            actor_module=module,
            micro_batch=micro,
            temperature=1.0,
            calculate_entropy=False,
        )
        lp_sum = (token_lp * new_rmask.to(token_lp.dtype)).sum(dim=-1)

        return DataProto.from_dict(tensors={"gt_logprob_sum": lp_sum})

    @staticmethod
    def _dedupe_consecutive_uids(data: DataProto) -> tuple[DataProto, torch.Tensor]:
        """Collapse consecutive repeated prompt UIDs for EMA baseline reuse.

        Repeated rollouts from the same prompt share the same EMA baseline
        ``prompt + empty_think``. When those repetitions stay consecutive, the
        EMA forward only has to be done once per UID group and then expanded
        back with ``inverse_index``.

        ``uid`` lives in ``non_tensor_batch`` as a NumPy object array, so this
        path intentionally uses NumPy rather than ``torch.unique_consecutive``.
        """
        if "uid" not in data.non_tensor_batch:
            inv = torch.arange(len(data), dtype=torch.long)
            return data, inv

        uids = np.asarray(data.non_tensor_batch["uid"], dtype=object)
        n = len(uids)
        if n == 0:
            return data, torch.empty(0, dtype=torch.long)

        is_new_group = np.empty(n, dtype=bool)
        is_new_group[0] = True
        is_new_group[1:] = uids[1:] != uids[:-1]

        first_idx = np.flatnonzero(is_new_group).astype(np.int64)
        inverse = (np.cumsum(is_new_group) - 1).astype(np.int64)

        dedup = data.select_idxs(torch.from_numpy(first_idx))
        return dedup, torch.from_numpy(inverse)

    def _build_ema_batch_from_prompts(self, micro_batch: DataProto) -> tuple[DataProto, torch.Tensor]:
        """Construct the EMA baseline batch from prompt-only prefixes.

        The EMA branch must evaluate
        ``logp_ema(ground_truth | prompt + empty_think)``, not the generated
        thought. This helper therefore:

        1. deduplicates consecutive repeated prompt UIDs
        2. slices the prompt portion from ``input_ids``
        3. appends the configured empty-think template
        4. rebuilds attention / position / response tensors
        5. computes a response mask for the synthetic empty-think response

        The implementation assumes the current verl batch formatting where
        prompt slices remain aligned after removing the response region. If
        prompt boundaries or position ids look wrong on the VM, inspect this
        method first.
        """
        dedup_batch, inverse_index = self._dedupe_consecutive_uids(micro_batch)

        input_ids = dedup_batch.batch["input_ids"]
        attn_full = dedup_batch.batch["attention_mask"]
        responses = dedup_batch.batch["responses"]
        position_ids = dedup_batch.batch["position_ids"]

        bs = input_ids.size(0)
        seq_len = input_ids.size(1)
        resp_len = responses.size(1)
        prompt_cut = seq_len - resp_len

        drop_think_ids = self.tokenizer.encode(self.drop_think_text, add_special_tokens=False)
        if len(drop_think_ids) == 0:
            raise ValueError("drop_think text produced zero tokens")

        drop_think = torch.tensor(drop_think_ids, device=input_ids.device, dtype=input_ids.dtype)
        dlen = drop_think.size(0)
        bdrop_think = drop_think.view(1, -1).expand(bs, -1)

        prompt_input_ids = input_ids[:, :prompt_cut]
        prompt_attn = attn_full[:, :prompt_cut]
        prompt_pos = position_ids[:, :prompt_cut]

        ema_input_ids = torch.cat((prompt_input_ids, bdrop_think), dim=1)
        ema_attn = torch.cat(
            (
                prompt_attn,
                torch.ones((bs, dlen), device=prompt_attn.device, dtype=prompt_attn.dtype),
            ),
            dim=1,
        )

        if prompt_pos.size(1) > 0:
            last_prompt_pos = prompt_pos[:, -1:]
        else:
            last_prompt_pos = torch.zeros((bs, 1), device=position_ids.device, dtype=position_ids.dtype)
        delta = torch.arange(1, dlen + 1, device=position_ids.device, dtype=position_ids.dtype).view(1, -1)
        ema_pos = torch.cat((prompt_pos, last_prompt_pos + delta), dim=1)

        ema_responses = bdrop_think
        response_attn = ema_attn[:, -dlen:]

        thought_mask = make_thought_mask(
            responses=ema_responses,
            end_think_ids=self.end_think_ids,
            response_attn=response_attn,
            dtype=attn_full.dtype,
        )

        ema_batch = DataProto.from_dict(
            tensors={
                "input_ids": ema_input_ids,
                "attention_mask": ema_attn,
                "position_ids": ema_pos,
                "responses": ema_responses,
                "response_mask": thought_mask,
            },
            non_tensors=dedup_batch.non_tensor_batch,
            meta_info=dedup_batch.meta_info,
        )
        return ema_batch, inverse_index

    def _set_overlong_advantages(self, mini_batch: DataProto):
        """Experimental hook for penalizing advantages after the thought end."""
        if self.length_penalty == 0.0 or "advantages" not in mini_batch.batch.keys():
            return

        response_mask = mini_batch.batch["response_mask"]
        last_idx = (response_mask.sum(dim=-1).long() - 1).clamp(min=0)
        for i in range(response_mask.size(0)):
            # Keep the terminal thought token intact: that is where the scalar
            # RPT reward lands. Only tokens after it should get the penalty.
            mini_batch.batch["advantages"][i, last_idx[i] + 1 :] = self.length_penalty

    def _compute_reward_lp_minibatch(self, mini_batch: DataProto) -> DataProto:
        """Compute the full RPT reward tensors for one PPO mini-batch.

        The logical computation is mini-batch scoped, but the expensive forward
        passes are micro-batched for memory fit. For each micro-batch the actor
        branch scores the target continuation after the generated thought, while
        the EMA branch scores the same target after ``prompt + empty_think``.

        The resulting scalar reward

        ``think_gt_logprob_sum - ema_gt_logprob_sum``

        is placed on the last thought token and copied into both
        ``token_level_scores`` and ``token_level_rewards``.
        """
        responses = mini_batch.batch["responses"]
        attn_full = mini_batch.batch["attention_mask"]
        resp_len = responses.size(1)
        response_attn = attn_full[:, -resp_len:]

        thought_mask = make_thought_mask(
            responses=responses,
            response_attn=response_attn,
            end_think_ids=self.end_think_ids,
            dtype=attn_full.dtype,
        )
        mini_batch.batch["response_mask"] = thought_mask

        micro_bsz = self.config.ppo_micro_batch_size_per_gpu or len(mini_batch)
        micro_batches = mini_batch.split(micro_bsz)

        think_lp_list = []
        ema_lp_list = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())

            think_lp = self.compute_gt_logprob(micro_batch, which="actor").batch["gt_logprob_sum"]
            think_lp_list.append(think_lp.detach().cpu())

            ema_micro_batch, inverse_index = self._build_ema_batch_from_prompts(micro_batch)
            ema_lp_dedup = self.compute_gt_logprob(ema_micro_batch.to(get_device_id()), which="ema").batch[
                "gt_logprob_sum"
            ]
            ema_lp = ema_lp_dedup[inverse_index.to(ema_lp_dedup.device)]
            ema_lp_list.append(ema_lp.detach().cpu())

        think_lp = torch.cat(think_lp_list, dim=0).to(mini_batch.batch["responses"].device)
        ema_lp = torch.cat(ema_lp_list, dim=0).to(mini_batch.batch["responses"].device)

        mini_batch.batch["think_gt_logprob_sum"] = think_lp
        mini_batch.batch["ema_gt_logprob_sum"] = ema_lp

        reward = think_lp - ema_lp
        bsz = mini_batch.batch["responses"].size(0)
        reward_tensor = torch.zeros_like(mini_batch.batch["responses"], dtype=torch.float32)
        last_idx = (thought_mask.sum(dim=-1).long() - 1).clamp(min=0)
        row = torch.arange(bsz, device=last_idx.device)
        reward_tensor[row, last_idx] = reward.float()

        mini_batch.batch["token_level_scores"] = reward_tensor.clone()
        mini_batch.batch["token_level_rewards"] = reward_tensor
        print(reward_tensor)
        print(thought_mask)
        return mini_batch

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        """Run reward computation, advantage computation, PPO, and EMA updates.

        This method is the core of the custom training flow:

        1. select the tensors PPO still needs from the rollout batch
        2. split into PPO mini-batches
        3. compute RPT rewards inside each mini-batch
        4. compute grouped advantages inside the actor
        5. preserve the first-epoch mini-batches so the trainer gets back a
           batch containing rewards, scores, advantages, and returns
        6. run standard PPO optimization over micro-batches
        7. update the EMA teacher after every optimizer step

        Keeping reward and advantage creation here is the main behavioral
        deviation from stock PPO and is what lets the EMA baseline stay fresh at
        mini-batch granularity.
        """
        self.actor_module.train()

        temperature = data.meta_info["temperature"]

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
        ]
        if "values" in data.batch.keys():
            select_keys.append("values")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        if "rollout_is_weights" in data.batch.keys():
            select_keys.append("rollout_is_weights")
        if "rollout_log_probs" in data.batch.keys():
            select_keys.append("rollout_log_probs")

        data = data.select(batch_keys=select_keys)
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        adv_estimator = data.meta_info.get("adv_estimator", AdvantageEstimator.RLOO)
        if isinstance(adv_estimator, str):
            adv_estimator = AdvantageEstimator(adv_estimator)
        gamma = float(data.meta_info.get("gamma", 1.0))
        lam = float(data.meta_info.get("lam", 1.0))
        num_repeat = int(data.meta_info.get("num_repeat", 1))
        norm_adv_by_std = bool(data.meta_info.get("norm_adv_by_std_in_grpo", True))

        metrics = {
            "actor/pg_loss": 0.0,
            "actor/kl_loss": 0.0,
        }

        updated_mini_batches = []

        for epoch_idx in range(self.config.ppo_epochs):
            for mini_batch in mini_batches:
                mini_batch = self._compute_reward_lp_minibatch(mini_batch)
                mini_batch = compute_advantage(
                    mini_batch,
                    adv_estimator=adv_estimator,
                    gamma=gamma,
                    lam=lam,
                    num_repeat=num_repeat,
                    norm_adv_by_std_in_grpo=norm_adv_by_std,
                )
                

                #self._set_overlong_advantages(mini_batch)

                if epoch_idx == 0:
                    updated_mini_batches.append(mini_batch)

                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    calculate_entropy = self.config.calculate_entropy or (entropy_coeff != 0)

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    entropy, log_prob = self._forward_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                    )

                    if hasattr(self.config, "use_rollout_log_probs") and self.config.use_rollout_log_probs:
                        old_log_prob = model_inputs["old_log_probs"]
                    else:
                        if on_policy:
                            old_log_prob = log_prob.detach()
                        else:
                            old_log_prob = model_inputs["old_log_probs"]

                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    rollout_is_weights = model_inputs.get("rollout_is_weights", None)
                    policy_loss_fn = get_policy_loss_fn(loss_mode)

                    pg_loss, pg_metrics = policy_loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        loss_agg_mode=loss_agg_mode,
                        config=self.config,
                        rollout_is_weights=rollout_is_weights,
                    )
                    micro_batch_metrics.update(pg_metrics)

                    rollout_log_prob = model_inputs.get("rollout_log_probs", None)
                    if loss_mode != "bypass_mode" and rollout_log_prob is not None:
                        from verl.trainer.ppo.rollout_corr_helper import compute_rollout_corr_metrics_from_logprobs

                        rollout_corr_metrics = compute_rollout_corr_metrics_from_logprobs(
                            log_prob=log_prob,
                            rollout_log_prob=rollout_log_prob,
                            response_mask=response_mask,
                        )
                        micro_batch_metrics.update(rollout_corr_metrics)

                    policy_loss = pg_loss
                    if calculate_entropy and entropy is not None:
                        entropy_agg = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        micro_batch_metrics["actor/entropy"] = entropy_agg.detach().item()
                        if entropy_coeff != 0:
                            policy_loss -= entropy_agg * entropy_coeff

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type)
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics["actor/kl_loss"] += kl_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    loss = policy_loss * loss_scale_factor
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    metrics["actor/pg_loss"] += pg_loss.detach().item() * loss_scale_factor
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
                self.ema_update(self.ema_tau)

        self.actor_optimizer.zero_grad()

        updated_data = DataProto.concat(updated_mini_batches) if updated_mini_batches else data
        return updated_data, metrics
