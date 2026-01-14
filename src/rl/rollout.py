"""Rollout logic for operator-controlled denoising."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from src.actor.operator_policy import OperatorPolicy
from src.critic.aggregators import FEATURE_DIM
from src.critic.rel_energy import RelationalEnergyCritic
from src.operators.apply_operator import apply_operator
from src.operators.definitions import OperatorSpec, operator_ids
from src.operators.math_heuristics import extract_key_token_mask
from src.rl.types import EpisodeRecord, RolloutState, RolloutStep


def _unwrap_module(module: Any) -> Any:
    return getattr(module, "module", module)


@dataclass
class RolloutConfig:
    max_steps: int = 8
    gen_len: int = 32
    branch_steps: List[int] = field(default_factory=list)
    branch_k: int = 2
    selection_mode: str = "argmax"
    seed: int = 0
    use_hidden: bool = True
    cost_lambda: float = 0.0
    batch_size: int = 1
    fast_critic: bool = False
    gpu_critic: bool = False
    save_relation: bool = True
    fast_operator: bool = False
    ensure_update: bool = True
    fallback_operator: str = "O_SCOPE"


class RolloutEngine:
    def __init__(
        self,
        backbone: Any,
        actor: OperatorPolicy,
        critic: RelationalEnergyCritic,
        operator_specs: Optional[List[OperatorSpec]] = None,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.backbone = backbone
        self.actor = actor
        self.critic = critic
        self.operator_specs = operator_specs or []
        self.operator_id_list = operator_ids(self.operator_specs) if self.operator_specs else operator_ids()
        cfg = config or {}
        self.config = RolloutConfig(
            max_steps=int(cfg.get("max_steps", 8)),
            gen_len=int(cfg.get("gen_len", 32)),
            branch_steps=list(cfg.get("branch_steps", [])),
            branch_k=int(cfg.get("branch_k", 2)),
            selection_mode=str(cfg.get("selection_mode", "argmax")),
            seed=int(cfg.get("seed", 0)),
            use_hidden=bool(cfg.get("use_hidden", True)),
            cost_lambda=float(cfg.get("cost_lambda", 0.0)),
            batch_size=int(cfg.get("batch_size", 1)),
            fast_critic=bool(cfg.get("fast_critic", False)),
            gpu_critic=bool(cfg.get("gpu_critic", False)),
            save_relation=bool(cfg.get("save_relation", True)),
            fast_operator=bool(cfg.get("fast_operator", False)),
            ensure_update=bool(cfg.get("ensure_update", True)),
            fallback_operator=str(cfg.get("fallback_operator", "O_SCOPE")),
        )
        self.rng = random.Random(self.config.seed)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _operator_params(self, operator_id: str) -> Dict[str, Any]:
        for spec in self.operator_specs:
            if spec.operator_id.value == operator_id:
                return dict(spec.params)
        return {}

    def _token_strings(self, token_ids: List[int]) -> List[str]:
        if hasattr(self.backbone.tokenizer, "convert_ids_to_tokens"):
            return self.backbone.tokenizer.convert_ids_to_tokens(token_ids)
        return [str(idx) for idx in token_ids]

    def _build_batch_inputs(
        self,
        tokens_list: List[List[int]],
    ) -> tuple[torch.Tensor, torch.Tensor, List[int]]:
        lengths = [len(tokens) for tokens in tokens_list]
        max_len = max(lengths) if lengths else 0
        pad_id = getattr(self.backbone.tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(self.backbone.tokenizer, "eos_token_id", None)
        if pad_id is None:
            pad_id = 0
        input_ids = torch.full(
            (len(tokens_list), max_len),
            int(pad_id),
            device=self.device,
            dtype=torch.long,
        )
        attention_mask = torch.zeros(
            (len(tokens_list), max_len),
            device=self.device,
            dtype=torch.long,
        )
        for idx, tokens in enumerate(tokens_list):
            seq_len = lengths[idx]
            if seq_len == 0:
                continue
            input_ids[idx, :seq_len] = torch.tensor(tokens, device=self.device, dtype=torch.long)
            attention_mask[idx, :seq_len] = 1
        return input_ids, attention_mask, lengths

    def rollout_episode(
        self,
        prompt: str,
        task: Optional[str] = None,
        verifier: Optional[Any] = None,
        target_answer: Optional[str] = None,
    ) -> EpisodeRecord:
        episodes = self.rollout_batch(
            [prompt],
            tasks=[task],
            verifiers=[verifier],
            target_answers=[target_answer],
        )
        return episodes[0]

    def rollout_batch(
        self,
        prompts: List[str],
        tasks: Optional[List[Optional[str]]] = None,
        verifiers: Optional[List[Optional[Any]]] = None,
        target_answers: Optional[List[Optional[str]]] = None,
    ) -> List[EpisodeRecord]:
        if not prompts:
            return []
        batch = len(prompts)
        tasks = tasks if tasks is not None else [None] * batch
        verifiers = verifiers if verifiers is not None else [None] * batch
        target_answers = target_answers if target_answers is not None else [None] * batch
        if len(tasks) != batch or len(verifiers) != batch or len(target_answers) != batch:
            raise ValueError("Batch inputs must have the same length")

        actor_module = _unwrap_module(self.actor)
        critic_module = _unwrap_module(self.critic)
        mask_id = getattr(self.backbone, "mask_id", getattr(self.backbone.tokenizer, "mask_token_id", 0))
        gen_len = self.config.gen_len

        prompt_tokens_list = [self.backbone.tokenize(prompt) for prompt in prompts]
        prompt_lens = [len(tokens) for tokens in prompt_tokens_list]
        tokens_cpu = [tokens + [mask_id] * gen_len for tokens in prompt_tokens_list]
        token_strings_list = [self._token_strings(tokens) for tokens in tokens_cpu]
        prompt_token_strings_list = [
            list(token_strings[:prompt_len]) for token_strings, prompt_len in zip(token_strings_list, prompt_lens)
        ]
        seq_lens = [len(tokens) for tokens in tokens_cpu]
        max_len = max(seq_lens) if seq_lens else 0
        pad_id = getattr(self.backbone.tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(self.backbone.tokenizer, "eos_token_id", None)
        if pad_id is None:
            pad_id = 0
        tokens_tensor = torch.full(
            (batch, max_len),
            int(pad_id),
            device=self.device,
            dtype=torch.long,
        )
        attention_mask = torch.zeros(
            (batch, max_len),
            device=self.device,
            dtype=torch.long,
        )
        context_mask = torch.zeros(
            (batch, max_len),
            device=self.device,
            dtype=torch.long,
        )
        for idx, seq_len in enumerate(seq_lens):
            if seq_len == 0:
                continue
            tokens_tensor[idx, :seq_len] = torch.tensor(tokens_cpu[idx], device=self.device, dtype=torch.long)
            attention_mask[idx, :seq_len] = 1
            prompt_len = prompt_lens[idx]
            if prompt_len > 0:
                context_mask[idx, : min(prompt_len, seq_len)] = 1
        key_masks = [
            extract_key_token_mask(prompt, tokenizer=self.backbone.tokenizer)[0]
            for prompt in prompts
        ]
        metadata_list = [
            {"prompt": prompt, "task": task or "", "target_answer": target_answer}
            for prompt, task, target_answer in zip(prompts, tasks, target_answers)
        ]

        target_tokens_list: List[Optional[List[int]]] = []
        for prompt, target_answer in zip(prompts, target_answers):
            if target_answer:
                target_tokens_list.append(self.backbone.tokenize(f"{prompt} {target_answer}"))
            else:
                target_tokens_list.append(None)

        steps_list: List[List[RolloutStep]] = [[] for _ in range(batch)]
        total_cost = [0.0 for _ in range(batch)]

        for step_idx in range(self.config.max_steps):
            with torch.no_grad():
                backbone_output = self.backbone.forward(
                    tokens_tensor,
                    step_idx,
                    attention_mask=attention_mask,
                    context_mask=context_mask,
                )
            logits = backbone_output.logits
            hidden = backbone_output.hidden

            hidden_vectors: List[Optional[List[float]]] = [None for _ in range(batch)]
            pooled_hidden: Optional[torch.Tensor] = None
            if self.config.use_hidden and hidden is not None:
                if hidden.dim() == 3:
                    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
                    denom = mask.sum(dim=1).clamp(min=1.0)
                    pooled_hidden = (hidden * mask).sum(dim=1) / denom
                else:
                    pooled_hidden = hidden.mean(dim=0, keepdim=True)
                if pooled_hidden.shape[0] == 1 and batch > 1:
                    pooled_hidden = pooled_hidden.expand(batch, -1)
                hidden_vectors = pooled_hidden.detach().cpu().tolist()

            if self.config.fast_critic:
                base_features = torch.zeros((batch, FEATURE_DIM), device=self.device, dtype=torch.float32)
                if self.config.use_hidden and pooled_hidden is not None:
                    feature_tensor = torch.cat([base_features, pooled_hidden.to(torch.float32)], dim=-1)
                else:
                    feature_tensor = base_features
                with torch.no_grad():
                    q_scores = critic_module.forward(feature_tensor)
                    policy_output = self.actor(feature_tensor)
                selected_operators = actor_module.select_batch(
                    self.operator_id_list,
                    policy_output,
                    mode=self.config.selection_mode,
                )
                op_ids = self.operator_id_list
                operator_scores_batch = [
                    {op_id: float(q_scores[idx, j].item()) for j, op_id in enumerate(op_ids)}
                    for idx in range(batch)
                ]
                feature_vectors = feature_tensor.detach().cpu().tolist()
                for idx, selected_operator in enumerate(selected_operators):
                    prompt_len = prompt_lens[idx]
                    seq_len = seq_lens[idx]
                    token_strings = token_strings_list[idx]
                    state = RolloutState(
                        tokens=list(tokens_cpu[idx]),
                        prompt_len=prompt_len,
                        timestep=step_idx,
                        token_strings=[] if self.config.fast_operator else list(token_strings),
                        metadata=metadata_list[idx],
                    )
                    decision = apply_operator(
                        state,
                        selected_operator,
                        operator_params=self._operator_params(selected_operator),
                        tokenizer=self.backbone.tokenizer,
                    )
                    if self.config.ensure_update and not decision.update_positions:
                        fallback_id = self.config.fallback_operator or "O_SCOPE"
                        if fallback_id not in self.operator_id_list:
                            fallback_id = "O_SCOPE"
                        fallback = apply_operator(
                            state,
                            fallback_id,
                            operator_params=self._operator_params(fallback_id),
                            tokenizer=self.backbone.tokenizer,
                        )
                        fallback.meta["fallback_for"] = selected_operator
                        decision = fallback
                    if step_idx in self.config.branch_steps or decision.branch:
                        candidates = sorted(
                            operator_scores_batch[idx].items(),
                            key=lambda item: item[1],
                            reverse=True,
                        )
                        decision.meta["branch_candidates"] = candidates[: self.config.branch_k]

                    with torch.no_grad():
                        updated_tokens = self.backbone.apply_unmask(
                            tokens_tensor[idx, :seq_len],
                            logits[idx, :seq_len],
                            decision.update_positions,
                        )
                    if updated_tokens.dim() > 1:
                        updated_tokens = updated_tokens.squeeze(0)
                    tokens_tensor[idx, :seq_len] = updated_tokens
                    if decision.update_positions:
                        updated_ids = updated_tokens[decision.update_positions].detach().cpu().tolist()
                        if isinstance(updated_ids, int):
                            updated_ids = [updated_ids]
                        new_strings = self.backbone.tokenizer.convert_ids_to_tokens(updated_ids)
                        if isinstance(new_strings, str):
                            new_strings = [new_strings]
                        for pos, tok_id, tok_str in zip(decision.update_positions, updated_ids, new_strings):
                            if 0 <= pos < len(tokens_cpu[idx]):
                                tokens_cpu[idx][pos] = int(tok_id)
                                if pos < len(token_strings):
                                    token_strings[pos] = tok_str
                    total_cost[idx] += decision.cost
                    steps_list[idx].append(
                        RolloutStep(
                            state=state,
                            decision=decision,
                            relation=None,
                            operator_scores=operator_scores_batch[idx],
                            feature_vector=feature_vectors[idx],
                            next_tokens=list(tokens_cpu[idx]),
                            target_tokens=target_tokens_list[idx],
                        )
                    )
                continue

            gen_token_strings_batch: List[List[str]] = []
            key_masks_batch: List[List[bool]] = []
            for idx in range(batch):
                prompt_len = prompt_lens[idx]
                seq_len = seq_lens[idx]
                token_strings = token_strings_list[idx]
                gen_token_strings_batch.append(token_strings[prompt_len:seq_len])
                key_mask_aligned = list(key_masks[idx][:prompt_len])
                if len(key_mask_aligned) < prompt_len:
                    key_mask_aligned.extend([False] * (prompt_len - len(key_mask_aligned)))
                key_masks_batch.append(key_mask_aligned)

            if self.config.gpu_critic:
                prompt_ids_batch: List[torch.Tensor] = []
                gen_ids_batch: List[torch.Tensor] = []
                logits_batch: List[torch.Tensor] = []
                for idx in range(batch):
                    prompt_len = prompt_lens[idx]
                    seq_len = seq_lens[idx]
                    prompt_ids_batch.append(tokens_tensor[idx, :prompt_len])
                    gen_ids_batch.append(tokens_tensor[idx, prompt_len:seq_len])
                    logits_batch.append(logits[idx, prompt_len:seq_len])
                critic_outputs = critic_module.evaluate_batch_gpu(
                    prompt_tokens_batch=prompt_token_strings_list,
                    gen_tokens_batch=gen_token_strings_batch,
                    prompt_ids_batch=prompt_ids_batch,
                    gen_ids_batch=gen_ids_batch,
                    key_prompt_mask_batch=key_masks_batch,
                    logits_batch=logits_batch,
                    special_token_ids={
                        "eos_id": getattr(self.backbone.tokenizer, "eos_token_id", None),
                        "pad_id": getattr(self.backbone.tokenizer, "pad_token_id", None),
                        "bos_id": getattr(self.backbone.tokenizer, "bos_token_id", None),
                        "mask_id": getattr(self.backbone, "mask_id", None),
                    },
                    operator_id_list=self.operator_id_list,
                    hidden_vectors=hidden_vectors,
                    device=self.device,
                    return_relation=self.config.save_relation,
                )
            else:
                gen_logits_batch: List[List[List[float]]] = []
                for idx in range(batch):
                    prompt_len = prompt_lens[idx]
                    seq_len = seq_lens[idx]
                    gen_logits = logits[idx, prompt_len:seq_len]
                    gen_logits_batch.append(gen_logits.detach().cpu().tolist())
                critic_outputs = critic_module.evaluate_batch(
                    prompt_tokens_batch=prompt_token_strings_list,
                    gen_tokens_batch=gen_token_strings_batch,
                    key_prompt_mask_batch=key_masks_batch,
                    logits_batch=gen_logits_batch,
                    special_token_ids={
                        "eos_id": getattr(self.backbone.tokenizer, "eos_token_id", None),
                        "pad_id": getattr(self.backbone.tokenizer, "pad_token_id", None),
                        "bos_id": getattr(self.backbone.tokenizer, "bos_token_id", None),
                        "mask_id": getattr(self.backbone, "mask_id", None),
                    },
                    operator_id_list=self.operator_id_list,
                    gen_texts=None,
                    hidden_vectors=hidden_vectors,
                    device=self.device,
                )

            feature_vectors = [output.feature_vector for output in critic_outputs]
            feature_tensor = torch.tensor(feature_vectors, device=self.device, dtype=torch.float32)
            with torch.no_grad():
                policy_output = self.actor(feature_tensor)
            selected_operators = actor_module.select_batch(
                self.operator_id_list,
                policy_output,
                mode=self.config.selection_mode,
            )

            for idx, (critic_output, selected_operator) in enumerate(
                zip(critic_outputs, selected_operators)
            ):
                prompt_len = prompt_lens[idx]
                seq_len = seq_lens[idx]
                token_strings = token_strings_list[idx]
                state = RolloutState(
                    tokens=list(tokens_cpu[idx]),
                    prompt_len=prompt_len,
                    timestep=step_idx,
                    token_strings=list(token_strings),
                    metadata=metadata_list[idx],
                )
                decision = apply_operator(
                    state,
                    selected_operator,
                    operator_params=self._operator_params(selected_operator),
                    tokenizer=self.backbone.tokenizer,
                )
                if self.config.ensure_update and not decision.update_positions:
                    fallback_id = self.config.fallback_operator or "O_SCOPE"
                    if fallback_id not in self.operator_id_list:
                        fallback_id = "O_SCOPE"
                    fallback = apply_operator(
                        state,
                        fallback_id,
                        operator_params=self._operator_params(fallback_id),
                        tokenizer=self.backbone.tokenizer,
                    )
                    fallback.meta["fallback_for"] = selected_operator
                    decision = fallback

                if step_idx in self.config.branch_steps or decision.branch:
                    candidates = sorted(
                        critic_output.operator_scores.items(),
                        key=lambda item: item[1],
                        reverse=True,
                    )
                    decision.meta["branch_candidates"] = candidates[: self.config.branch_k]

                with torch.no_grad():
                    updated_tokens = self.backbone.apply_unmask(
                        tokens_tensor[idx, :seq_len],
                        logits[idx, :seq_len],
                        decision.update_positions,
                    )
                if updated_tokens.dim() > 1:
                    updated_tokens = updated_tokens.squeeze(0)
                tokens_tensor[idx, :seq_len] = updated_tokens
                if decision.update_positions:
                    updated_ids = updated_tokens[decision.update_positions].detach().cpu().tolist()
                    if isinstance(updated_ids, int):
                        updated_ids = [updated_ids]
                    new_strings = self.backbone.tokenizer.convert_ids_to_tokens(updated_ids)
                    if isinstance(new_strings, str):
                        new_strings = [new_strings]
                    for pos, tok_id, tok_str in zip(decision.update_positions, updated_ids, new_strings):
                        if 0 <= pos < len(tokens_cpu[idx]):
                            tokens_cpu[idx][pos] = int(tok_id)
                            if pos < len(token_strings):
                                token_strings[pos] = tok_str
                total_cost[idx] += decision.cost
                steps_list[idx].append(
                    RolloutStep(
                        state=state,
                        decision=decision,
                        relation=critic_output.relation,
                        operator_scores=critic_output.operator_scores,
                        feature_vector=critic_output.feature_vector,
                        next_tokens=list(tokens_cpu[idx]),
                        target_tokens=target_tokens_list[idx],
                    )
                )

        episodes: List[EpisodeRecord] = []
        for idx in range(batch):
            prompt_len = prompt_lens[idx]
            final_tokens = list(tokens_cpu[idx])
            gen_tokens = list(final_tokens[prompt_len:])
            special_ids: set[int] = set()
            mask_id = getattr(self.backbone, "mask_id", None)
            if mask_id is not None:
                special_ids.add(int(mask_id))
            for attr in ("eos_token_id", "pad_token_id", "bos_token_id"):
                tok_id = getattr(self.backbone.tokenizer, attr, None)
                if tok_id is not None:
                    special_ids.add(int(tok_id))
            if special_ids:
                gen_tokens = [tok for tok in gen_tokens if tok not in special_ids]
            final_text = self.backbone.decode(gen_tokens)
            reward = 0.0
            verifier = verifiers[idx]
            if verifier is not None:
                try:
                    reward = float(verifier(final_text, metadata_list[idx]))
                except TypeError:
                    reward = float(verifier(final_text))
            episode_return = reward - self.config.cost_lambda * total_cost[idx]
            for step in steps_list[idx]:
                step.return_value = episode_return
            episodes.append(
                EpisodeRecord(
                    prompt=prompts[idx],
                    prompt_tokens=prompt_tokens_list[idx],
                    steps=steps_list[idx],
                    final_tokens=final_tokens,
                    final_text=final_text,
                    reward=reward,
                    cost=total_cost[idx],
                    metrics={"task": tasks[idx] or ""},
                )
            )
        return episodes
