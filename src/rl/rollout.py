"""Rollout logic for operator-controlled denoising."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from src.actor.operator_policy import OperatorPolicy
from src.critic.rel_energy import RelationalEnergyCritic
from src.operators.apply_operator import apply_operator
from src.operators.definitions import OperatorSpec, operator_ids
from src.operators.math_heuristics import extract_key_token_mask
from src.rl.types import EpisodeRecord, RolloutState, RolloutStep


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

    def rollout_episode(
        self,
        prompt: str,
        task: Optional[str] = None,
        verifier: Optional[Any] = None,
        target_answer: Optional[str] = None,
    ) -> EpisodeRecord:
        prompt_tokens = self.backbone.tokenize(prompt)
        prompt_len = len(prompt_tokens)
        mask_id = getattr(self.backbone, "mask_id", getattr(self.backbone.tokenizer, "mask_token_id", 0))
        gen_len = self.config.gen_len
        tokens = prompt_tokens + [mask_id] * gen_len
        steps: List[RolloutStep] = []
        total_cost = 0.0
        key_mask, _ = extract_key_token_mask(prompt, tokenizer=self.backbone.tokenizer)

        target_tokens: Optional[List[int]] = None
        if target_answer:
            target_tokens = self.backbone.tokenize(f"{prompt} {target_answer}")

        for step_idx in range(self.config.max_steps):
            token_strings = self._token_strings(tokens)
            state = RolloutState(
                tokens=list(tokens),
                prompt_len=prompt_len,
                timestep=step_idx,
                token_strings=token_strings,
                metadata={"prompt": prompt, "task": task or "", "target_answer": target_answer},
            )

            prompt_token_strings = token_strings[:prompt_len]
            gen_token_strings = token_strings[prompt_len:]
            key_mask_aligned = list(key_mask[:prompt_len])
            if len(key_mask_aligned) < prompt_len:
                key_mask_aligned.extend([False] * (prompt_len - len(key_mask_aligned)))

            input_ids = torch.tensor(tokens, device=self.device)
            with torch.no_grad():
                backbone_output = self.backbone.forward(input_ids, step_idx)
            hidden = backbone_output.hidden
            hidden_vector: Optional[List[float]] = None
            if self.config.use_hidden and hidden is not None:
                pooled = hidden.mean(dim=1) if hidden.dim() == 3 else hidden.mean(dim=0)
                hidden_vector = pooled.squeeze(0).detach().cpu().tolist()

            gen_logits = backbone_output.logits[0, prompt_len:]
            gen_logits_list = gen_logits.detach().cpu().tolist()

            critic_output = self.critic.evaluate(
                prompt_tokens=prompt_token_strings,
                gen_tokens=gen_token_strings,
                key_prompt_mask=key_mask_aligned,
                logits=gen_logits_list,
                special_token_ids={
                    "eos_id": getattr(self.backbone.tokenizer, "eos_token_id", None),
                    "pad_id": getattr(self.backbone.tokenizer, "pad_token_id", None),
                    "bos_id": getattr(self.backbone.tokenizer, "bos_token_id", None),
                    "mask_id": getattr(self.backbone, "mask_id", None),
                },
                operator_id_list=self.operator_id_list,
                gen_text=self.backbone.decode(tokens[prompt_len:]),
                hidden_vector=hidden_vector,
                device=self.device,
            )

            feature_tensor = torch.tensor(critic_output.feature_vector, device=self.device).unsqueeze(0)
            with torch.no_grad():
                policy_output = self.actor(feature_tensor)
            selected_operator = self.actor.select(
                self.operator_id_list,
                policy_output,
                mode=self.config.selection_mode,
            )

            decision = apply_operator(
                state,
                selected_operator,
                operator_params=self._operator_params(selected_operator),
                tokenizer=self.backbone.tokenizer,
            )

            if step_idx in self.config.branch_steps or decision.branch:
                candidates = sorted(
                    critic_output.operator_scores.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
                decision.meta["branch_candidates"] = candidates[: self.config.branch_k]

            with torch.no_grad():
                next_tokens_tensor, _ = self.backbone.step(input_ids, step_idx, decision.update_mask)
            tokens = next_tokens_tensor.detach().cpu().tolist()
            total_cost += decision.cost
            steps.append(
                RolloutStep(
                    state=state,
                    decision=decision,
                    relation=critic_output.relation,
                    operator_scores=critic_output.operator_scores,
                    feature_vector=critic_output.feature_vector,
                    next_tokens=list(tokens),
                    target_tokens=target_tokens,
                )
            )

        final_text = self.backbone.decode(tokens[prompt_len:])
        reward = 0.0
        if verifier is not None:
            try:
                reward = float(verifier(final_text, state.metadata))
            except TypeError:
                reward = float(verifier(final_text))

        episode_return = reward - self.config.cost_lambda * total_cost
        for step in steps:
            step.return_value = episode_return

        return EpisodeRecord(
            prompt=prompt,
            prompt_tokens=prompt_tokens,
            steps=steps,
            final_tokens=tokens,
            final_text=final_text,
            reward=reward,
            cost=total_cost,
            metrics={"task": task or ""},
        )
