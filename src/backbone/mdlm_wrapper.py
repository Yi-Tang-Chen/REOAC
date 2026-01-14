"""MDLM backbone wrapper using Hugging Face masked LMs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import importlib.util
import sys

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
import math
from typing import Optional

_MDLM_DATALOADER = None


def _load_mdlm_dataloader():
    global _MDLM_DATALOADER
    if _MDLM_DATALOADER is not None:
        return _MDLM_DATALOADER
    root = Path(__file__).resolve().parents[2]
    mdlm_dir = root / "third_party" / "mdlm"
    dataloader_path = mdlm_dir / "dataloader.py"
    if str(mdlm_dir) not in sys.path:
        sys.path.insert(0, str(mdlm_dir))
    spec = importlib.util.spec_from_file_location("mdlm_dataloader", str(dataloader_path))
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to locate MDLM dataloader.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _MDLM_DATALOADER = module
    return module


def _get_mdlm_tokenizer(tokenizer_name: str):
    dataloader = _load_mdlm_dataloader()
    cfg = SimpleNamespace(data=SimpleNamespace(tokenizer_name_or_path=tokenizer_name))
    return dataloader.get_tokenizer(cfg)


@dataclass
class BackboneOutput:
    logits: torch.Tensor
    hidden: torch.Tensor


class MDLMWrapper(torch.nn.Module):
    def __init__(self, config: Optional[Dict[str, object]] = None) -> None:
        super().__init__()
        self.config = config or {}
        model_name = self.config.get("model_name_or_path", "kuleshov-group/mdlm-owt")
        tokenizer_name = self.config.get("tokenizer_name_or_path", model_name)
        trust_remote_code = bool(self.config.get("trust_remote_code", True))
        self.max_length = int(self.config.get("max_length", 256))
        self.sample_strategy = self.config.get("sample_strategy", "argmax")
        self.temperature = float(self.config.get("temperature", 1.0))

        tokenizer_source = self.config.get("tokenizer_source", "auto")
        if tokenizer_source == "mdlm":
            try:
                self.tokenizer = _get_mdlm_tokenizer(tokenizer_name)
            except Exception:
                fallback_name = self.config.get("tokenizer_fallback", "gpt2")
                self.tokenizer = _get_mdlm_tokenizer(fallback_name)
        else:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=trust_remote_code)
            except (ValueError, OSError):
                fallback_name = self.config.get("tokenizer_fallback", "gpt2")
                self.tokenizer = AutoTokenizer.from_pretrained(fallback_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        if self.tokenizer.mask_token is None:
            self.tokenizer.add_special_tokens({"mask_token": "<mask>"})

        dtype = self.config.get("torch_dtype")
        torch_dtype = None
        if dtype == "float16":
            torch_dtype = torch.float16
        elif dtype == "bfloat16":
            torch_dtype = torch.bfloat16

        self.model = AutoModelForMaskedLM.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
        )
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        try:
            emb = self.model.get_input_embeddings()
            if emb is not None and len(self.tokenizer) > emb.weight.shape[0]:
                self.model.resize_token_embeddings(len(self.tokenizer))
        except NotImplementedError:
            # Some custom models may not expose embeddings; skip resize.
            pass
        self.hidden_size = int(getattr(self.model.config, "hidden_size", 0)) or int(
            getattr(self.model.config, "n_embd", 0)
        )
        self.vocab_size = int(getattr(self.model.config, "vocab_size", len(self.tokenizer)))

        self.mask_id = self.tokenizer.mask_token_id
        if self.mask_id is None:
            fallback = self.tokenizer.eos_token_id
            if fallback is None:
                raise RuntimeError("Tokenizer does not define mask/eos token id.")
            self.mask_id = fallback
        if self.mask_id >= self.vocab_size:
            self.mask_id = self.vocab_size - 1

    def to(self, *args, **kwargs):
        self.model.to(*args, **kwargs)
        return self

    def parameters(self, recurse: bool = True):
        return self.model.parameters(recurse=recurse)

    def tokenize(self, prompt: str) -> List[int]:
        encoded = self.tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )
        return list(encoded["input_ids"])

    def decode(self, ids: List[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def _sigma_from_step(self, step: int, batch: int, device: torch.device) -> torch.Tensor:
        # rollout 的 step_idx -> 連續 timesteps(sigma)
        T = int(self.config.get("num_steps", 32))
        sigma_min = float(self.config.get("sigma_min", 0.0))
        sigma_max = float(self.config.get("sigma_max", 1.0))
        schedule = str(self.config.get("sigma_schedule", "cosine"))

        if T <= 1:
            val = sigma_min
        else:
            t = float(step) / float(T - 1)  # 0 -> 1
            if schedule == "linear":
                # high -> low
                val = sigma_max + (sigma_min - sigma_max) * t
            else:
                # cosine, high -> low
                val = sigma_min + 0.5 * (sigma_max - sigma_min) * (1.0 + math.cos(math.pi * t))

        return torch.full((batch,), float(val), device=device, dtype=torch.float32)


    def forward(self, x_t: torch.Tensor, timestep: int, cond: Optional[object] = None) -> BackboneOutput:
        if x_t.dim() == 1:
            x_t = x_t.unsqueeze(0)

        # 這裡的 timesteps 就是 MDLM backbone 期待的 sigma tensor
        timesteps = self._sigma_from_step(timestep, x_t.shape[0], x_t.device)
        # --- DEBUG: catch invalid token ids early (prevents CUDA assert) ---
        if x_t.dtype != torch.long:
            x_t = x_t.long()

        vocab_size = self.vocab_size
        bad = (x_t < 0) | (x_t >= vocab_size)
        if bad.any():
            x_t = torch.clamp(x_t, 0, vocab_size - 1)
        outputs = self.model(
            input_ids=x_t,
            timesteps=timesteps,
            output_hidden_states=True,
            return_dict=True,
        )
        logits = outputs.logits
        hidden = outputs.hidden_states[-1]
        return BackboneOutput(logits=logits, hidden=hidden)


    def get_hidden(self, x_t: torch.Tensor, timestep: int, cond: Optional[object] = None) -> torch.Tensor:
        return self.forward(x_t, timestep, cond=cond).hidden

    def apply_unmask(self, x_t: torch.Tensor, logits: torch.Tensor, update_positions: List[int]) -> torch.Tensor:
        if x_t.dim() == 1:
            x_t = x_t.unsqueeze(0)
        if logits.dim() == 2:
            logits = logits.unsqueeze(0)

        B, L = x_t.shape
        new_tokens = x_t.clone()

        for pos in update_positions:
            if 0 <= pos < L:
                scores = logits[:, pos, :] / max(self.temperature, 1e-6)  # (B, V)

                if self.sample_strategy == "sample":
                    probs = torch.softmax(scores, dim=-1)
                    # torch.multinomial 需要 2D，回 (B, 1)
                    sampled = torch.multinomial(probs, 1).squeeze(1)  # (B,)
                    new_tokens[:, pos] = sampled.to(new_tokens.dtype)
                else:
                    new_tokens[:, pos] = torch.argmax(scores, dim=-1).to(new_tokens.dtype)

        return new_tokens.squeeze(0) if new_tokens.shape[0] == 1 else new_tokens

    def step(
        self,
        x_t: torch.Tensor,
        timestep: int,
        operator_mask: List[bool],
        cond: Optional[object] = None,
    ) -> Tuple[torch.Tensor, BackboneOutput]:
        output = self.forward(x_t, timestep, cond=cond)
        update_positions = [idx for idx, flag in enumerate(operator_mask) if flag]
        x_next = self.apply_unmask(x_t, output.logits, update_positions)
        return x_next, output
