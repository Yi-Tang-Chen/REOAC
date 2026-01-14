"""MDLM/E2D2 backbone wrapper using Hugging Face masked LMs."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Dict, List, Optional, Tuple

import importlib.util
import sys

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers.utils import logging as hf_logging
import math

_MDLM_DATALOADER = None


def _resolve_dataloader_path() -> Path:
    root = Path(__file__).resolve().parents[2]
    candidates = [
        root / "third_party" / "e2d2" / "dataloader.py",
        root / "third_party" / "mdlm" / "dataloader.py",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[-1]


def _load_mdlm_dataloader():
    global _MDLM_DATALOADER
    if _MDLM_DATALOADER is not None:
        return _MDLM_DATALOADER
    dataloader_path = _resolve_dataloader_path()
    mdlm_dir = dataloader_path.parent
    if str(mdlm_dir) not in sys.path:
        sys.path.insert(0, str(mdlm_dir))
    if not dataloader_path.is_file():
        raise RuntimeError("Failed to locate dataloader.py in third_party/e2d2 or third_party/mdlm")
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


def _load_hf_tokenizer(tokenizer_name: str, trust_remote_code: bool, fallback_name: str) -> AutoTokenizer:
    try:
        return AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=trust_remote_code)
    except (ValueError, OSError):
        return AutoTokenizer.from_pretrained(fallback_name)


def _ensure_special_tokens(tokenizer: AutoTokenizer) -> None:
    if tokenizer.bos_token is None and tokenizer.cls_token is not None:
        tokenizer.bos_token = tokenizer.cls_token
    if tokenizer.eos_token is None and tokenizer.sep_token is not None:
        tokenizer.eos_token = tokenizer.sep_token
    if tokenizer.bos_token is None and tokenizer.eos_token is not None:
        tokenizer.bos_token = tokenizer.eos_token
    if tokenizer.eos_token is None and tokenizer.bos_token is not None:
        tokenizer.eos_token = tokenizer.bos_token
    if tokenizer.pad_token is None:
        if hasattr(tokenizer, "get_added_vocab"):
            added_vocab = tokenizer.get_added_vocab()
            if "<|finetune_right_pad_id|>" in added_vocab:
                tokenizer.pad_token = "<|finetune_right_pad_id|>"
            elif tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        elif tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.mask_token is None:
        added_vocab = tokenizer.get_added_vocab() if hasattr(tokenizer, "get_added_vocab") else {}
        if "<|reserved_special_token_0|>" in added_vocab:
            tokenizer.mask_token = "<|reserved_special_token_0|>"
            tokenizer.mask_token_id = added_vocab["<|reserved_special_token_0|>"]
        elif "<|fim_middle|>" in added_vocab:
            tokenizer.mask_token = "<|fim_middle|>"
            tokenizer.mask_token_id = added_vocab["<|fim_middle|>"]
        elif hasattr(tokenizer, "vocab") and "_MASK" in tokenizer.vocab:
            tokenizer.mask_token = "_MASK"
            tokenizer.mask_token_id = tokenizer.vocab["_MASK"]
        else:
            tokenizer.add_special_tokens({"mask_token": "<|fim_middle|>"})


@dataclass
class BackboneOutput:
    logits: torch.Tensor
    hidden: torch.Tensor


class MDLMWrapper(torch.nn.Module):
    def __init__(self, config: Optional[Dict[str, object]] = None) -> None:
        super().__init__()
        self.config = config or {}
        model_name = self.config.get("model_name_or_path", "kuleshov-group/e2d2-owt")
        tokenizer_name = self.config.get("tokenizer_name_or_path", model_name)
        trust_remote_code = bool(self.config.get("trust_remote_code", True))
        self.max_length = int(self.config.get("max_length", 256))
        self.sample_strategy = self.config.get("sample_strategy", "argmax")
        self.temperature = float(self.config.get("temperature", 1.0))

        tokenizer_source = self.config.get("tokenizer_source", "auto")
        fallback_name = self.config.get("tokenizer_fallback", "gpt2")
        if int(os.environ.get("RANK", "0")) != 0 or self.config.get("suppress_load_warnings"):
            hf_logging.set_verbosity_error()
        if tokenizer_source == "mdlm":
            try:
                self.tokenizer = _get_mdlm_tokenizer(tokenizer_name)
            except Exception:
                self.tokenizer = _load_hf_tokenizer(tokenizer_name, trust_remote_code, fallback_name)
        elif tokenizer_source == "e2d2":
            self.tokenizer = _load_hf_tokenizer(tokenizer_name, trust_remote_code, fallback_name)
        else:
            self.tokenizer = _load_hf_tokenizer(tokenizer_name, trust_remote_code, fallback_name)
        _ensure_special_tokens(self.tokenizer)

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
        if self._uses_e2d2():
            if hasattr(self.model.config, "use_cache"):
                self.model.config.use_cache = False
            self._patch_e2d2_encoder_forward()
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

        mask_id = getattr(self.model.config, "mask_token_id", None)
        if mask_id is None:
            mask_id = self.tokenizer.mask_token_id
        if mask_id is None:
            fallback = self.tokenizer.eos_token_id
            if fallback is None:
                raise RuntimeError("Tokenizer does not define mask/eos token id.")
            mask_id = fallback
        self.mask_id = int(mask_id)
        if self.mask_id >= self.vocab_size:
            self.mask_id = self.vocab_size - 1
        try:
            self.model.config.mask_token_id = self.mask_id
        except Exception:
            pass

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

    def _uses_e2d2(self) -> bool:
        name = str(self.config.get("name", "")).lower()
        model_name = str(self.config.get("model_name_or_path", "")).lower()
        return name == "e2d2" or "e2d2" in model_name

    def _unwrap_model(self):
        return self.model.module if hasattr(self.model, "module") else self.model

    def _e2d2_embed_tokens(self, input_ids: torch.Tensor) -> Optional[torch.Tensor]:
        base_model = self._unwrap_model()
        backbone = getattr(base_model, "backbone", None)
        for attr in ("encoder", "decoder"):
            module = getattr(backbone, attr, None)
            model = getattr(module, "model", None)
            embed = getattr(model, "embed_tokens", None)
            if embed is not None:
                return embed(input_ids)
        if hasattr(base_model, "get_input_embeddings"):
            emb = base_model.get_input_embeddings()
            if emb is not None:
                return emb(input_ids)
        return None

    def _patch_e2d2_encoder_forward(self) -> None:
        backbone = getattr(self.model, "backbone", None)
        encoder = getattr(backbone, "encoder", None)
        encoder_model = getattr(encoder, "model", None)
        if encoder_model is None or not hasattr(encoder_model, "layers"):
            return
        if getattr(encoder_model, "_reoac_tuple_fix", False):
            return
        orig_forward = encoder_model.forward

        def _wrap_layer_forward(bound_forward):
            def _forward(self_layer, *args, **kwargs):
                out = bound_forward(*args, **kwargs)
                return out[0] if isinstance(out, tuple) else out
            return _forward

        def _forward_with_tuple_fix(self_model, *args, **kwargs):
            original_forwards = []
            try:
                for layer in self_model.layers:
                    original_forwards.append(layer.forward)
                    layer.forward = MethodType(_wrap_layer_forward(layer.forward), layer)
                return orig_forward(*args, **kwargs)
            finally:
                for layer, original in zip(self_model.layers, original_forwards):
                    layer.forward = original

        encoder_model.forward = MethodType(_forward_with_tuple_fix, encoder_model)
        encoder_model._reoac_tuple_fix = True

    def _prepare_e2d2_inputs(
        self,
        x_t: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], int]:
        seq_len = x_t.shape[1]
        base_model = self._unwrap_model()
        model_len = int(getattr(base_model.config, "length", seq_len) or seq_len)
        if model_len <= 0:
            model_len = seq_len
        if attention_mask is None:
            attention_mask = torch.ones((x_t.shape[0], seq_len), device=x_t.device, dtype=torch.long)
        else:
            if attention_mask.dim() == 1:
                attention_mask = attention_mask.unsqueeze(0)
            attention_mask = attention_mask.to(device=x_t.device, dtype=torch.long)
        if context_mask is not None:
            if context_mask.dim() == 1:
                context_mask = context_mask.unsqueeze(0)
            context_mask = context_mask.to(device=x_t.device, dtype=torch.long)
        if seq_len > model_len:
            x_t = x_t[:, :model_len]
            attention_mask = attention_mask[:, :model_len]
            if context_mask is not None:
                context_mask = context_mask[:, :model_len]
            seq_len = model_len
        if model_len > seq_len:
            pad_id = self.tokenizer.pad_token_id
            if pad_id is None:
                pad_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 0
            pad = torch.full(
                (x_t.shape[0], model_len - seq_len),
                int(pad_id),
                device=x_t.device,
                dtype=x_t.dtype,
            )
            input_ids = torch.cat([x_t, pad], dim=1)
            attention_mask = torch.cat(
                [attention_mask, torch.zeros((x_t.shape[0], model_len - seq_len), device=x_t.device, dtype=torch.long)],
                dim=1,
            )
            if context_mask is not None:
                context_mask = torch.cat(
                    [
                        context_mask,
                        torch.zeros((x_t.shape[0], model_len - seq_len), device=x_t.device, dtype=torch.long),
                    ],
                    dim=1,
                )
        else:
            input_ids = x_t
            attention_mask = attention_mask[:, :seq_len]
            if context_mask is not None:
                context_mask = context_mask[:, :seq_len]
        return input_ids, attention_mask, context_mask, seq_len


    def forward(
        self,
        x_t: torch.Tensor,
        timestep: int,
        cond: Optional[object] = None,
        attention_mask: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
    ) -> BackboneOutput:
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
        if self._uses_e2d2():
            input_ids, attention_mask, context_mask, seq_len = self._prepare_e2d2_inputs(
                x_t,
                attention_mask=attention_mask,
                context_mask=context_mask,
            )
            model_kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "t": timesteps,
            }
            if context_mask is not None:
                model_kwargs["context_mask"] = context_mask
            outputs = self.model.forward(**model_kwargs)

            logits = outputs.logits[:, :seq_len, :]
            embeddings = self._e2d2_embed_tokens(input_ids)
            if embeddings is not None:
                hidden = embeddings[:, :seq_len, :]
            else:
                hidden = torch.zeros(
                    (input_ids.shape[0], seq_len, self.hidden_size),
                    device=input_ids.device,
                    dtype=logits.dtype,
                )
        else:
            model_kwargs = {
                "input_ids": x_t,
                "timesteps": timesteps,
                "output_hidden_states": True,
                "return_dict": True,
            }
            if attention_mask is not None:
                model_kwargs["attention_mask"] = attention_mask
            outputs = self.model(**model_kwargs)
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
