"""LoRA management utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

try:
    from peft import LoraConfig, PeftModel, get_peft_model
    _PEFT_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency.
    LoraConfig = None  # type: ignore[assignment]
    PeftModel = None  # type: ignore[assignment]
    get_peft_model = None  # type: ignore[assignment]
    _PEFT_AVAILABLE = False


@dataclass
class LoRAConfig:
    r: int = 8
    alpha: int = 16
    dropout: float = 0.0
    target_modules: List[str] = field(default_factory=list)


class LoRAManager:
    def __init__(self, mode: str = "lora", config: Optional[LoRAConfig] = None) -> None:
        self.mode = mode
        self.config = config or LoRAConfig()
        self.enabled = self.mode == "lora"

    def apply(self, model: Any) -> Any:
        if not _PEFT_AVAILABLE:
            raise RuntimeError("peft is required for LoRA support")
        if not self.enabled:
            return model
        lora_cfg = LoraConfig(
            r=self.config.r,
            lora_alpha=self.config.alpha,
            lora_dropout=self.config.dropout,
            target_modules=self.config.target_modules or None,
            bias="none",
            task_type="MASKED_LM",
        )
        return get_peft_model(model, lora_cfg)

    def merge_lora(self, model: Any) -> Any:
        if _PEFT_AVAILABLE and isinstance(model, PeftModel):
            return model.merge_and_unload()
        return model

    def save_lora(self, model: Any, path: str) -> None:
        if not _PEFT_AVAILABLE:
            raise RuntimeError("peft is required for LoRA support")
        if isinstance(model, PeftModel):
            model.save_pretrained(path)
        else:
            raise RuntimeError("LoRA is not enabled; no adapter to save.")

    def load_lora(self, model: Any, path: str) -> Any:
        if not _PEFT_AVAILABLE:
            raise RuntimeError("peft is required for LoRA support")
        return PeftModel.from_pretrained(model, path)

    def lora_state(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "config": self.config.__dict__,
            "enabled": self.enabled,
        }


def select_finetune_parameters(model: Any, mode: str) -> Iterable[Any]:
    if not hasattr(model, "parameters"):
        return []
    if mode == "full":
        return list(model.parameters())
    return [param for param in model.parameters() if getattr(param, "requires_grad", False)]
