"""GSM8K answer extraction and grading."""

from __future__ import annotations

import re
from fractions import Fraction
from typing import Dict, Optional, Tuple


_NUMBER_PATTERN = re.compile(r"[-+]?\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?")


def _parse_number(text: str) -> Optional[float]:
    text = text.strip()
    if not text:
        return None
    if "/" in text and not text.endswith("/"):
        try:
            return float(Fraction(text))
        except Exception:
            return None
    try:
        return float(text)
    except Exception:
        return None


def extract_final_answer(text: str) -> str:
    if "####" in text:
        text = text.split("####")[-1]
    matches = _NUMBER_PATTERN.findall(text)
    if matches:
        return matches[-1]
    return text.strip()


def grade_gsm8k_answer(prediction: str, metadata: Dict[str, object]) -> float:
    target = metadata.get("target_answer") if metadata else None
    if target is None:
        return 0.0
    pred = extract_final_answer(prediction)
    target_str = extract_final_answer(str(target))
    pred_num = _parse_number(pred)
    target_num = _parse_number(target_str)
    if pred_num is not None and target_num is not None:
        return 1.0 if abs(pred_num - target_num) < 1e-4 else 0.0
    return 1.0 if pred.strip() == target_str.strip() else 0.0
