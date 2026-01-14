"""MATH answer extraction and grading."""

from __future__ import annotations

import re
from typing import Dict, Optional


_BOXED_PATTERN = re.compile(r"\\\\boxed\{([^}]*)\}")


def _strip_latex(text: str) -> str:
    text = text.replace("$", "")
    text = text.replace("\\\\", "\\")
    return text.strip()


def extract_final_answer(text: str) -> str:
    boxed_match = _BOXED_PATTERN.search(text)
    if boxed_match:
        return boxed_match.group(1)
    text = text.strip()
    if text.endswith("}") and "\\boxed" in text:
        return text.split("\\boxed")[-1].strip("{}")
    return text


def _try_sympy_equiv(pred: str, target: str) -> Optional[bool]:
    try:
        import sympy
    except ImportError:
        return None
    try:
        pred_expr = sympy.simplify(sympy.sympify(pred))
        target_expr = sympy.simplify(sympy.sympify(target))
        diff = sympy.simplify(pred_expr - target_expr)
        return bool(diff == 0)
    except Exception:
        return None


def _numeric_equal(pred: str, target: str) -> Optional[bool]:
    try:
        return abs(float(pred) - float(target)) < 1e-4
    except Exception:
        return None


def grade_math_answer(prediction: str, metadata: Dict[str, object]) -> float:
    target = metadata.get("target_answer") if metadata else None
    if target is None:
        return 0.0
    pred = _strip_latex(extract_final_answer(prediction))
    target_str = _strip_latex(extract_final_answer(str(target)))

    sympy_equal = _try_sympy_equiv(pred, target_str)
    if sympy_equal is not None:
        return 1.0 if sympy_equal else 0.0
    numeric_equal = _numeric_equal(pred, target_str)
    if numeric_equal is not None:
        return 1.0 if numeric_equal else 0.0
    return 1.0 if pred == target_str else 0.0
