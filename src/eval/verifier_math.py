"""MATH answer extraction and grading."""

from __future__ import annotations

import re
from typing import Callable, Dict, Optional


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


def _parse_sympy_latex(expr: str):
    try:
        from sympy.parsing.latex import parse_latex
    except Exception:
        return None
    try:
        return parse_latex(expr)
    except Exception:
        return None


def _parse_latex2sympy2(expr: str):
    try:
        import latex2sympy2  # type: ignore[import-not-found]
    except Exception:
        return None
    for name in ("latex2sympy", "latex2sympy2"):
        parser = getattr(latex2sympy2, name, None)
        if parser is None:
            continue
        try:
            return parser(expr)
        except Exception:
            return None
    return None


def _parse_sympify(expr: str):
    try:
        import sympy
    except ImportError:
        return None
    try:
        return sympy.simplify(sympy.sympify(expr))
    except Exception:
        return None


def _normalize_math_parser(parser: Optional[str]) -> str:
    if not parser:
        return "sympy"
    return str(parser).strip().lower()


def _parse_expr(expr: str, parser: str):
    parser = _normalize_math_parser(parser)
    if parser in ("auto", "deepseek"):
        for name in ("latex2sympy2", "sympy_latex", "sympy"):
            result = _parse_expr(expr, name)
            if result is not None:
                return result
        return None
    if parser == "latex2sympy2":
        return _parse_latex2sympy2(expr)
    if parser in ("sympy_latex", "latex"):
        return _parse_sympy_latex(expr)
    return _parse_sympify(expr)


def _try_sympy_equiv(pred_expr, target_expr) -> Optional[bool]:
    try:
        import sympy
    except ImportError:
        return None
    try:
        diff = sympy.simplify(pred_expr - target_expr)
        return bool(diff == 0)
    except Exception:
        return None


def _numeric_equal(pred: str, target: str) -> Optional[bool]:
    try:
        return abs(float(pred) - float(target)) < 1e-4
    except Exception:
        return None


def _numeric_distance(pred: str, target: str) -> Optional[float]:
    try:
        return abs(float(pred) - float(target))
    except Exception:
        return None


def grade_math_answer(
    prediction: str,
    metadata: Dict[str, object],
    parser: Optional[str] = None,
    mode: str = "strict",
) -> float:
    target = metadata.get("target_answer") if metadata else None
    if target is None:
        return 0.0
    pred = _strip_latex(extract_final_answer(prediction))
    target_str = _strip_latex(extract_final_answer(str(target)))

    parser_name = _normalize_math_parser(parser)
    mode_name = str(mode or "strict").lower()
    pred_expr = _parse_expr(pred, parser_name)
    target_expr = _parse_expr(target_str, parser_name)
    if pred_expr is not None and target_expr is not None:
        sympy_equal = _try_sympy_equiv(pred_expr, target_expr)
        if sympy_equal is not None:
            return 1.0 if sympy_equal else 0.0
    numeric_equal = _numeric_equal(pred, target_str)
    if numeric_equal is not None:
        return 1.0 if numeric_equal else 0.0
    if mode_name != "strict":
        numeric_distance = _numeric_distance(pred, target_str)
        if numeric_distance is not None:
            try:
                denom = max(abs(float(target_str)), 1.0)
            except Exception:
                denom = max(abs(float(pred)) if pred else 1.0, 1.0)
            return max(0.0, 1.0 - (numeric_distance / denom))
        if pred and target_str and (pred in target_str or target_str in pred):
            return 0.1
        return 0.05 if pred else 0.0
    return 1.0 if pred == target_str else 0.0


def make_math_verifier(
    parser: Optional[str] = None,
    mode: str = "strict",
) -> Callable[[str, Dict[str, object]], float]:
    parser_name = _normalize_math_parser(parser)
    mode_name = str(mode or "strict").lower()

    def _verifier(prediction: str, metadata: Dict[str, object]) -> float:
        return grade_math_answer(prediction, metadata, parser=parser_name, mode=mode_name)

    return _verifier
