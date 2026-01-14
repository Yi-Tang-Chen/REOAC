"""Heuristics for locating math-relevant spans in text."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    kind: str


@dataclass
class TokenizationResult:
    tokens: List[str]
    offsets: List[Tuple[int, int]]


_NUMBER_PATTERN = re.compile(
    r"(?<![\w\.])"
    r"[+-]?"
    r"(?:\d+\.\d+|\d+|\d+/\d+)"
    r"%?"
    r"(?![\w\.])"
)

_UNIT_WORDS = [
    "dollar",
    "dollars",
    "usd",
    "cent",
    "cents",
    "kg",
    "kilogram",
    "kilograms",
    "g",
    "gram",
    "grams",
    "meter",
    "meters",
    "cm",
    "mm",
    "km",
    "hour",
    "hours",
    "minute",
    "minutes",
    "second",
    "seconds",
    "percent",
    "percentage",
    "each",
    "per",
    "rate",
]

_OPERATOR_WORDS = [
    "sum",
    "difference",
    "product",
    "ratio",
    "times",
    "total",
    "plus",
    "minus",
    "divide",
    "divided",
    "equal",
    "equals",
]

_LATEX_COMMANDS = [
    "frac",
    "sqrt",
    "boxed",
    "log",
    "sin",
    "cos",
    "tan",
    "cdot",
    "times",
    "div",
    "le",
    "ge",
]

_LATEX_COMMAND_PATTERN = re.compile(r"\\(" + "|".join(_LATEX_COMMANDS) + r")")

_OPERATOR_SYMBOL_PATTERN = re.compile(r"[+*/=<>&^_-]")


def extract_key_spans(text: str) -> List[Span]:
    spans: List[Span] = []
    for match in _NUMBER_PATTERN.finditer(text):
        spans.append(Span(match.start(), match.end(), "number"))

    for word in _UNIT_WORDS:
        for match in re.finditer(r"\b" + re.escape(word) + r"\b", text, re.IGNORECASE):
            spans.append(Span(match.start(), match.end(), "unit"))

    for word in _OPERATOR_WORDS:
        for match in re.finditer(r"\b" + re.escape(word) + r"\b", text, re.IGNORECASE):
            spans.append(Span(match.start(), match.end(), "operator_word"))

    for match in _LATEX_COMMAND_PATTERN.finditer(text):
        spans.append(Span(match.start(), match.end(), "latex"))

    for match in _OPERATOR_SYMBOL_PATTERN.finditer(text):
        spans.append(Span(match.start(), match.end(), "operator_symbol"))

    for match in re.finditer(r"\$\$|\$", text):
        spans.append(Span(match.start(), match.end(), "latex_delim"))

    return spans


def tokenize_with_offsets(text: str, tokenizer: Optional[object] = None) -> TokenizationResult:
    if tokenizer is not None:
        try:
            encoded = tokenizer(
                text,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
            offsets = [tuple(pair) for pair in encoded["offset_mapping"]]
            input_ids = encoded.get("input_ids")
            if hasattr(tokenizer, "convert_ids_to_tokens") and input_ids is not None:
                tokens = tokenizer.convert_ids_to_tokens(input_ids)
            else:
                tokens = [text[start:end] for start, end in offsets]
            return TokenizationResult(tokens=tokens, offsets=offsets)
        except TypeError:
            pass
        except Exception:
            pass

    tokens: List[str] = []
    offsets: List[Tuple[int, int]] = []
    for match in re.finditer(r"\S+", text):
        tokens.append(match.group(0))
        offsets.append((match.start(), match.end()))
    return TokenizationResult(tokens=tokens, offsets=offsets)


def spans_to_token_mask(spans: Iterable[Span], offsets: Sequence[Tuple[int, int]]) -> List[bool]:
    mask = [False for _ in offsets]
    for span in spans:
        for i, (start, end) in enumerate(offsets):
            if start < span.end and end > span.start:
                mask[i] = True
    return mask


def extract_key_token_mask(text: str, tokenizer: Optional[object] = None) -> Tuple[List[bool], TokenizationResult]:
    spans = extract_key_spans(text)
    tokenization = tokenize_with_offsets(text, tokenizer=tokenizer)
    mask = spans_to_token_mask(spans, tokenization.offsets)
    return mask, tokenization


def tokens_with_mask(tokens: Sequence[str], mask: Sequence[bool]) -> List[str]:
    return [token for token, keep in zip(tokens, mask) if keep]
