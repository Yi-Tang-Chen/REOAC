"""Simple tokenizer for baseline runs and tests."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class TokenIds:
    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2
    mask_id: int = 3


class SimpleTokenizer:
    def __init__(self) -> None:
        self._ids = TokenIds()
        self._token_to_id: Dict[str, int] = {
            "<pad>": self._ids.pad_id,
            "<bos>": self._ids.bos_id,
            "<eos>": self._ids.eos_id,
            "<mask>": self._ids.mask_id,
        }
        self._id_to_token: Dict[int, str] = {v: k for k, v in self._token_to_id.items()}

    @property
    def pad_id(self) -> int:
        return self._ids.pad_id

    @property
    def bos_id(self) -> int:
        return self._ids.bos_id

    @property
    def eos_id(self) -> int:
        return self._ids.eos_id

    @property
    def mask_id(self) -> int:
        return self._ids.mask_id

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\S+", text)

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        tokens = self._tokenize(text)
        ids = [self._token_to_id.setdefault(token, len(self._token_to_id)) for token in tokens]
        if add_special_tokens:
            return [self._ids.bos_id] + ids + [self._ids.eos_id]
        return ids

    def decode(self, ids: List[int]) -> str:
        tokens = [self._id_to_token.get(idx, "<unk>") for idx in ids]
        return " ".join(tokens)

    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        return [self._id_to_token.get(idx, "<unk>") for idx in ids]

    def __call__(self, text: str, add_special_tokens: bool = False, return_offsets_mapping: bool = False):
        tokens = self._tokenize(text)
        ids = [self._token_to_id.setdefault(token, len(self._token_to_id)) for token in tokens]
        output = {"input_ids": ids}
        if return_offsets_mapping:
            offsets = []
            cursor = 0
            for token in tokens:
                match = re.search(re.escape(token), text[cursor:])
                if match is None:
                    offsets.append((cursor, cursor + len(token)))
                    cursor += len(token)
                else:
                    start = cursor + match.start()
                    end = start + len(token)
                    offsets.append((start, end))
                    cursor = end
            output["offset_mapping"] = offsets
        return output
