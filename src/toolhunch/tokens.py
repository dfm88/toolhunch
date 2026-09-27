"""Token counting for budgets.

Core ships a conservative heuristic instead of an exact tokenizer: the limits that matter are
enforced on each provider's own tokenizer, which is often unpublished, so an upper bound is safer
than an exact count for the wrong vocabulary. Plug an exact counter through [`Tokenizer`][].
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Protocol

__all__ = ["HeuristicTokenizer", "Tokenizer"]

# ASCII letter runs, digit groups of at most three, and single ASCII symbols. Non-ASCII characters
# are counted separately, one each, so the classes here stay ASCII-only.
_ASCII_PIECE = re.compile(r"[A-Za-z]+|[0-9]{1,3}|[^\sA-Za-z0-9\x80-\U0010ffff]")


class Tokenizer(Protocol):
    """Counts the tokens a text takes in some model's context."""

    def count(self, text: str, /) -> int:
        """Return the number of tokens in `text`."""
        ...


@dataclass(frozen=True, slots=True)
class HeuristicTokenizer:
    """Conservative token estimate: `max(ceil(utf8_bytes / bytes_per_token), pieces)`.

    `pieces` counts runs that tokenizers rarely merge: ASCII letter runs, digit groups of up to
    three, ASCII symbols, and each non-ASCII character. The byte term dominates ordinary prose; the
    piece term catches hash-like identifiers and non-Latin scripts, where bytes alone undercount.

    Calibrated on 2026-09-27 against tiktoken `cl100k_base` and `o200k_base` over the 44,453 tool
    documents of ToolRet (name + description, and the raw JSON): median overestimate 1.44x;
    0.2-0.3 % of single texts undercounted; no undercount on 20,000 random groups of 20 texts.
    On the cards' search texts alone the median overestimate is 1.6x and 0.6-0.7 % of single texts
    are undercounted, worst by half on Burmese (three bytes per character, two tokens); groups of 20
    still never are. Use it for budgets over several texts, not for exact single-text counts.

    Attributes:
        bytes_per_token: UTF-8 bytes assumed per token; lower is more conservative.
    """

    bytes_per_token: float = 3.0

    def count(self, text: str, /) -> int:
        """Return an upper-bound estimate of the tokens in `text`."""
        non_ascii = len(text) - len(text.encode("ascii", "ignore"))
        pieces = len(_ASCII_PIECE.findall(text)) + non_ascii
        return max(math.ceil(len(text.encode("utf-8")) / self.bytes_per_token), pieces)
