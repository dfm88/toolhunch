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
# are counted separately, from their UTF-8 length, so the classes here stay ASCII-only.
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
    three and ASCII symbols, plus each non-ASCII character's UTF-8 length minus one (one for
    Cyrillic or Arabic, two for most other scripts, three for emoji). The byte term dominates
    ordinary prose; the piece term catches hash-like identifiers and non-Latin scripts, where bytes
    alone undercount.

    Calibrated on 2026-09-28 against tiktoken `cl100k_base` and `o200k_base` over the 44,453 tool
    documents of ToolRet. Median overestimate: 1.36x on the raw JSON, 1.6-1.7x on the cards' search
    texts and on name + description. Single texts undercounted: 0.1 % of the raw JSON, 0.4-0.7 % of
    the shorter texts, worst by about half on hash-like names (`Sholltna_1 1st 1`: 6 against 11).
    No undercount on 20,000 random groups of 20 texts. Per script, on short sample sentences and
    cl100k: Burmese, Georgian, Tamil and Khmer 0.94-1.4x, emoji 1.25x; CJK and Thai are overcounted
    about 2x, and Hebrew (0.8x) and Ethiopic (0.67x) still undercount. Use it for budgets over
    several texts, not for exact single-text counts.

    Attributes:
        bytes_per_token: UTF-8 bytes assumed per token; lower is more conservative.
    """

    bytes_per_token: float = 3.0

    def count(self, text: str, /) -> int:
        """Return an upper-bound estimate of the tokens in `text`."""
        size = len(text.encode("utf-8", "surrogatepass"))
        # Each character adds its UTF-8 length minus one: 0 for ASCII, 1 to 3 for the rest.
        pieces = len(_ASCII_PIECE.findall(text)) + size - len(text)
        return max(math.ceil(size / self.bytes_per_token), pieces)
