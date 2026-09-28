import pytest

from toolhunch import HeuristicTokenizer, Tokenizer


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", 0),
        ("Send an email to a user", 8),  # 23 bytes / 3 → 8 beats 6 pieces
        ("a-b-c", 5),  # 5 pieces (a - b - c) beat 5 bytes / 3 → 2
        ("get_diary_day", 5),
        ("明天北京的天气怎么样", 20),  # a non-ASCII character counts its UTF-8 bytes minus one: 2 here
        ("မြန်မာ", 12),  # Burmese: cl100k spends about two tokens per character
        ("👍", 3),  # four-byte characters count 3
        ("\ud800", 2),  # a lone surrogate (valid in JSON escapes) is counted, not an encoding error
        ("Sholltna_1 1st 1", 6),  # hash-like names tokenise badly: Sholltna _ 1 1 st 1
        ("1234567", 3),  # digits in groups of at most 3
        ("٣٤٥", 3),  # two-byte Unicode digits count once each, not also as ASCII digits
    ],
)
def test_heuristic_count(text: str, expected: int) -> None:
    tokenizer: Tokenizer = HeuristicTokenizer()
    assert tokenizer.count(text) == expected


def test_bytes_per_token_is_configurable() -> None:
    assert HeuristicTokenizer(bytes_per_token=4).count("Send an email to a user") == 6
