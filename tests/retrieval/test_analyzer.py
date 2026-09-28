import pytest

from toolhunch.retrieval import TextAnalyzer, s_stemmer


@pytest.mark.parametrize(
    ("text", "tokens"),
    [
        ("getDiaryDay", ["get", "diary", "day"]),
        ("get_diary_day", ["get", "diary", "day"]),
        ("HTTPServerError", ["http", "server", "error"]),
        ("Search the Web for news", ["search", "web", "news"]),
        ("base64_encode mp3", ["base64", "encode", "mp3"]),
        ("Qual è il meteo a Milano?", ["qual", "è", "il", "meteo", "milano"]),  # "a" is an English stop word
        ("नमस्ते मौसम", ["नमस्ते", "मौसम"]),  # combining vowel signs stay inside the word
        ("မြန်မာ ရာသီဥတု", ["မြန်မာ", "ရာသီဥတု"]),
        ("caffe\u0300 caff\u00e8", ["caff\u00e8", "caff\u00e8"]),  # decomposed and precomposed agree (NFKC)
        ("STRASSE Straße \uff21\uff30\uff29", ["strasse", "strasse", "api"]),  # case-folded; full-width API
        ("the of and ???", []),
        ("", []),
    ],
)
def test_default_analyzer(text: str, tokens: list[str]) -> None:
    assert TextAnalyzer().analyze(text) == tokens


@pytest.mark.parametrize(
    ("word", "stem"),
    [
        ("queries", "query"),
        ("emails", "email"),
        ("files", "file"),
        ("boxes", "boxe"),  # known S-stemmer quirk, harmless when both sides are stemmed alike
        ("class", "class"),
        ("status", "status"),
        ("gas", "gas"),
    ],
)
def test_s_stemmer(word: str, stem: str) -> None:
    assert s_stemmer(word) == stem


def test_stemmer_runs_after_stop_word_removal() -> None:
    assert TextAnalyzer(stemmer=s_stemmer).analyze("list the files") == ["list", "file"]
