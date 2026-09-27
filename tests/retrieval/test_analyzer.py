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
