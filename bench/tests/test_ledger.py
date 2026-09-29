from datetime import UTC, datetime
from pathlib import Path

import pytest

from toolhunch_bench.ledger import LedgerEntry, append_ledger, f2a_spend, read_ledger


def test_entries_are_appended_as_json_lines(tmp_path: Path) -> None:
    path = tmp_path / "results" / "cost-ledger.jsonl"
    entries = [
        LedgerEntry(
            timestamp=datetime(2026, 9, 27, 12, 0, tzinfo=UTC),
            run_id=f"20260927T1200{i}0Z",
            provider="openai",
            model="text-embedding-3-small",
            input_tokens=1_370_000 * i,
            usd=0.0274 * i,
            purpose="ToolRet pilot",
        )
        for i in (1, 2)
    ]

    for entry in entries:
        append_ledger(entry, path=path)

    assert [LedgerEntry.model_validate_json(line) for line in path.read_text().splitlines()] == entries


def test_f2a_spend_counts_f2a_lines_and_keeps_modal_credits_apart(tmp_path: Path) -> None:
    path = tmp_path / "cost-ledger.jsonl"
    assert read_ledger(path) == []  # a ledger that was never written is an empty one

    def entry(provider: str, usd: float, purpose: str) -> LedgerEntry:
        return LedgerEntry(
            timestamp=datetime(2026, 9, 29, 12, 0, tzinfo=UTC),
            run_id="20260929T120000Z",
            provider=provider,
            model="some-model",
            input_tokens=1_000,
            output_tokens=100,
            usd=usd,
            purpose=purpose,
        )

    # Written before the token split and the note existed: it still has to parse.
    path.write_text(
        '{"timestamp":"2026-09-28T15:52:29.085204Z","run_id":"20260928T154543Z","provider":"openai",'
        '"model":"text-embedding-3-small","input_tokens":1454448,"usd":0.5,'
        '"purpose":"ToolRet retrieval (dense, hybrid) on toolret-pilot-50.json"}\n'
    )
    for line in [
        entry("openai", 1.0, "F2a: model-written queries for toolret-pilot-50.json"),
        entry("typesafe", 0.25, "F2a verification: model ids, answer shapes, error shapes, usage"),
        entry("modal", 3.0, "F2a: Jev on Modal"),
        entry("modal", 9.0, "An unrelated Modal run"),
    ]:
        append_ledger(line, path=path)

    entries = read_ledger(path)
    spend = f2a_spend(entries)

    assert (entries[0].output_tokens, entries[0].note) == (0, "")
    assert (spend.usd, spend.modal_usd) == (pytest.approx(1.25), pytest.approx(3.0))  # pyright: ignore[reportUnknownMemberType]
