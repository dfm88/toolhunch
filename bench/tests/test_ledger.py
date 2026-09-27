from datetime import UTC, datetime
from pathlib import Path

from toolhunch_bench.ledger import LedgerEntry, append_ledger


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
