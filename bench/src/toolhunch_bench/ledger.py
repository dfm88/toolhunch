"""The cost ledger: one JSON line per paid run, appended and never edited (`bench/results/cost-ledger.jsonl`)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import AwareDatetime, BaseModel, ConfigDict

from toolhunch_bench import BENCH_DIR

if TYPE_CHECKING:
    from pathlib import Path

__all__ = ["LEDGER_PATH", "LedgerEntry", "append_ledger"]

LEDGER_PATH = BENCH_DIR / "results" / "cost-ledger.jsonl"


class LedgerEntry(BaseModel):
    """What one run billed, as the provider reported it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    timestamp: AwareDatetime
    run_id: str
    provider: str
    model: str
    input_tokens: int
    usd: float
    purpose: str


def append_ledger(entry: LedgerEntry, *, path: Path = LEDGER_PATH) -> None:
    """Append `entry` to the ledger at `path` as one JSON line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as ledger:
        ledger.write(entry.model_dump_json() + "\n")
