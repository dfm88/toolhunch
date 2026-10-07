"""The cost ledger: one JSON line per paid run, appended and never edited (`bench/results/cost-ledger.jsonl`)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import AwareDatetime, BaseModel, ConfigDict

from toolhunch_bench import BENCH_DIR

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

__all__ = [
    "BUDGET",
    "F2A_CAP_EUR",
    "LEDGER_PATH",
    "Budget",
    "F2aSpend",
    "LedgerEntry",
    "append_ledger",
    "f2a_spend",
    "read_ledger",
]

LEDGER_PATH = BENCH_DIR / "results" / "cost-ledger.jsonl"
F2A_CAP_EUR = 7.0
"""The most the F2a total may reach; a paid F2a run whose estimate would pass it stops. A dollar counts as a euro."""


@dataclass(frozen=True, slots=True)
class Budget:
    """A phase's spending limits, over the ledger lines whose purpose starts with `prefix`."""

    prefix: str
    target_usd: float
    cap_usd: float


BUDGET = Budget("P4:", 2.0, 3.0)
"""The current phase: P4, OpenAI's Decisions API in every test (spec 2026-10-07 §5.5). Paid runs stop before the
cap."""


class LedgerEntry(BaseModel):
    """What one run billed, as the provider reported it.

    `output_tokens` and `note` came later than the other fields: lines written without them read as 0 and empty.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    timestamp: AwareDatetime
    run_id: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int = 0
    usd: float
    purpose: str
    note: str = ""


@dataclass(frozen=True, slots=True)
class F2aSpend:
    """What the F2a lines of the ledger add up to: `usd` is the total the cap applies to."""

    usd: float
    modal_usd: float
    """F2a lines billed by provider `modal`: Modal credits, reported apart and not counted in `usd`."""


def append_ledger(entry: LedgerEntry, *, path: Path = LEDGER_PATH) -> None:
    """Append `entry` to the ledger at `path` as one JSON line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as ledger:
        ledger.write(entry.model_dump_json() + "\n")


def read_ledger(path: Path = LEDGER_PATH) -> list[LedgerEntry]:
    """The entries of the ledger at `path`, oldest first; a ledger that does not exist yet has none."""
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [LedgerEntry.model_validate_json(line) for line in lines if line.strip()]


def f2a_spend(entries: Iterable[LedgerEntry], *, purpose_prefix: str = "F2a") -> F2aSpend:
    """Add up lines beginning with `purpose_prefix`, preserving F2a's default and separate Modal accounting."""
    usd = modal_usd = 0.0
    for entry in entries:
        if not entry.purpose.startswith(purpose_prefix):
            continue
        if entry.provider == "modal":
            modal_usd += entry.usd
        else:
            usd += entry.usd
    return F2aSpend(usd=usd, modal_usd=modal_usd)
