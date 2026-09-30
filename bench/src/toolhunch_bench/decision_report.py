"""Reports of decision runs: abstention thresholds chosen on dev runs, then frozen and applied to a held-out run.

`build_decision_report` reads the run directories that `toolhunch-bench decision` writes and produces
`summary.json` and `README.md`. On dev it chooses one threshold per threshold key, sets the P@1 of every run side by
side (the ablations next to the main run) and measures how far repeated searches drift. On held-out, per arm and
query source, it reports P@1 against retrieval alone and the ceiling, the two abstention rules, latency, cost per
1,000 searches, errors and the model source's fallbacks; a second held-out run that repeats its searches adds how
much each arm varies from one repeat to the next. Both come with the Jev token check and the provenance of every
run. The CLM endpoint's host is published as `modal`.
"""

from __future__ import annotations

import contextlib
import functools
import json
import math
import statistics
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

from genai_prices import Usage, calc_price

from toolhunch_bench.metrics import (
    THRESHOLD_GRID,
    Outcome,
    choose_threshold,
    cluster_bootstrap_ci,
    linear_fit,
    percentile,
    selective_metrics,
    utility,
)

if TYPE_CHECKING:
    from collections.abc import Hashable, Iterable, Iterator, Sequence
    from pathlib import Path

__all__ = ["build_decision_report"]

BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 0
DETERMINISM_TOLERANCE = 0.01
"""The largest probability change between repeats for which one repetition of a search is enough."""
PUBLIC_CLM_HOST = "modal"
"""What the published files call the CLM endpoint's host: the deployment script documents it, its URL stays private."""
SEARCHED = "model (searched)"
"""The model source without the tasks whose writer made no search: those fell back to the plain request."""

type Record = dict[str, Any]


@dataclass(frozen=True, slots=True)
class _Run:
    """A run directory: its manifest, its search records, and its arm records by arm name."""

    manifest: dict[str, Any]
    searches: list[Record]
    arms: dict[str, Record]

    @property
    def run_id(self) -> str:
        return self.manifest["run_id"]


def build_decision_report(
    dev_runs: Sequence[Path], heldout_run: Path | None, *, out_dir: Path, heldout_repeats: Path | None = None
) -> None:
    """Choose the abstention thresholds on `dev_runs`, apply them to `heldout_run`, and write the report to `out_dir`.

    One threshold τ is chosen per threshold key on the dev runs that made each search once: the τ of the grid with
    the highest (correct - wrong) / searches, the lower one of equal scores. Held-out searches are then reported
    under the reserved option alone and under the reserved option plus the dev τ of their key; a held-out key without
    dev searches gets no τ. `summary.json` holds every figure and the runs' manifests, `README.md` the tables.

    A repeats run makes the held-out searches again, several times each with the decision cache bypassed. It never
    feeds the held-out tables: it adds, per arm and query source, how P@1 and U at the dev τ move between repeats and
    the largest change of one probability, next to the main run's P@1.

    Args:
        dev_runs: Run directories of the dev split: the main run, its ablations, reruns and the determinism run.
        heldout_run: The run directory of the held-out split; without one, the report covers dev alone.
        out_dir: Where `summary.json` and `README.md` go; created when missing.
        heldout_repeats: A held-out run that repeats its searches, on the tasks and the dataset of `heldout_run`.

    Raises:
        ValueError: A run's manifest names another split than the one it is given as. `heldout_repeats` comes without
            `heldout_run`, repeats fewer than twice, runs other tasks or another dataset than `heldout_run`, or has a
            decider arm that did not bypass the decision cache.
    """
    if heldout_repeats is not None and heldout_run is None:
        raise ValueError(f"{heldout_repeats} is a repeats run, given without a held-out run to compare it with")
    dev = [_read_run(path, split="dev") for path in dev_runs]
    heldout = None if heldout_run is None else _read_run(heldout_run, split="heldout")
    repeats = None if heldout_repeats is None or heldout is None else _read_repeats(heldout_repeats, main=heldout)
    thresholds = _thresholds(dev)
    fallbacks_in_tau = any(record["fallback"] is True for _, record in _tau_searches(dev))
    taus = {entry["key"]: entry["tau"] for entry in thresholds}
    runs = [*dev, *([] if heldout is None else [heldout]), *([] if repeats is None else [repeats])]
    summary: dict[str, Any] = {
        "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED, "level": 0.95, "unit": "task"},
        "threshold_grid": list(THRESHOLD_GRID),
        "runs": [_run_entry(run, role="repeats" if run is repeats else "main") for run in runs],
        "dev": {
            "thresholds": thresholds,
            "precision": _dev_precision(dev),
            "determinism": _determinism(dev),
            "errors": [_errors(run, arm) for run in dev for arm in run.manifest["arms"]],
        },
        "heldout": None if heldout is None else _heldout(heldout, taus, repeats=repeats),
        "tokenizer": [_tokenizer("dev", dev), *([] if heldout is None else [_tokenizer("heldout", [heldout])])],
    }
    hosts = _clm_hosts(runs)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(_published(json.dumps(summary, indent=2) + "\n", hosts), encoding="utf-8")
    (out_dir / "README.md").write_text(
        _published(_markdown(summary, fallbacks_in_tau=fallbacks_in_tau), hosts), encoding="utf-8"
    )


def _read_run(run_dir: Path, *, split: str) -> _Run:
    manifest: dict[str, Any] = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest["split"] != split:
        raise ValueError(f"{run_dir} holds a {manifest['split']} run, given as a {split} run")
    lines = (run_dir / "run.jsonl").read_text(encoding="utf-8").splitlines()
    records: list[Record] = [json.loads(line) for line in lines if line.strip()]
    searches = [record for record in records if record["record"] == "search"]
    return _Run(manifest, searches, {record["arm"]: record for record in records if record["record"] == "arm"})


def _read_repeats(run_dir: Path, *, main: _Run) -> _Run:
    """The repeats run of `run_dir`, checked to be one that can stand next to the `main` held-out run."""
    run = _read_run(run_dir, split="heldout")
    manifest, reference = run.manifest, main.manifest
    if manifest["repeat"] < 2:
        raise ValueError(f"{run_dir} makes each search {manifest['repeat']} time, not a repeats run")
    if manifest["tasks"]["sha256"] != reference["tasks"]["sha256"] or manifest["dataset"] != reference["dataset"]:
        raise ValueError(f"{run_dir} runs other tasks or another dataset than the held-out run {main.run_id}")
    for name, config in manifest["arms"].items():
        # The manifest keeps the bypass only in the repr of the arm's model.
        if config["decider"] is not None and "bypass=True" not in (config["model"] or ""):
            raise ValueError(
                f"{run_dir}: arm {name} did not bypass the decision cache, so its repeats replay one answer"
            )
    return run


def _outcome(record: Record, *, tau: float) -> Outcome:
    """What the search's arm did at threshold `tau`, with the reserved option when the search asked it.

    Retrieval alone answers every search with its first candidate. A decider abstains when the reserved option is
    at least as likely as its first card, or when that card's probability is below `tau`; a decision that asked
    nothing keeps the abstention it recorded.
    """
    relevant = record["relevant"]
    if record["decider"] is None:
        return Outcome(True, _retrieval_right(record), record["gold_in_candidates"])
    ranked, probabilities = record["ranked"], record["probabilities"]
    if probabilities:
        best = probabilities[ranked[0]]
        reserved = record["none_probability"]
        answered = not (reserved is not None and reserved >= best) and not best < tau
    else:
        answered = not record["abstained"]
    return Outcome(answered, answered and bool(ranked) and ranked[0] in relevant, record["gold_in_candidates"])


def _outcomes(records: Sequence[Record], tau: float) -> list[Outcome]:
    return [_outcome(record, tau=tau) for record in records]


def _counts(outcomes: Sequence[Outcome]) -> dict[str, int]:
    correct = sum(outcome.correct for outcome in outcomes)
    answered = sum(outcome.answered for outcome in outcomes)
    return {"correct": correct, "wrong": answered - correct, "abstained": len(outcomes) - answered}


def _ok(records: Iterable[Record]) -> list[Record]:
    """The searches that did not fail: a failed search counts as an error and stays out of every rate."""
    return [record for record in records if record["error"] is None]


def _decided(records: Iterable[Record]) -> list[Record]:
    """The searches a decider answered, abstentions included."""
    return [record for record in _ok(records) if record["decider"] is not None]


def _first_right(record: Record) -> bool:
    """Whether the first card of the search's ranking is a gold tool; retrieval alone ranks by retrieval."""
    ranked = record["ranked"]
    return bool(ranked) and ranked[0] in record["relevant"]


def _retrieval_right(record: Record) -> bool:
    """Whether retrieval ranked a gold tool first among the search's candidates."""
    candidates = record["candidates"]
    return bool(candidates) and candidates[0] in record["relevant"]


def _share(flags: Sequence[bool]) -> float | None:
    return sum(flags) / len(flags) if flags else None


def _unique[T: Hashable](items: Iterable[T]) -> list[T]:
    return list(dict.fromkeys(items))


def _sources(manifest: dict[str, Any]) -> list[str]:
    """The query sources of a run, then the model source without its fallbacks."""
    sources = list(manifest["sources"])
    return [*sources, SEARCHED] if "model" in sources else sources


def _in_source(record: Record, source: str) -> bool:
    if source == SEARCHED:
        return record["source"] == "model" and record["fallback"] is False
    return record["source"] == source


def _tau_searches(runs: Sequence[_Run]) -> Iterator[tuple[_Run, Record]]:
    """The decided searches of the dev runs that made each search once, each with the run it is taken from.

    A search that several of those runs made, such as a rerun of the same settings, counts once, from the first run.
    """
    seen: set[tuple[str, str, str, str]] = set()
    for run in runs:
        if run.manifest["repeat"] != 1:
            continue
        for record in _decided(run.searches):
            search = (record["key"], record["source"], record["task"], record["variant"])
            if search not in seen:
                seen.add(search)
                yield run, record


def _thresholds(runs: Sequence[_Run]) -> list[dict[str, Any]]:
    """One τ per threshold key, chosen on the dev runs that made each search once, sources and variants pooled."""
    pooled: dict[str, list[Record]] = {}
    contributors: dict[str, list[str]] = {}
    for run, record in _tau_searches(runs):
        pooled.setdefault(record["key"], []).append(record)
        contributors.setdefault(record["key"], [])
        if run.run_id not in contributors[record["key"]]:
            contributors[record["key"]].append(run.run_id)
    return [
        {
            "key": key,
            "decider": records[0]["decider"],
            "k": records[0]["k"],
            "shape": records[0]["shape"],
            "arms": _unique(record["arm"] for record in records),
            "runs": contributors[key],
            "searches": len(records),
            "positives": sum(record["variant"] == "positive" for record in records),
            "negatives": sum(record["variant"] == "negative" for record in records),
            "tau": choose_threshold(functools.partial(_outcomes, records)),
            "grid": _grid(records),
        }
        for key, records in pooled.items()
    ]


def _grid(records: Sequence[Record]) -> list[dict[str, Any]]:
    """Each threshold of the grid applied to every one of `records`: coverage, the other rates and U."""
    points: list[dict[str, Any]] = []
    for tau in THRESHOLD_GRID:
        outcomes = _outcomes(records, tau)
        points.append(
            {"tau": tau, **asdict(selective_metrics(outcomes)), **_counts(outcomes), "utility": utility(outcomes)}
        )
    return points


def _dev_precision(runs: Sequence[_Run]) -> list[dict[str, Any]]:
    """P@1 of every arm and source of each dev run that made each search once."""
    entries: list[dict[str, Any]] = []
    for run in runs:
        if run.manifest["repeat"] != 1:
            continue
        for arm in run.manifest["arms"]:
            for source in _sources(run.manifest):
                positives = [
                    record
                    for record in _ok(run.searches)
                    if record["arm"] == arm and record["variant"] == "positive" and _in_source(record, source)
                ]
                if positives:
                    p_at_1 = _share([_first_right(record) for record in positives])
                    entries.append(
                        {
                            "run_id": run.run_id,
                            "arm": arm,
                            "source": source,
                            "positives": len(positives),
                            "p_at_1": p_at_1,
                        }
                    )
    return entries


def _determinism(runs: Sequence[_Run]) -> list[dict[str, Any]]:
    """For each decider arm of the runs that repeat their searches, the largest probability change between repeats."""
    entries: list[dict[str, Any]] = []
    for run in runs:
        if run.manifest["repeat"] < 2:
            continue
        repeats: dict[tuple[str, str, str, str], list[Record]] = {}
        for record in _decided(run.searches):
            repeats.setdefault((record["arm"], record["source"], record["task"], record["variant"]), []).append(record)
        for arm in run.manifest["arms"]:
            changes = [
                (_largest_change(found), search)
                for search, found in repeats.items()
                if search[0] == arm and len(found) > 1
            ]
            if not changes:
                continue
            change, (_, source, task, variant) = max(changes, key=lambda item: item[0])
            entries.append(
                {
                    "run_id": run.run_id,
                    "arm": arm,
                    "repeat": run.manifest["repeat"],
                    "compared": len(changes),
                    "max_abs_delta_p": change,
                    "at": {"source": source, "task": task, "variant": variant},
                    # Two-decimal probabilities 0.01 apart differ by 0.010000000000000009 in floating point.
                    "within_tolerance": change <= DETERMINISM_TOLERANCE or math.isclose(change, DETERMINISM_TOLERANCE),
                }
            )
    return entries


def _largest_change(repeats: Sequence[Record]) -> float:
    """The largest spread of one probability over the repeats of a search: a card's, or the reserved option's.

    A card left out of a repeat's final question counts as 0.0 there, where that repeat ranks it.
    """
    cards = _unique(card for record in repeats for card in record["probabilities"])
    spreads = [_spread([record["probabilities"].get(card, 0.0) for record in repeats]) for card in cards]
    reserved = [record["none_probability"] for record in repeats]
    if None not in reserved:
        spreads.append(_spread(reserved))
    return max(spreads, default=0.0)


def _spread(values: Sequence[float]) -> float:
    return max(values) - min(values)


def _errors(run: _Run, arm: str) -> dict[str, Any]:
    own = [record for record in run.searches if record["arm"] == arm]
    return {"run_id": run.run_id, "arm": arm, "searches": len(own), "errors": len(own) - len(_ok(own))}


def _heldout(run: _Run, taus: dict[str, float], *, repeats: _Run | None) -> dict[str, Any]:
    manifest = run.manifest
    without_tau: dict[str, list[Record]] = {}
    for record in _decided(run.searches):
        if record["key"] not in taus:
            without_tau.setdefault(record["key"], []).append(record)
    rows = [
        row
        for arm in manifest["arms"]
        for source in _sources(manifest)
        if (row := _row(run, arm=arm, source=source, taus=taus)) is not None
    ]
    return {
        "run_id": run.run_id,
        "keys_without_dev_tau": [
            {"key": key, "arms": _unique(record["arm"] for record in records), "searches": len(records)}
            for key, records in without_tau.items()
        ],
        "rows": rows,
        "arms": [_arm_entry(run, arm) for arm in manifest["arms"]],
        "risk_coverage": [curve for arm in manifest["arms"] if (curve := _curve(run, arm=arm, taus=taus)) is not None],
        "repeats": None if repeats is None else _variation(repeats, taus=taus, main_rows=rows),
    }


def _variation(run: _Run, *, taus: dict[str, float], main_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """How far each arm and query source of the repeats `run` moves between the repeats of its searches.

    `main_rows` are the held-out rows of the main run, for the P@1 it measured from one repetition of each search.
    """
    main = {(row["arm"], row["source"]): row["p_at_1"] for row in main_rows}
    rows = [
        row
        for arm in run.manifest["arms"]
        for source in _sources(run.manifest)
        if (row := _repeat_row(run, arm=arm, source=source, taus=taus, main=main)) is not None
    ]
    return {"run_id": run.run_id, "repeat": run.manifest["repeat"], "rows": rows}


def _repeat_row(
    run: _Run, *, arm: str, source: str, taus: dict[str, float], main: dict[tuple[str, str], float | None]
) -> dict[str, Any] | None:
    """The figures of one arm and query source, repeat by repeat; `None` when the arm made no search from it.

    `main` holds the P@1 of the main run by arm and source. A failed search is left out, so a repeat in which every
    search of the arm failed has no entry.
    """
    records = [record for record in run.searches if record["arm"] == arm and _in_source(record, source)]
    if not records:
        return None
    config = run.manifest["arms"][arm]
    ok = _ok(records)
    by_repeat: dict[int, list[Record]] = {}
    for record in ok:
        by_repeat.setdefault(record["repeat"], []).append(record)
    per_repeat: list[dict[str, Any]] = []
    for index, found in sorted(by_repeat.items()):
        outcomes = [_outcome(record, tau=tau) for record, tau in zip(found, _dev_taus(found, taus), strict=True)]
        metrics = selective_metrics(outcomes)
        positives = [record for record in found if record["variant"] == "positive"]
        per_repeat.append(
            {
                "repeat": index,
                "p_at_1": _share([_first_right(record) for record in positives]),
                "reserved_and_dev_tau": {
                    **_counts(outcomes),
                    "coverage": metrics.coverage,
                    "wrong_tool_rate": metrics.wrong_tool_rate,
                    "utility": utility(outcomes),
                },
            }
        )
    searches: dict[tuple[str, str], list[Record]] = {}
    for record in _decided(ok):
        searches.setdefault((record["task"], record["variant"]), []).append(record)
    changes = [(_largest_change(found), search) for search, found in searches.items() if len(found) > 1]
    largest: float | None = None
    at: dict[str, str] | None = None
    if changes:
        largest, (task, variant) = max(changes, key=lambda item: item[0])
        at = {"task": task, "variant": variant}
    return {
        "arm": arm,
        "decider": config["decider"],
        "k": config["k"],
        "source": source,
        "searches": _search_count(ok),
        "records": len(ok),
        "errors": len(records) - len(ok),
        "per_repeat": per_repeat,
        "p_at_1_range": _range([entry["p_at_1"] for entry in per_repeat]),
        "utility_range": _range([entry["reserved_and_dev_tau"]["utility"] for entry in per_repeat]),
        "largest_abs_dp": largest,
        "at": at,
        "main_p_at_1": main.get((arm, source)),
    }


def _range(values: Sequence[float | None]) -> list[float] | None:
    """[min, max] of the values present; `None` when there is none."""
    present = [value for value in values if value is not None]
    return [min(present), max(present)] if present else None


def _row(run: _Run, *, arm: str, source: str, taus: dict[str, float]) -> dict[str, Any] | None:
    """The held-out figures of one arm and query source; `None` when the arm made no search from it."""
    records = [record for record in run.searches if record["arm"] == arm and _in_source(record, source)]
    if not records:
        return None
    config = run.manifest["arms"][arm]
    ok = _ok(records)
    positives = [record for record in ok if record["variant"] == "positive"]
    decided = [record for record in ok if record["decider"] is not None]
    applied = [taus[record["key"]] for record in decided if record["key"] in taus]
    reserved_and_tau = _rule(ok, _dev_taus(ok, taus))
    differences = [float(_first_right(record)) - float(_retrieval_right(record)) for record in positives]
    paired = config["decider"] is not None and bool(differences)
    return {
        "arm": arm,
        "decider": config["decider"],
        "k": config["k"],
        "source": source,
        "searches": _search_count(ok),
        "records": len(ok),
        "errors": len(records) - len(ok),
        "fallbacks": sum(record["fallback"] is True for record in ok) if source in ("model", SEARCHED) else None,
        "positives": len(positives),
        "negatives": sum(record["variant"] == "negative" for record in ok),
        "p_at_1": _share([_first_right(record) for record in positives]),
        "p_at_1_ci95": _interval(positives, [float(_first_right(record)) for record in positives]),
        "hybrid_p_at_1": _share([_retrieval_right(record) for record in positives]),
        "delta_p_at_1": statistics.fmean(differences) if paired else None,
        "delta_p_at_1_ci95": _interval(positives, differences) if paired else None,
        "ceiling": _share([record["gold_in_candidates"] for record in positives]),
        "rules": {
            "answer_always": _rule(ok, [0.0] * len(ok), answer_always=True),
            "reserved": _rule(ok, [0.0] * len(ok)),
            "reserved_and_dev_tau": None
            if reserved_and_tau is None
            else reserved_and_tau | {"taus": sorted(set(applied)), "without_dev_tau": len(decided) - len(applied)},
        },
        "risk_coverage": {"grid": _grid(decided), "dev_taus": sorted(set(applied))} if decided else None,
        "latency_ms": {
            "decision": _percentiles([record["decision_seconds"] for record in ok]),
            "server": _percentiles([record["server_seconds"] for record in ok]),
            "retrieval": _percentiles([record["retrieval_seconds"] for record in ok]),
        },
        "cost": _cost(ok, config=config, manifest=run.manifest),
    }


def _dev_taus(records: Sequence[Record], taus: dict[str, float]) -> list[float]:
    """The threshold each record is held to: the dev τ of its key, 0.0 without one and for retrieval alone."""
    return [taus.get(record["key"], 0.0) if record["decider"] is not None else 0.0 for record in records]


def _search_count(records: Iterable[Record]) -> int:
    """How many searches `records` make: a search that the run repeats counts once."""
    return len({(record["source"], record["task"], record["variant"]) for record in records})


def _rule(
    records: Sequence[Record], thresholds: Sequence[float], *, answer_always: bool = False
) -> dict[str, Any] | None:
    """The abstention metrics of `records`, each at its own threshold, with an interval for the wrong-tool rate."""
    if not records:
        return None
    outcomes = (
        [Outcome(True, _first_right(record), record["gold_in_candidates"]) for record in records]
        if answer_always
        else [_outcome(record, tau=tau) for record, tau in zip(records, thresholds, strict=True)]
    )
    wrong = [float(outcome.answered and not outcome.correct) for outcome in outcomes]
    return asdict(selective_metrics(outcomes)) | _counts(outcomes) | {"wrong_tool_rate_ci95": _interval(records, wrong)}


def _interval(records: Sequence[Record], values: Sequence[float]) -> list[float] | None:
    """The 95% bootstrap interval of the mean of `values`, one per record, resampling tasks.

    A task's positive search, its negative and all their repeats are drawn together: when the gold tool is not among
    the candidates the negative is the positive again, so the two are not independent.
    """
    if not records:
        return None
    tasks: dict[str, list[float]] = {}
    for record, value in zip(records, values, strict=True):
        tasks.setdefault(record["task"], []).append(value)
    low, high = cluster_bootstrap_ci(list(tasks.values()), resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED)
    return [low, high]


def _percentiles(seconds: Sequence[float | None]) -> dict[str, float] | None:
    """p50 and p95 in milliseconds of the values present; `None` when there is none."""
    present = [value * 1000 for value in seconds if value is not None]
    if not present:
        return None
    return {"p50": percentile(present, 50), "p95": percentile(present, 95)}


def _cost(records: Sequence[Record], *, config: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any] | None:
    """What deciding 1,000 searches like `records` costs, and the asks and tokens per search.

    Jev is priced at the input and output prices its declared limits give, the logprob model by genai-prices from
    the usage its provider reported. CLM runs on a GPU billed by the hour: busy prices the server's own seconds per
    search, wall the seconds of the calls as the client saw them. Replays from the decision cache keep the usage and
    the latency of the call they replay, so the figures do not depend on what the cache held.
    """
    if not records:
        return None
    count = len(records)
    usage = [record["usage"] for record in records if record["usage"] is not None]
    input_tokens = sum(item["input_tokens"] for item in usage)
    output_tokens = sum(item["output_tokens"] for item in usage)
    usd: float | None = None
    busy: float | None = None
    wall: float | None = None
    match config["decider"]:
        case None:
            usd = 0.0
        case "jev":
            limits = config["limits"]
            if limits["price_input_per_mtok"] is not None:
                output_price = limits["price_output_per_mtok"] or 0.0
                usd = (input_tokens * limits["price_input_per_mtok"] + output_tokens * output_price) / 1_000_000
        case "logprob" | "luna":
            model = config["model_id"].partition("@")[0]
            with contextlib.suppress(LookupError):  # a model genai-prices does not know stays unpriced
                # Per search: price tiers (gpt-6-luna: 2x above 272K input tokens) apply to one request.
                usd = float(
                    sum(
                        calc_price(
                            Usage(input_tokens=item["input_tokens"], output_tokens=item["output_tokens"]),
                            model_ref=model,
                            provider_id="openai",
                        ).total_price
                        for item in usage
                    )
                )
        case "clm":
            per_second = manifest["clm_deployment"]["usd_per_gpu_hour"] / 3600
            server = [record["server_seconds"] for record in records if record["server_seconds"] is not None]
            if server:
                busy = statistics.fmean(server) * per_second * 1000
            wall = sum(record["decision_seconds"] for record in records) / count * per_second * 1000
        case _:
            pass
    return {
        "asks_per_search": sum(item["requests"] for item in usage) / count,
        "input_tokens_per_search": input_tokens / count,
        "output_tokens_per_search": output_tokens / count,
        "usd_per_1000_searches": None if usd is None else usd / count * 1000,
        "clm_busy_usd_per_1000_searches": busy,
        "clm_wall_usd_per_1000_searches": wall,
    }


def _arm_entry(run: _Run, arm: str) -> dict[str, Any]:
    """An arm's searches, records (one per repeat) and errors, the summed seconds of its decisions, clock and cache."""
    config = run.manifest["arms"][arm]
    own = [record for record in run.searches if record["arm"] == arm]
    decided = _decided(own)
    record = run.arms.get(arm, {})  # missing when the run stopped inside this arm
    return {
        "arm": arm,
        "decider": config["decider"],
        "k": config["k"],
        "searches": _search_count(own),
        "records": len(own),
        "errors": len(own) - len(_ok(own)),
        "decision_seconds": sum(search["decision_seconds"] for search in decided) if decided else None,
        "wall_seconds": record.get("wall_seconds"),
        "cache_hits": record.get("cache_hits"),
        "cache_misses": record.get("cache_misses"),
    }


def _curve(run: _Run, *, arm: str, taus: dict[str, float]) -> dict[str, Any] | None:
    """The risk-coverage table of a decider arm over the grid, all its searches together, and the dev τ it got."""
    decided = [record for record in _decided(run.searches) if record["arm"] == arm]
    if not decided:
        return None
    applied = {taus.get(record["key"]) for record in decided}
    dev_tau = next(iter(applied)) if len(applied) == 1 else None
    return {
        "arm": arm,
        "dev_tau": dev_tau,
        "grid": _grid(decided),
        "negative_share": sum(record["variant"] == "negative" for record in decided) / len(decided),
    }


def _tokenizer(split: str, runs: Sequence[_Run]) -> dict[str, Any]:
    """The token heuristic against Jev's reported input tokens, one point per exchange with a reported count."""
    points = [
        (exchange["estimated_input_tokens"], exchange["input_tokens"])
        for run in runs
        for record in _decided(run.searches)
        if record["decider"] == "jev"
        for exchange in record["exchanges"]
        if exchange["estimated_input_tokens"] > 0 and exchange["input_tokens"] > 0
    ]
    entry: dict[str, Any] = {
        "split": split,
        "exchanges": len(points),
        "intercept": None,
        "slope": None,
        "median_ratio": None,
        "share_under_1": None,
    }
    if points:
        ratios = [reported / estimated for estimated, reported in points]
        entry["median_ratio"] = statistics.median(ratios)
        entry["share_under_1"] = sum(ratio < 1 for ratio in ratios) / len(ratios)
        with contextlib.suppress(ValueError):  # fewer than two points, or a single estimate: no line
            entry["intercept"], entry["slope"] = linear_fit(
                [estimated for estimated, _ in points], [reported for _, reported in points]
            )
    return entry


def _run_entry(run: _Run, *, role: str) -> dict[str, Any]:
    """A run's manifest and role, and how many tasks of its model source fell back to the plain request."""
    model = [record for record in run.searches if record["source"] == "model"]
    return {
        "split": run.manifest["split"],
        "role": role,
        "run_id": run.run_id,
        "model_tasks": len({record["task"] for record in model}) if model else None,
        "fallback_tasks": len({record["task"] for record in model if record["fallback"]}) if model else None,
        "manifest": run.manifest,
    }


def _clm_hosts(runs: Sequence[_Run]) -> list[str]:
    """The host of every CLM endpoint the runs asked, longest first."""
    hosts = {
        config["model_id"].partition("@")[2]
        for run in runs
        for config in run.manifest["arms"].values()
        if config["decider"] == "clm" and config["model_id"]
    }
    return sorted((host for host in hosts if host), key=lambda host: (-len(host), host))


def _published(text: str, hosts: Sequence[str]) -> str:
    """`text` with every CLM host, in model ids, threshold keys, URLs and errors alike, replaced by `modal`."""
    for host in hosts:
        text = text.replace(host, PUBLIC_CLM_HOST)
    return text


def published_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Normalize a raw manifest using the exact F2a publication sanitizer, retaining every field."""
    run = _Run(manifest=manifest, searches=[], arms={})
    return json.loads(_published(json.dumps(manifest), _clm_hosts([run])))


# Markdown


def _markdown(summary: dict[str, Any], *, fallbacks_in_tau: bool) -> str:
    heldout = summary["heldout"]
    deployments: list[dict[str, Any]] = []
    for entry in summary["runs"]:
        if (deployment := entry["manifest"]["clm_deployment"]) is not None and deployment not in deployments:
            deployments.append(deployment)
    title = "dev runs" if heldout is None else f"held-out run {heldout['run_id']}"
    lines = [f"# ToolRet decision stage, {title}", "", _intro(summary), ""]
    if deployments:
        lines += [
            "**CLM caveat.** We run CLM ourselves, from the code its authors published "
            f"(`{deployments[0]['script']}`). Its figures are provisional until the CLM authors confirm that this "
            "deployment matches their reference setup.",
            "",
        ]
    lines += _runs_section(summary["runs"], deployments=deployments)
    if heldout is not None:
        lines += _heldout_section(heldout, runs=summary["runs"])
    lines += [
        *_dev_section(summary["dev"]),
        *_tokenizer_section(summary["tokenizer"]),
        *_notes(fallbacks_in_tau=fallbacks_in_tau),
    ]
    return "\n".join(
        [*lines, *_risk_coverage_section(summary), ""]
    )


def _intro(summary: dict[str, Any]) -> str:
    heldout = summary["heldout"]
    dev_runs = ", ".join(f"`{entry['run_id']}`" for entry in summary["runs"] if entry["split"] == "dev")
    if heldout is None:
        return (
            f"Dev runs only ({dev_runs}): the thresholds, the P@1 of the ablations and the determinism check below "
            "settle the settings of the held-out run. None of these numbers is a held-out result."
        )
    manifest = _main_heldout(summary["runs"])["manifest"]
    dataset, tasks = manifest["dataset"], manifest["tasks"]
    return (
        f"Held-out run `{heldout['run_id']}`: {tasks['count']} tasks from `{tasks['file']}` (sha256 "
        f"`{tasks['sha256'][:12]}`), `{dataset['name']}` at `{dataset['revision'][:7]}` with {dataset['tools']:,} "
        f"tools, catalog `{dataset['catalog_fingerprint'][:19]}`; toolhunch {manifest['versions']['toolhunch']}, "
        f"commit `{_commit(manifest['git'])}`, Python {manifest['versions']['python']}. Every abstention threshold "
        f"applied here was chosen on the dev runs ({dev_runs}); no held-out search chose one."
    )


def _main_heldout(runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """The entry of the held-out run the tables come from, not the one that repeats its searches."""
    [entry] = [entry for entry in runs if entry["split"] == "heldout" and entry["role"] == "main"]
    return entry


def _commit(git: dict[str, Any]) -> str:
    return (git["commit"] or "unknown")[:7] + (" (dirty)" if git["dirty"] else "")


def _runs_section(runs: Sequence[dict[str, Any]], *, deployments: Sequence[dict[str, Any]]) -> list[str]:
    rows: list[list[str]] = []
    models: list[tuple[str, str, str, str]] = []
    for entry in runs:
        manifest, tasks = entry["manifest"], entry["manifest"]["tasks"]
        deciders = [config for config in manifest["arms"].values() if config["decider"] is not None]
        fallbacks = "-" if entry["model_tasks"] is None else f"{entry['fallback_tasks']} of {entry['model_tasks']}"
        rows.append(
            [
                entry["split"],
                entry["role"],
                f"`{entry['run_id']}`",
                f"{tasks['count']} (`{tasks['file']}`, sha256 `{tasks['sha256'][:12]}`)",
                ", ".join(manifest["sources"]),
                "yes" if manifest["negatives"] else "no",
                str(manifest["repeat"]),
                ", ".join(_unique(str(config["k"]) for config in manifest["arms"].values())),
                ", ".join(_unique(f"{config['decider']} ({config['max_detail']})" for config in deciders)) or "none",
                ", ".join(_unique("on" if config["reserved_option"] else "off" for config in deciders)) or "-",
                fallbacks,
                f"`{_commit(manifest['git'])}`",
            ]
        )
        for config in deciders:
            declared = f"{config['limits']['source']} (checked {config['limits']['checked']})"
            models.append((config["decider"], f"`{config['model_id']}`", f"`{config['prompt_version']}`", declared))
    header = ["split", "role", "run", "tasks", "sources", "negatives", "repeat", "K", "deciders (max detail)"]
    header += ["reserved option", "model-source fallbacks", "commit"]
    lines = ["## Runs", "", *_table(header, rows, align="l" * len(header)), ""]
    if models:
        lines += [*_table(["decider", "model", "prompt version", "declared limits"], _unique(models), align="llll"), ""]
    for deployment in deployments:
        lines += [
            f"CLM runs from `{deployment['script']}` on one {deployment['gpu']} GPU (vLLM {deployment['vllm']}, "
            f"contrastive-lm {deployment['contrastive_lm']}), listed at ${deployment['usd_per_gpu_hour']:.2f} per "
            f"GPU hour (checked {deployment['price_checked']}).",
            "",
        ]
    for entry in runs:
        if (queries := entry["manifest"]["model_queries"]) is not None:
            lines += [
                f"Model-written queries of `{entry['run_id']}`: `{queries['file']}` "
                f"(sha256 `{queries['sha256'][:12]}`), written by `{queries['writer'].get('model')}`.",
                "",
            ]
    return lines


def _heldout_section(heldout: dict[str, Any], *, runs: Sequence[dict[str, Any]]) -> list[str]:
    rows = heldout["rows"]
    entry = _main_heldout(runs)
    lines = ["## Held-out", ""]
    if entry["model_tasks"] is not None:
        lines += [
            "`plain` searches with the ToolRet request, `model` with the queries a model wrote for the task, and "
            f"`{SEARCHED}` keeps the tasks where that model did search: on {entry['fallback_tasks']} of "
            f"{entry['model_tasks']} tasks it answered without searching, and the request itself was used.",
            "",
        ]
    header = ["arm", "source", "positives", "P@1 (95% CI)", "hybrid P@1", "Δ vs hybrid (95% CI)", "ceiling"]
    lines += [
        "### P@1",
        "",
        "P@1 measures a relevant tool first on positive requests, ignoring abstention; "
        "it does not measure task completion.",
        "",
        *_table(header, [_precision_row(row) for row in rows], align="llrrrrr"),
        "",
    ]
    header = ["arm", "source", "rule", "τ", "searches", "errors", "coverage", "selective accuracy"]
    header += [
        "correct",
        "wrong",
        "abstained",
        "negatives",
        "wrong-tool rate (95% CI)",
        "abstention precision",
        "abstention recall",
    ]
    abstention = [cells for row in rows for cells in _abstention_rows(row)]
    lines += [
        "### Abstention",
        "",
        "Positive and negative requests are pooled below; the negative share is shown beside every row.",
        "",
        *_table(header, abstention, align="llll" + "r" * 11),
        "",
    ]
    if any(row["records"] != row["searches"] for row in rows):
        lines += [
            "Rates are over records: a search the run repeats counts once in `searches` and once per repeat in every "
            "rate.",
            "",
        ]
    if heldout["keys_without_dev_tau"]:
        lines += [
            "Held-out threshold keys with no dev τ (no dev search under the key in a run that made each search "
            "once): the reserved option alone decides their searches under both rules.",
            "",
            *(
                f"- `{item['key']}`: {item['searches']:,} searches of {', '.join(item['arms'])}"
                for item in heldout["keys_without_dev_tau"]
            ),
            "",
        ]
    latency = [[row["arm"], row["source"], *_latency_cells(row["latency_ms"])] for row in rows]
    header = ["arm", "source", "decision p50 ms", "p95", "server p50 ms", "p95", "retrieval p50 ms", "p95"]
    lines += ["### Latency", "", *_table(header, latency, align="llrrrrrr"), ""]
    cost = [[row["arm"], row["source"], *_cost_cells(row["cost"])] for row in rows]
    header = ["arm", "source", "asks / search", "input tokens / search", "USD", "CLM busy USD", "CLM wall USD"]
    lines += ["### Cost per 1,000 searches", "", *_table(header, cost, align="llrrrrr"), ""]
    header = ["arm", "searches", "errors", "decision s", "arm clock s", "cache hits", "cache misses"]
    lines += [
        "Per arm, all sources together: the summed seconds of its decisions, which CLM's wall figure prices, next "
        "to the arm's own clock time and the asks the decision cache replayed (hits) or sent (misses) during it.",
        "",
        *_table(header, [_arm_row(arm) for arm in heldout["arms"]], align="lrrrrrr"),
        "",
    ]
    if any(arm["records"] != arm["searches"] for arm in heldout["arms"]):
        lines += [
            "Errors, seconds and cache counts are over records: a search the run repeats counts once in `searches` "
            "and once per repeat in the rest.",
            "",
        ]
    return [*lines, *([] if heldout["repeats"] is None else _variation_section(heldout["repeats"]))]


def _variation_section(repeats: dict[str, Any]) -> list[str]:
    header = ["arm", "source", "searches", "P@1 per repeat", "P@1 of the main run", "U at dev τ per repeat"]
    header += ["largest abs Δp"]
    rows = [
        [
            row["arm"],
            row["source"],
            f"{row['searches']:,}",
            " / ".join(_fixed(entry["p_at_1"]) for entry in row["per_repeat"]) or "n/a",
            _fixed(row["main_p_at_1"]),
            " / ".join(_fixed(entry["reserved_and_dev_tau"]["utility"]) for entry in row["per_repeat"]) or "n/a",
            _fixed(row["largest_abs_dp"], 4),
        ]
        for row in repeats["rows"]
    ]
    return [
        "### Run-to-run variation",
        "",
        f"The repeats run `{repeats['run_id']}` makes each search {repeats['repeat']} times with the decision cache "
        "bypassed. The tables above come from the main held-out run, which asks each search once.",
        "",
        *_table(header, rows, align="llrrrrr"),
        "",
        "P@1 is over the positives of each repeat; U = (correct - wrong) / searches under the reserved option plus "
        "the dev τ of each search's key. `P@1 of the main run` is the single-run figure from the P@1 table. The last "
        "column is the largest change of one probability (a card's or the reserved option's) between repeats of the "
        "same search, over every repeated search of the arm and source.",
        "",
    ]


def _precision_row(row: dict[str, Any]) -> list[str]:
    first = [row["arm"], row["source"], f"{row['positives']:,}"]
    delta = "-" if row["delta_p_at_1"] is None else _with_ci(row["delta_p_at_1"], row["delta_p_at_1_ci95"], signed=True)
    return [
        *first,
        _with_ci(row["p_at_1"], row["p_at_1_ci95"]),
        _fixed(row["hybrid_p_at_1"]),
        delta,
        _fixed(row["ceiling"]),
    ]


def _arm_row(arm: dict[str, Any]) -> list[str]:
    counts = [f"{arm['searches']:,}", f"{arm['errors']:,}"]
    seconds = [_fixed(arm["decision_seconds"], 1, missing="-"), _fixed(arm["wall_seconds"], 1)]
    return [arm["arm"], *counts, *seconds, _whole(arm["cache_hits"]), _whole(arm["cache_misses"])]


def _abstention_rows(row: dict[str, Any]) -> list[list[str]]:
    rules, first, counts = row["rules"], [row["arm"], row["source"]], [f"{row['searches']:,}", f"{row['errors']:,}"]
    if rules["reserved"] is None:
        return [[*first, "-", "-", *counts, *["n/a"] * 9]]
    share = f"{row['negatives'] / row['records']:.0%}"
    if row["decider"] is None:
        return [[*first, "answer always", "-", *counts, *_rule_cells(rules["answer_always"], negatives=share)]]
    frozen = rules["reserved_and_dev_tau"]
    taus = ", ".join(f"{tau:.2f}" for tau in frozen["taus"]) or "no dev τ"
    if frozen["taus"] and frozen["without_dev_tau"]:
        taus += f" ({frozen['without_dev_tau']:,} searches without)"
    return [
        [*first, "answer always", "-", *counts, *_rule_cells(rules["answer_always"], negatives=share)],
        [*first, "reserved", "-", *counts, *_rule_cells(rules["reserved"], negatives=share)],
        [*first, "with an abstention threshold", taus, *counts, *_rule_cells(frozen, negatives=share)],
    ]


def _rule_cells(metrics: dict[str, Any], *, negatives: str) -> list[str]:
    wrong = _with_ci(metrics["wrong_tool_rate"], metrics["wrong_tool_rate_ci95"])
    abstention = [_fixed(metrics["abstention_precision"]), _fixed(metrics["abstention_recall"])]
    counts = [str(metrics[key]) for key in ("correct", "wrong", "abstained")]
    return [_fixed(metrics["coverage"]), _fixed(metrics["selective_accuracy"]), *counts, negatives, wrong, *abstention]


def _latency_cells(latency: dict[str, dict[str, float] | None]) -> list[str]:
    cells: list[str] = []
    for kind in ("decision", "server", "retrieval"):
        timing = latency[kind]
        cells += ["-", "-"] if timing is None else [f"{timing['p50']:,.0f}", f"{timing['p95']:,.0f}"]
    return cells


def _cost_cells(cost: dict[str, Any] | None) -> list[str]:
    if cost is None:
        return ["n/a"] * 5
    usd = ("usd_per_1000_searches", "clm_busy_usd_per_1000_searches", "clm_wall_usd_per_1000_searches")
    return [
        f"{cost['asks_per_search']:.2f}",
        f"{cost['input_tokens_per_search']:,.0f}",
        *(_fixed(cost[name], 4, missing="-") for name in usd),
    ]


def _dev_section(dev: dict[str, Any]) -> list[str]:
    lines = ["## Dev", "", "### Thresholds", ""]
    if dev["thresholds"]:
        header = ["threshold key", "decider", "K", "payload", "searches (positives + negatives)", "τ", "U at τ"]
        header += ["U at 0.00", "coverage at τ", "wrong-tool rate at τ"]
        rows = [_threshold_row(entry) for entry in dev["thresholds"]]
        lines += [*_table(header, rows, align="llrlrrrrrr"), ""]
    else:
        lines += ["No dev run made each search once with a decider, so no threshold was chosen.", ""]
    lines += ["### P@1 by dev run", ""]
    if dev["precision"]:
        run_ids = _unique(entry["run_id"] for entry in dev["precision"])
        cells = {(entry["arm"], entry["source"], entry["run_id"]): entry for entry in dev["precision"]}
        rows = [
            [arm, source, *(_precision_cell(cells.get((arm, source, run_id))) for run_id in run_ids)]
            for arm, source in _unique((entry["arm"], entry["source"]) for entry in dev["precision"])
        ]
        header = ["arm", "source", *(f"`{run_id}`" for run_id in run_ids)]
        lines += [
            "P@1 on positives (their count in brackets), one column per dev run that made each search once; the "
            "ablations' settings are in the runs table.",
            "",
            *_table(header, rows, align="ll" + "r" * len(run_ids)),
            "",
        ]
    lines += ["### Determinism", ""]
    if dev["determinism"]:
        header = ["run", "arm", "searches repeated", "largest abs Δp", "at", f"within {DETERMINISM_TOLERANCE}"]
        lines += [
            "Largest change of one probability (a card's or the reserved option's) between repeats of the same "
            "search, over every repeated search of the arm.",
            "",
            *_table(header, [_determinism_row(entry) for entry in dev["determinism"]], align="llrrll"),
            "",
        ]
    else:
        lines += ["No dev run repeats its searches, so determinism was not checked.", ""]
    lines += ["### Errors", ""]
    if failed := [entry for entry in dev["errors"] if entry["errors"]]:
        rows = [[f"`{e['run_id']}`", e["arm"], f"{e['errors']:,}", f"{e['searches']:,}"] for e in failed]
        return [*lines, *_table(["run", "arm", "errors", "searches"], rows, align="llrr"), ""]
    return [*lines, "No dev search failed.", ""]


def _threshold_row(entry: dict[str, Any]) -> list[str]:
    at = {point["tau"]: point for point in entry["grid"]}
    chosen, shape = at[entry["tau"]], entry["shape"]
    reserved = "reserved option" if shape["reserved_option"] else "no reserved option"
    payload = f"{reserved}, {shape['max_detail']}, questions {shape['questions']}"
    searches = f"{entry['searches']:,} ({entry['positives']:,} + {entry['negatives']:,})"
    utilities = [f"{chosen['utility']:.3f}", f"{at[0.0]['utility']:.3f}"]
    first = [f"`{entry['key']}`", entry["decider"], str(entry["k"]), payload, searches, f"{entry['tau']:.2f}"]
    return [*first, *utilities, _fixed(chosen["coverage"]), _fixed(chosen["wrong_tool_rate"])]


def _determinism_row(entry: dict[str, Any]) -> list[str]:
    at = entry["at"]
    where = f"{at['task']} ({at['source']}, {at['variant']})"
    change = f"{entry['max_abs_delta_p']:.4f}"
    within = "yes" if entry["within_tolerance"] else "no"
    return [f"`{entry['run_id']}`", entry["arm"], f"{entry['compared']:,}", change, where, within]


def _precision_cell(entry: dict[str, Any] | None) -> str:
    return "-" if entry is None else f"{entry['p_at_1']:.3f} ({entry['positives']:,})"


def _tokenizer_section(entries: Sequence[dict[str, Any]]) -> list[str]:
    rows = [_tokenizer_row(entry) for entry in entries]
    header = ["split", "Jev exchanges", "intercept", "slope", "median reported / estimated", "share under 1"]
    return [
        "## Token heuristic against Jev's count",
        "",
        "For every exchange with Jev, the input tokens Jev reported against the heuristic's estimate that the "
        "planner budgets with: a least-squares line (reported = intercept + slope x estimate), and the ratios "
        "reported / estimate. A ratio under 1 means the heuristic counted more than Jev billed.",
        "",
        *_table(header, rows, align="lrrrrr"),
        "",
    ]


def _tokenizer_row(entry: dict[str, Any]) -> list[str]:
    fit = [_fixed(entry["intercept"], 1), _fixed(entry["slope"])]
    return [
        entry["split"],
        f"{entry['exchanges']:,}",
        *fit,
        _fixed(entry["median_ratio"]),
        _fixed(entry["share_under_1"]),
    ]


def _notes(*, fallbacks_in_tau: bool) -> list[str]:
    return [
        "## Notes",
        "",
        "- **Outcomes.** A search is answered when the decider names a tool, correct when that tool is a gold one, "
        "and wrong otherwise. A negative is the same search with its gold tools removed from the candidates, so "
        "every answer to it is wrong.",
        "- **P@1** counts positives only and ignores abstention: is the first card of the decider's ranking a gold "
        "tool? `hybrid P@1` asks the same of retrieval order on the same searches, and `ceiling` is the share of "
        "them with a gold tool among the K candidates. `Δ vs hybrid` is the decider's P@1 minus retrieval's own P@1 "
        "on the same positives. Its interval resamples tasks with both answers drawn together, so it says whether "
        "the decision helped even where the two separate intervals overlap.",
        "- **Answer always.** This reinterprets the same ranking obtained with the none option available: its first "
        "card is chosen, ignoring abstention. It does not show what the model would answer without that option. "
        "The dev ablation tested that setting separately; for logprob it also changed the number of rounds.",
        '- **Rules.** `reserved`: the decider abstains when its reserved "none of these" option is at least as '
        "likely as its best card. `with an abstention threshold`: it also abstains when that card's probability "
        "is below τ, the "
        "threshold dev chose for the search's threshold key (model, prompt version, payload shape, question kind). "
        "Probabilities compare only within one question, so a τ never crosses keys.",
        "- **Rates.** coverage: answered / searches. Selective accuracy: correct / answered. Wrong-tool rate: "
        "wrong / searches. Abstention precision: abstentions on searches without a gold tool among the "
        "candidates / abstentions. Abstention recall: those abstentions / searches without a gold tool among the "
        "candidates.",
        f"- **Choosing τ.** Per threshold key, on dev: of τ = 0.00, 0.05, ..., {THRESHOLD_GRID[-1]:.2f}, the one "
        "with the highest U = (correct - wrong) / searches, where abstaining counts 0; of equal U, the lower τ wins. "
        "It uses the dev runs that made each search once, both query sources, positives and negatives; a search that "
        "several of those runs made counts once. "
        + (
            "Fallback searches of the `model` source (its writer made no search, so the request was used) repeat "
            "their `plain` search and count twice in U(τ) and in the searches counts of the thresholds table. "
            if fallbacks_in_tau
            else ""
        )
        + "The held-out risk-coverage tables are there to read, never to choose.",
        f"- **Intervals.** 95% percentile bootstrap, {BOOTSTRAP_RESAMPLES:,} resamples with seed {BOOTSTRAP_SEED}, "
        "resampling tasks: a task's positive search, its negative and their repeats are drawn together.",
        "- **Errors.** A search whose decider raised `DecisionError` counts as an error and stays out of every rate, "
        "latency and cost figure.",
        "- **Latency.** Decision: the critical path of the decider's calls as the client timed them, network "
        "included; a call the decision cache replays keeps the time the original call took. Server: the same by the "
        "server's own clock, where it reports one. Retrieval: the first hybrid retrieval of the search's queries, "
        "measured live; a query whose vector the embedding cache already held skips the embeddings call, so it "
        "reads faster than a cold one.",
        "- **Cost.** Jev: reported input tokens at the price its declared limits give. Logprob: the reported usage, "
        "priced by genai-prices. CLM is paid in GPU time, at the list price in the run's manifest, two ways. Busy "
        "prices the server's own seconds per search, as if the GPU never sat idle: a lower bound. Wall prices the "
        "summed seconds of the arm's decision calls as the client saw them, as if one container served the searches "
        "one after another. It leaves out the cold starts, the idle gaps between arms and the scale-down window, "
        "which Modal also bills. Modal's CPU and memory charges come on top of both and are not included. "
        "Every arm also embeds its queries for hybrid retrieval, at the same cost for each arm, which the F1 "
        "retrieval results report.",
        "- **Negatives** stand in for a catalog without the gold tools: the gold ids are dropped from a deeper "
        "retrieval, so the other tools' order can move slightly from what a smaller catalog would give.",
        "",
    ]


def _risk_coverage_section(summary: dict[str, Any]) -> list[str]:
    lines = ["## Risk-coverage over the threshold grid", ""]
    for entry in summary["dev"]["thresholds"]:
        lines += [
            f"### Dev: `{entry['key']}`",
            "",
            *_grid_table(entry["grid"], marked=entry["tau"], negative_share=entry["negatives"] / entry["searches"]),
            "",
        ]
    if summary["heldout"] is not None:
        for curve in summary["heldout"]["risk_coverage"]:
            lines += [
                f"### Held-out: {curve['arm']}, for reading only",
                "",
                *_grid_table(curve["grid"], marked=curve["dev_tau"], negative_share=curve["negative_share"]),
                "",
            ]
    return lines


def _grid_table(grid: Sequence[dict[str, Any]], *, marked: float | None, negative_share: float) -> list[str]:
    """One row per τ; the τ chosen on dev is in bold."""
    rows = [
        [
            f"**{point['tau']:.2f}**" if point["tau"] == marked else f"{point['tau']:.2f}",
            _fixed(point["coverage"]),
            _fixed(point["selective_accuracy"]),
            *[str(point[key]) for key in ("correct", "wrong", "abstained")],
            f"{negative_share:.0%}",
            _fixed(point["wrong_tool_rate"]),
            f"{point['utility']:.3f}",
        ]
        for point in grid
    ]
    return _table(
        ["τ", "coverage", "selective accuracy", "correct", "wrong", "abstained", "negatives", "wrong-tool rate", "U"],
        rows,
        align="r" * 9,
    )


def _table(header: Sequence[str], rows: Iterable[Sequence[str]], *, align: str) -> list[str]:
    """A Markdown table; `align` has an `l` for each left-aligned column and an `r` for each right-aligned one."""
    if len(align) != len(header):
        raise ValueError(f"{len(header)} columns, {len(align)} alignments")
    return [
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" if side == "l" else "---:" for side in align) + "|",
        # A pipe splits a table cell even inside a code span, as in a threshold key: escape it.
        *("| " + " | ".join(cell.replace("|", "\\|") for cell in row) + " |" for row in rows),
    ]


def _fixed(value: float | None, digits: int = 3, *, missing: str = "n/a") -> str:
    return missing if value is None else f"{value:.{digits}f}"


def _with_ci(value: float | None, interval: Sequence[float] | None, *, signed: bool = False) -> str:
    if value is None:
        return "n/a"
    shown = f"{value:+.3f}" if signed else f"{value:.3f}"
    return shown if interval is None else f"{shown} ({interval[0]:.2f} to {interval[1]:.2f})"


def _whole(value: int | None) -> str:
    return "-" if value is None else f"{value:,}"
