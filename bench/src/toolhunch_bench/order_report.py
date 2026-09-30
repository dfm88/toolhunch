"""Generate clustered order diagnostics and validate pilot evidence, without provider calls."""

from __future__ import annotations

import hashlib
import itertools
import json
import random
import statistics
from typing import TYPE_CHECKING, Any

from toolhunch_bench import BENCH_DIR
from toolhunch_bench.metrics import cluster_bootstrap_ci, percentile
from toolhunch_bench.order import ORDER_SEEDS

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

_RESAMPLES = 2000


def _records(path: Path) -> list[dict[str, Any]]:
    return [] if not path.exists() else [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _metric(clusters: Sequence[Sequence[float]]) -> dict[str, Any]:
    if not clusters:
        return {"value": None, "ci95": None, "tasks": 0}
    values = [value for cluster in clusters for value in cluster]
    return {
        "value": statistics.mean(values),
        "ci95": list(cluster_bootstrap_ci(clusters, resamples=_RESAMPLES, seed=0)),
        "tasks": len(clusters),
    }


def _groups(records: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str, str], list[Mapping[str, Any]]]:
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for row in records:
        if row.get("record") == "search" and row.get("decider") is not None:
            grouped.setdefault((row["decider"], row["task"], row["variant"]), []).append(row)
    return {key: sorted(rows, key=lambda row: row["order_seed"]) for key, rows in grouped.items()}


def pilot_gates(run_dir: Path) -> dict[str, Any]:
    """Recompute completeness, distinct asked orders, wire mappings and verified cost gates from raw records."""
    manifest: dict[str, Any] = json.loads((run_dir / "manifest.json").read_text())
    records = _records(run_dir / "run.jsonl")
    calls = _records(run_dir / "calls.jsonl")
    groups = _groups(records)
    searches = [row for row in records if row.get("record") == "search" and row.get("decider") is not None]
    expected = len(manifest.get("task_ids", [])) * 2 * 2 * 5
    errors = sum(row.get("error") is not None for row in searches)
    mappings_valid = True
    orders_valid = True
    asks = 0
    for (_decider, task_id, variant), rows in groups.items():
        if [row["order_seed"] for row in rows] != list(ORDER_SEEDS):
            orders_valid = False
            continue
        presented = [tuple(row["candidates"]) for row in rows]
        if len(set(presented)) != 5 or any(len(order) != 20 or len(set(order)) != 20 for order in presented):
            orders_valid = False
        for row in rows:
            expected_order = list(presented[0])
            if row["order_seed"]:
                digest = hashlib.sha256(f"tool-order-v1:{task_id}:{row['order_seed']}".encode()).digest()
                random.Random(int.from_bytes(digest, "big")).shuffle(expected_order)
            orders_valid &= row["candidates"] == expected_order
            if variant == "negative" and set(row["candidates"]) & set(row["relevant"]):
                orders_valid = False
            exchanges: list[dict[str, Any]] = row.get("exchanges") or []
            asks += len(exchanges)
            if not exchanges or row.get("replayed") is not False:
                mappings_valid = False
            for exchange in exchanges:
                by_option: dict[str, str] = exchange.get("option_card_ids", {})
                candidate_order: list[str] = exchange.get("candidate_order", [])
                request: dict[str, Any] = exchange.get("request", {})
                questions: dict[str, Any] = request.get("questions", {})
                if len(questions) != 1:
                    mappings_valid = False
                    continue
                question_id, question = next(iter(questions.items()))
                keys = [option[0] for option in question["options"]]
                card_keys = [key for key in keys if key != "none"]
                mappings_valid &= list(by_option) == card_keys and list(by_option.values()) == candidate_order
                mappings_valid &= len(candidate_order) == len(set(candidate_order))
                mappings_valid &= set(candidate_order) <= set(row["candidates"])
                response: dict[str, Any] = exchange.get("response", {})
                answers: dict[str, Any] = response.get("answers", {})
                response_keys: dict[str, float] = answers.get(question_id, {})
                mappings_valid &= set(response_keys) == set(keys)
                if exchange is exchanges[-1]:
                    mappings_valid &= keys[-1:] == ["none"]
            if exchanges:
                first_ids: list[str] = [
                    card
                    for exchange in exchanges
                    if exchange["round"] == 1
                    for card in exchange.get("candidate_order", [])
                ]
                mappings_valid &= set(first_ids) == set(row["candidates"])
                chunks: list[list[str]] = [
                    exchange["candidate_order"] for exchange in exchanges if exchange["round"] == 1
                ]
                asked_order = [
                    cards[index]
                    for index in range(max(map(len, chunks), default=0))
                    for cards in chunks
                    if index < len(cards)
                ]
                mappings_valid &= asked_order == row["candidates"]
    known = all(
        call.get("usd") is not None
        and call.get("error") is None
        and call.get("input_tokens") is not None
        and call.get("output_tokens") is not None
        for call in calls
    )
    decision_calls = [
        call
        for call in calls
        if call.get("arm", "").startswith("hybrid+") and call.get("model") != "text-embedding-3-small"
    ]
    physical = len(decision_calls) == asks
    estimate = manifest["estimate"]["usd"]
    verified = sum(call.get("usd") or 0 for call in calls)
    charge = sum(call.get("budget_charge_usd", 0) for call in calls)
    complete = manifest.get("completed") is True and len(searches) == expected and len(groups) == expected // 5
    complete &= set(groups) == {
        (name, task, variant)
        for name in ("jev", "logprob")
        for task in manifest.get("task_ids", [])
        for variant in ("positive", "negative")
    }
    complete &= (
        manifest.get("experiment") == "order-sensitivity-v1"
        and manifest.get("split") == "heldout"
        and manifest.get("sources") == ["plain"]
        and manifest.get("orders", {}).get("cache_bypass") is True
        and manifest.get("orders", {}).get("seeds") == list(ORDER_SEEDS)
    )
    budget_ok = manifest["prior_p1_usd"] + charge <= manifest.get("cap_usd", 7.0)
    cost_ok = known and physical and bool(calls) and verified <= 1.5 * estimate and budget_ok
    passed = complete and errors == 0 and orders_valid and mappings_valid and cost_ok
    return {
        "passed": passed,
        "complete": complete,
        "errors": errors,
        "searches": len(searches),
        "expected_searches": expected,
        "five_different_orders": orders_valid,
        "exchange_mappings_valid": mappings_valid,
        "physical_attempts_match": physical,
        "verified_cost": known,
        "cost_gate": cost_ok,
        "verified_usd": verified,
        "budget_charge_usd": charge,
        "estimate_usd": estimate,
        "actual_to_estimate": verified / estimate if estimate else None,
    }


def _correct(row: Mapping[str, Any]) -> float:
    return float(bool(row["ranked"]) and row["ranked"][0] in row["relevant"])


def _shuffle_bounds(vectors: Sequence[Sequence[float]]) -> dict[str, Any]:
    if not vectors:
        return {"mean": _metric([]), "min": _metric([]), "max": _metric([])}
    observed = [statistics.mean(column) for column in zip(*vectors, strict=True)]
    rng = random.Random(0)
    lows: list[float] = []
    highs: list[float] = []
    for _ in range(_RESAMPLES):
        sample = rng.choices(vectors, k=len(vectors))
        rates = [statistics.mean(column) for column in zip(*sample, strict=True)]
        lows.append(min(rates))
        highs.append(max(rates))
    return {
        "mean": _metric(vectors),
        "min": {"value": min(observed), "ci95": [percentile(lows, 2.5), percentile(lows, 97.5)]},
        "max": {"value": max(observed), "ci95": [percentile(highs, 2.5), percentile(highs, 97.5)]},
    }


def _counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    correct = sum(not row["abstained"] and row["variant"] == "positive" and _correct(row) for row in rows)
    abstained = sum(bool(row["abstained"]) for row in rows)
    negatives = sum(row["variant"] == "negative" for row in rows)
    return {
        "correct": int(correct),
        "wrong": len(rows) - int(correct) - abstained,
        "abstained": abstained,
        "searches": len(rows),
        "negatives": negatives,
        "negative_share": negatives / len(rows) if rows else None,
    }


def _positions(task_groups: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    clusters: dict[str, list[int]] = {}
    for (task, _), rows in task_groups.items():
        clusters.setdefault(task, []).extend(row["candidates"].index(row["ranked"][0]) + 1 for row in rows[1:])
    slots = [slot for values in clusters.values() for slot in values]
    counts = [slots.count(slot) for slot in range(1, 21)]
    expected = len(slots) / 20
    return {
        "definition": "outer presented candidate slot of the top card; reserved option ignored",
        "searches": len(slots),
        "counts": dict(zip(map(str, range(1, 21)), counts, strict=True)),
        "slot_1": _metric([[float(slot == 1) for slot in values] for values in clusters.values()]),
        "slots_1_to_3": _metric([[float(slot <= 3) for slot in values] for values in clusters.values()]),
        "uniform_slot_1": 0.05,
        "uniform_slots_1_to_3": 0.15,
        "chi_square": sum((count - expected) ** 2 / expected for count in counts) if expected else None,
        "degrees_of_freedom": 19,
        "p_value": None,
        "caveat": "Descriptive diagnostic only; repeated tasks are correlated. No causal uniform-position claim.",
    }


def _average(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    identity = rows[0]
    cards = identity["candidates"]
    probabilities = {card: statistics.mean(row["probabilities"].get(card, 0.0) for row in rows) for card in cards}
    ranked = sorted(cards, key=lambda card: probabilities[card], reverse=True)
    none = statistics.mean(row["none_probability"] for row in rows)
    return dict(identity) | {
        "ranked": ranked,
        "probabilities": probabilities,
        "none_probability": none,
        "abstained": none >= probabilities[ranked[0]],
    }


def _drift(
    manifest: Mapping[str, Any],
    decider: str,
    groups: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]],
    *,
    published: Mapping[str, Any] | None,
    reference_run: Path | None,
) -> dict[str, Any]:
    rejected = {"comparable": False, "reason": "matching published F2a raw evidence is unavailable"}
    if published is None or reference_run is None or not (reference_run / "manifest.json").exists():
        return rejected
    old = json.loads((reference_run / "manifest.json").read_text())
    recorded = [run for run in published["runs"] if run["role"] == "main" and run["split"] == "heldout"]
    if len(recorded) != 1 or recorded[0]["manifest"] != old:
        return rejected | {"reason": "reference run is not the published main F2a run"}
    for section, fields in (
        ("dataset", ("revision", "corpus_sha256", "catalog_fingerprint")),
        ("tasks", ("sha256", "count")),
    ):
        if any(manifest[section][key] != old[section][key] for key in fields):
            return rejected | {"reason": f"{section} identities differ"}
    arm = f"hybrid+{decider}@20"
    fields = ("model_id", "limits", "prompt_version", "max_detail", "reserved_option", "k")
    if arm not in old["arms"] or any(manifest["arms"][arm][field] != old["arms"][arm][field] for field in fields):
        return rejected | {"reason": "model, prompt, limits, or configuration differs"}
    previous = {
        (row["task"], row["variant"]): row
        for row in _records(reference_run / "run.jsonl")
        if row.get("record") == "search"
        and row.get("arm") == arm
        and row.get("source") == "plain"
        and row.get("repeat") == 0
    }
    if set(previous) != set(groups):
        return rejected | {"reason": "task/variant identities differ"}
    for key, rows in groups.items():
        before, now = previous[key], rows[0]
        if before["error"] is not None or any(
            before[field] != now[field]
            for field in ("queries", "context", "candidates", "relevant", "key", "shape", "state_cut")
        ):
            return rejected | {"reason": "query, candidate, or payload identities differ"}
    positives = [key for key in groups if key[1] == "positive"]
    rate = statistics.mean(_correct(previous[key]) for key in positives)
    published_rows = [row for row in published["heldout"]["rows"] if row["arm"] == arm and row["source"] == "plain"]
    if len(published_rows) != 1 or rate != published_rows[0]["p_at_1"]:
        return rejected | {"reason": "raw reference does not reproduce the published P@1"}
    return {
        "comparable": True,
        "reference_run": reference_run.name,
        "published_p_at_1": rate,
        "identity_minus_published": _metric(
            [[_correct(groups[key][0]) - _correct(previous[key])] for key in positives]
        ),
        "caveat": "Identity drift is a date/run diagnostic and cannot be attributed to order.",
    }


def build_order_report(
    run_dir: Path,
    *,
    out_dir: Path,
    published_summary: Path | None = BENCH_DIR / "results/2026-09-toolret-decision/summary.json",
    reference_run: Path | None = None,
) -> dict[str, Any]:
    """Write an auditable summary and a public table of task-clustered order diagnostics."""
    manifest = json.loads((run_dir / "manifest.json").read_text())
    records, calls = _records(run_dir / "run.jsonl"), _records(run_dir / "calls.jsonl")
    published = (
        None
        if published_summary is None or not published_summary.exists()
        else json.loads(published_summary.read_text())
    )
    if reference_run is None and published is not None:
        reference_run = BENCH_DIR / "runs" / published["heldout"]["run_id"]
    groups = _groups(records)
    complete = {
        key: rows
        for key, rows in groups.items()
        if len(rows) == 5
        and [row["order_seed"] for row in rows] == list(ORDER_SEEDS)
        and all(row["error"] is None and row["ranked"] for row in rows)
    }
    summary: dict[str, Any] = {
        "manifest": manifest,
        "bootstrap": {"resamples": _RESAMPLES, "seed": 0, "unit": "task cluster", "level": 0.95},
        "gates": pilot_gates(run_dir),
        "excluded_incomplete_or_error_groups": len(groups) - len(complete),
        "deciders": {},
    }
    for decider in ("jev", "logprob"):
        own = {(task, variant): rows for (name, task, variant), rows in complete.items() if name == decider}
        positive = {key: rows for key, rows in own.items() if key[1] == "positive"}
        by_task: dict[str, list[float]] = {}
        for (task, _), rows in own.items():
            by_task.setdefault(task, []).append(float(len({row["abstained"] for row in rows}) == 1))
        averaged = [_average(rows) for rows in own.values()]
        actual = [
            call
            for call in calls
            if call.get("arm") == f"hybrid+{decider}@20" and call["model"] != "text-embedding-3-small"
        ]
        verified = sum(call.get("usd") or 0 for call in actual)
        counts = [_counts([rows[index] for rows in own.values()]) for index in range(5)]
        summary["deciders"][decider] = {
            "positive_tasks": len(positive),
            "p_at_1": {
                str(seed): _metric([[_correct(rows[seed])] for rows in positive.values()]) for seed in ORDER_SEEDS
            },
            "shuffle_p_at_1": _shuffle_bounds([[_correct(row) for row in rows[1:]] for rows in positive.values()]),
            "top_card_stability": _metric(
                [[float(len({row["ranked"][0] for row in rows}) == 1)] for rows in positive.values()]
            ),
            "pairwise_agreement": _metric(
                [
                    [float(left["ranked"][0] == right["ranked"][0]) for left, right in itertools.combinations(rows, 2)]
                    for rows in positive.values()
                ]
            ),
            "chosen_slots_positive": _positions(positive),
            "chosen_slots_50_50_mix": _positions(own),
            "answer_abstain_stability": _metric(list(by_task.values())),
            "abstention_by_order": dict(zip(map(str, ORDER_SEEDS), counts, strict=True)),
            "permutation_averaging": {
                "p_at_1": _metric([[_correct(_average(rows))] for rows in positive.values()]),
                "abstention": _counts(averaged),
                "production_decisions_per_search": 5,
                "decision_usd_per_1000_searches": verified / len(own) * 1000 if own else None,
                "physical_attempts_per_search": len(actual) / len(own) if own else None,
                "probabilities": "Average final distributions mapped to card ids; "
                "cards dropped in a logprob round get zero; identity breaks ties.",
            },
            "physical_attempts": len(actual),
            "verified_decision_usd": verified,
            "budget_charge_usd": sum(call["budget_charge_usd"] for call in actual),
            "planner_lowered_exchanges": sum(
                exchange["detail"] != manifest["arms"][f"hybrid+{decider}@20"]["max_detail"]
                for rows in own.values()
                for row in rows
                for exchange in row["exchanges"]
            ),
            "identity_drift": _drift(manifest, decider, own, published=published, reference_run=reference_run),
        }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out_dir / "README.md").write_text(_readme(summary))
    return summary


def _readme(summary: Mapping[str, Any]) -> str:
    def fmt(metric: Mapping[str, Any]) -> str:
        return (
            "n/a"
            if metric["value"] is None
            else f"{metric['value']:.3f} ({metric['ci95'][0]:.3f} to {metric['ci95'][1]:.3f})"
        )

    lines = [
        "# ToolRet candidate-order sensitivity",
        "",
        "K=20, plain queries, Jev BRIEF and logprob FULL, reserved option last. "
        "P@1 means a relevant tool first with abstention ignored; selecting it does not complete a task.",
        "",
        "Identity and task-seeded shuffles 1-4 were asked with no local decision replay. "
        "95% intervals resample whole task clusters 2,000 times, seed 0.",
        "",
        f"Automatic gates passed: {summary['gates']['passed']}. "
        f"Excluded incomplete/error groups: {summary['excluded_incomplete_or_error_groups']}.",
        "",
        "| Decider | Identity P@1 | Shuffle mean | Shuffle min | Shuffle max | "
        "Five-order top-card stability | Pairwise agreement |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in summary["deciders"].items():
        shuffle = row["shuffle_p_at_1"]
        lines.append(
            f"| {name} | {fmt(row['p_at_1']['0'])} | {fmt(shuffle['mean'])} | {fmt(shuffle['min'])} | "
            f"{fmt(shuffle['max'])} | {fmt(row['top_card_stability'])} | {fmt(row['pairwise_agreement'])} |"
        )
    lines += [
        "",
        "The minimum and maximum are the smallest/largest population P@1 across the four shuffles; "
        "their intervals resample tasks before taking that extremum.",
        "",
    ]
    for name, row in summary["deciders"].items():
        lines += [
            f"## {name}",
            "",
            f"P@1 by order: {', '.join(f'{seed}: {fmt(metric)}' for seed, metric in row['p_at_1'].items())}.",
            "",
            "| Order | Correct | Wrong | Abstained | Negative share |",
            "|---|---:|---:|---:|---:|",
        ]
        for seed, counts in row["abstention_by_order"].items():
            lines.append(
                f"| {seed} | {counts['correct']} | {counts['wrong']} | {counts['abstained']} | "
                f"{counts['negatives']}/{counts['searches']} ({(counts['negative_share'] or 0):.0%}) |"
            )
        lines += [
            "",
            f"Answer/abstain stability: {fmt(row['answer_abstain_stability'])}, on the stated 50/50 mix above.",
            "",
        ]
        for label, field in (("Positives", "chosen_slots_positive"), ("50/50 mix", "chosen_slots_50_50_mix")):
            slots = row[field]
            lines += [
                f"Shuffled chosen slots ({label}; outer candidate slot, reserved option ignored): "
                + ", ".join(f"{slot}: {count}" for slot, count in slots["counts"].items())
                + ".",
                "",
                f"Slot 1: {fmt(slots['slot_1'])} vs 5%; slots 1-3: {fmt(slots['slots_1_to_3'])} vs 15%; "
                f"descriptive chi-square {slots['chi_square']} (19 degrees of freedom, no p-value).",
                "",
            ]
        average = row["permutation_averaging"]
        counts = average["abstention"]
        lines += [
            f"Permutation averaging: P@1 {fmt(average['p_at_1'])}; correct/wrong/abstained "
            f"{counts['correct']}/{counts['wrong']}/{counts['abstained']}, "
            f"{counts['negatives']}/{counts['searches']} negatives ({(counts['negative_share'] or 0):.0%}). "
            f"Production requires five decisions per search; measured decision cost per 1,000 searches "
            f"${(average['decision_usd_per_1000_searches'] or 0):.4f}; "
            f"{average['physical_attempts_per_search']} physical asks per search for the five decisions.",
            "",
            average["probabilities"],
            "",
            f"Identity drift: {json.dumps(row['identity_drift'])}",
            "",
        ]
    lines += [
        "Only this date, K=20, plain queries and these two configurations are covered. "
        "The chosen-slot statistic is descriptive: repeated searches are correlated and uniform-position causality "
        "is not established. Logprob final questions contain finalists; actual per-round slot/card mappings "
        "are retained in the raw exchanges. Gold-removed negatives may still admit an unlabeled relevant tool.",
        "",
        f"Verified run cost ${summary['gates']['verified_usd']:.6f}; guarded charge "
        f"${summary['gates']['budget_charge_usd']:.6f}. Model, prompt, payload, catalog, task, seed, order, "
        "git and scheduling provenance are pinned in summary.json and the raw manifest.",
        "",
        "Calls are serialized and never locally replayed. Provider cache reads use their returned discount; "
        "unreported cache usage is charged at list price. Jev cache usage is unmeasured.",
        "",
        "Not affiliated with TypeSafe, OpenAI or Pydantic.",
        "",
    ]
    return "\n".join(lines)
