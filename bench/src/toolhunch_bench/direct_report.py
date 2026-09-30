"""Metrics and publication for direct-choice runs, excluding replay usage from provider measurements."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from toolhunch_bench.direct import DIRECT_ARMS, NOT_APPLICABLE
from toolhunch_bench.metrics import Outcome, cluster_bootstrap_ci, percentile, selective_metrics

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


def _interval(values: Sequence[tuple[str, float]]) -> dict[str, Any] | None:
    if not values:
        return None
    clusters: dict[str, list[float]] = defaultdict(list)
    for task, value in values:
        clusters[task].append(value)
    low, high = cluster_bootstrap_ci(list(clusters.values()), resamples=2000, seed=0)
    return {"value": sum(value for _, value in values) / len(values), "ci95": [low, high]}


def _cost(records: Sequence[dict[str, Any]], *, phase: str, arm: str) -> dict[str, Any]:
    own = [r for r in records if r["phase"] == phase and (r["provider_calls"] or arm == "hybrid@20")]
    calls = [call for r in own for call in r["provider_calls"]]
    input_tokens = sum(call["input_tokens"] or 0 for call in calls)
    cached = sum(call["cache_read_tokens"] or 0 for call in calls)
    unpriced = sum(call["usd"] is None or call["list_usd"] is None for call in calls)
    billed = sum(call["usd"] or 0 for call in calls) + sum(r["search_usd"] for r in own)
    list_price = sum(call["list_usd"] or 0 for call in calls) + sum(r["search_usd"] for r in own)
    latencies = [r["search_seconds"] + (r["decision_seconds"] or 0) for r in own if r["error"] is None]
    return {
        "observed_searches": len(own),
        "provider_calls": len(calls),
        "replays": sum(r["replayed"] for r in records if r["phase"] == phase),
        "input_tokens": input_tokens,
        "cache_read_tokens": cached,
        "cache_share": cached / input_tokens
        if input_tokens and all(call["provider"] != "typesafe" for call in calls)
        else None,
        "billed_usd": billed if own and not unpriced else None,
        "list_usd": list_price if own and not unpriced else None,
        "billed_usd_per_1000": billed * 1000 / len(own) if own and not unpriced else None,
        "list_usd_per_1000": list_price * 1000 / len(own) if own and not unpriced else None,
        "latency_p50_ms": percentile(latencies, 50) * 1000 if latencies else None,
        "latency_p95_ms": percentile(latencies, 95) * 1000 if latencies else None,
        "unpriced_attempts": unpriced,
        "known_billed_subtotal_usd": billed,
        "known_list_subtotal_usd": list_price,
    }


def _row(
    records: Sequence[dict[str, Any]],
    *,
    arm: str,
    catalog: str,
    baseline: dict[tuple[str, str], bool],
) -> dict[str, Any]:
    positives = [r for r in records if r["variant"] == "positive"]
    parsed = [r for r in records if r["error"] is None and r["abstained"] is not None]

    def correct(r: dict[str, Any]) -> bool:
        return r["variant"] == "positive" and r["pick"] in r["relevant"] and r["error"] is None

    def task_id(r: dict[str, Any]) -> str:
        return f"{r['catalog']}:{r['task']}"

    outcomes = [
        Outcome(
            answered=not r["abstained"],
            correct=correct(r),
            gold_in_candidates=r["variant"] == "positive" and bool(set(r["candidates"]) & set(r["relevant"])),
        )
        for r in parsed
    ]
    none = asdict(selective_metrics(outcomes)) if outcomes else {}
    values: dict[str, list[tuple[str, float]]] = {
        field: []
        for field in (
            "coverage",
            "selective_accuracy",
            "wrong_tool_rate",
            "abstention_precision",
            "abstention_recall",
        )
    }
    for r, o in zip(parsed, outcomes, strict=True):
        key = task_id(r)
        values["coverage"].append((key, float(o.answered)))
        values["wrong_tool_rate"].append((key, float(o.answered and not o.correct)))
        if o.answered:
            values["selective_accuracy"].append((key, float(o.correct)))
        else:
            values["abstention_precision"].append((key, float(not o.gold_in_candidates)))
        if not o.gold_in_candidates:
            values["abstention_recall"].append((key, float(not o.answered)))
    for field, observations in values.items():
        none[field] = _interval(observations)
    none |= {
        "correct": sum(correct(r) for r in parsed),
        "wrong": sum(not r["abstained"] and not correct(r) for r in parsed),
        "abstained": sum(r["abstained"] for r in parsed),
        "requests": len(records),
        "parsed_requests": len(parsed),
        "errors": len(records) - len(parsed),
        "negative_requests": sum(r["variant"] == "negative" for r in records),
        "negative_share": sum(r["variant"] == "negative" for r in records) / len(records) if records else None,
    }
    return {
        "arm": arm,
        "catalog": catalog,
        "status": "applicable",
        "catalogs": sorted({r["catalog"] for r in records}),
        "positives": len(positives),
        "relevant_pick_rate": _interval([(task_id(r), float(correct(r))) for r in positives]),
        "delta_vs_hybrid": _interval(
            [
                (task_id(r), float(correct(r)) - baseline[(r["catalog"], r["task"])])
                for r in positives
                if (r["catalog"], r["task"]) in baseline
            ]
        ),
        "ranking_ignoring_abstention": _interval(
            [
                (task_id(r), float(r["error"] is None and bool(r["ranked"]) and r["ranked"][0] in r["relevant"]))
                for r in positives
            ]
        )
        if arm in {"hybrid@20+jev", "jev-all"}
        else None,
        "none_option": none,
        "detail_counts": {level: sum(level in r["detail"] for r in records) for level in ("FULL", "BRIEF", "NAME")},
        "multi_calls": sum(r["extra_calls"] > 0 for r in records),
        "agent_text_none": sum(r.get("text_is_none") is True for r in parsed),
        "agent_other_text_abstentions": sum(r["abstained"] and r.get("text_is_none") is False for r in parsed),
        "positive_cost": {phase: _cost(records, phase=phase, arm=arm) for phase in ("cold", "warm")},
        "negative_cost": _cost(records, phase="negative", arm=arm),
    }


def build_direct_report(run_dir: Path, *, out_dir: Path) -> dict[str, Any]:
    """Generate a summary and public table from recorded requests; this never contacts a provider."""
    manifest = json.loads((run_dir / "manifest.json").read_text())
    records = [json.loads(line) for line in (run_dir / "run.jsonl").read_text().splitlines() if line.strip()]
    calls = [json.loads(line) for line in (run_dir / "calls.jsonl").read_text().splitlines() if line.strip()]
    not_applicable = dict(NOT_APPLICABLE)
    for exclusion in manifest.get("not_applicable", []):
        not_applicable[(exclusion["arm"], exclusion["catalog"])] = {
            field: exclusion[field] for field in ("reason", "source", "date")
        }
    sources = [c["source"] for c in manifest["catalogs"]]
    common = [source for source in sources if all((arm, source) not in not_applicable for arm in DIRECT_ARMS)]
    applicable = [r for r in records if (r["arm"], r["catalog"]) not in not_applicable]
    baseline = {
        (r["catalog"], r["task"]): r["error"] is None and r["pick"] in r["relevant"]
        for r in records
        if r["arm"] == "hybrid@20" and r["variant"] == "positive"
    }
    rows: list[dict[str, Any]] = []
    validation: dict[str, Any] = {}
    for arm in DIRECT_ARMS:
        raw = [r for r in records if r["arm"] == arm]
        excluded = [r for r in raw if (arm, r["catalog"]) in not_applicable]
        own = [r for r in applicable if r["arm"] == arm]
        rows.append(_row([r for r in own if r["catalog"] in common], arm=arm, catalog="pooled", baseline=baseline))
        if len(common) != len(sources) and all((arm, source) not in not_applicable for source in sources):
            rows.append(_row(own, arm=arm, catalog="all_catalogs", baseline=baseline))
        for selected in manifest["catalogs"]:
            pair = (arm, selected["source"])
            if pair in not_applicable:
                rejected = [r for r in excluded if r["catalog"] == selected["source"]]
                rows.append(
                    {
                        "arm": arm,
                        "catalog": selected["source"],
                        "status": "not applicable",
                        "reason": not_applicable[pair]["reason"],
                        "evidence": not_applicable[pair],
                        "excluded_requests": len(rejected),
                        "excluded_rejected_attempts": sum(r["error"] is not None for r in rejected),
                        "rejection_diagnostics": sorted({r["error"] for r in rejected if r["error"] is not None}),
                        "positives": None,
                        "relevant_pick_rate": None,
                        "delta_vs_hybrid": None,
                        "ranking_ignoring_abstention": None,
                        "none_option": None,
                        "positive_cost": None,
                        "negative_cost": None,
                        "excluded_attempt_cost": {
                            phase: _cost(rejected, phase=phase, arm=arm) for phase in ("cold", "warm", "negative")
                        },
                    }
                )
                continue
            subset = [r for r in own if r["catalog"] == selected["source"]]
            if subset:
                rows.append(_row(subset, arm=arm, catalog=selected["source"], baseline=baseline))
        successful = [r for r in own if r["error"] is None]
        raw_calls = [call for call in calls if call["arm"] == arm and call["model"] != "text-embedding-3-small"]
        actual = [call for call in raw_calls if (arm, call["catalog"]) not in not_applicable]
        planned = sum(
            c["positives"] + c["negatives"] for c in manifest["catalogs"] if (arm, c["source"]) not in not_applicable
        )
        errors = sum(r["error"] is not None for r in own)
        validation[arm] = {
            "planned_requests": planned,
            "raw_requests": len(raw),
            "raw_errors": sum(r["error"] is not None for r in raw),
            "raw_provider_calls": len(raw_calls),
            "excluded_requests": len(excluded),
            "excluded_rejected_attempts": sum(r["error"] is not None for r in excluded),
            "requests": len(own),
            "errors": errors,
            "error_rate": errors / planned if planned else None,
            "classified_successful": sum(r["abstained"] is not None for r in successful),
            "successful_requests": len(successful),
            "classified_fraction": sum(r["abstained"] is not None for r in successful) / len(successful)
            if successful
            else None,
            "provider_calls": len(actual),
            "replayed_requests": sum(r["replayed"] for r in own),
            "cache_fields_present": all("cache_read_tokens" in call for call in actual),
            "cache_usage_received": all(
                call["cache_read_tokens"] is not None for call in actual if call["error"] is None
            ),
        }
    actual_usd = sum(call["usd"] or 0 for call in calls)
    estimated = manifest["estimate"]["usd"]
    full_estimate = manifest.get("full_estimate", manifest["estimate"])["usd"]
    remaining, projection_method = _remaining_projection(manifest, applicable, calls, not_applicable=not_applicable)
    budget_charge = sum(call["budget_charge_usd"] for call in calls)
    workload = _remaining_workload(manifest, applicable, not_applicable=not_applicable)
    budget_reference = full_estimate if any(workload.values()) else 0.0
    summary = {
        "schema_version": 1,
        "manifest": manifest,
        "applicability": {
            "common_catalogs": common,
            "primary_pool": "pooled: common catalogs across all five arms",
            "secondary_pool": "all_catalogs: all selected catalogs, only for arms applicable everywhere",
            "not_applicable": [
                {"arm": arm, "catalog": source, **evidence}
                for (arm, source), evidence in not_applicable.items()
                if source in sources
            ],
            "historical_run_status": {
                "completed": manifest["completed"],
                "stop_reason": manifest.get("stop_reason"),
                "interpretation": "Original run status retained; applicability does not retroactively "
                "complete a stopped run.",
            },
        },
        "rows": rows,
        "validation": validation,
        "run_cost": {
            "provider_attempts": len(calls),
            "verified_usd": actual_usd,
            "budget_charge_usd": budget_charge,
            "unpriced_attempts": sum(call["usd"] is None for call in calls),
            "estimate_usd": estimated,
            "actual_to_estimate": actual_usd / estimated if estimated else None,
            "projected_remaining_usd": remaining,
            "projected_p1_total_usd": manifest.get("prior_p1_usd", 0) + budget_charge + remaining
            if remaining is not None
            else None,
            "projection_method": projection_method,
            "observed_remaining_usd": remaining if projection_method.startswith("observed projection") else None,
            "remaining_requests_by_arm": {
                arm: sum(count for (_, planned_arm, _), count in workload.items() if planned_arm == arm)
                for arm in DIRECT_ARMS
            },
            "uncached_remaining_budget_reference_usd": budget_reference,
            "uncached_p1_budget_reference_usd": manifest.get("prior_p1_usd", 0) + budget_charge + budget_reference,
            "budget_reference_method": "Recorded full uncached/max-output estimate retained as a conservative "
            "reference for any unfinished workload, without subtracting successful request costs. Historical "
            "retrieval approximations and token framing prevent a guaranteed upper bound; physical-call guard "
            "reservations remain independent. This report does not authorize another run.",
            "full_estimate_usd": full_estimate,
        },
        "caveats": [
            "The metric is selecting a relevant tool, not completing a task; tasks can list several relevant tools.",
            "Negatives remove the task's labeled relevant tools; an unlabeled alternative may still serve it.",
            "Pooled none-option metrics state their observed positive/negative mix; errors are reported separately.",
            "Replay outcomes are scored, but their historical usage, latency and cache reads are excluded.",
            "Search overhead is attributed to each strategy that searches; physical search calls are paid once.",
            "Cold is the first scored positive; a replay there provides no new cold measurement.",
            "Caching reflects one provider's routing in one single-turn run and does not isolate a latency effect.",
        ],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out_dir / "README.md").write_text(_markdown(summary))
    return summary


def _remaining_workload(
    manifest: dict[str, Any],
    records: Sequence[dict[str, Any]],
    *,
    not_applicable: dict[tuple[str, str], Any],
) -> dict[tuple[str, str, str], int]:
    if not manifest["pilot"] and manifest["completed"] is True:
        return {}
    workload: dict[tuple[str, str, str], int] = {}
    for catalog in manifest.get("full_catalogs", manifest["catalogs"]):
        for arm in DIRECT_ARMS:
            if (arm, catalog["source"]) in not_applicable:
                continue
            for variant, field in (("positive", "positives"), ("negative", "negatives")):
                successful = {
                    r["task"]
                    for r in records
                    if r["catalog"] == catalog["source"]
                    and r["arm"] == arm
                    and r["variant"] == variant
                    and r["error"] is None
                    and r["abstained"] is not None
                }
                workload[(catalog["source"], arm, variant)] = max(0, catalog[field] - len(successful))
    return workload


def _remaining_projection(
    manifest: dict[str, Any],
    records: Sequence[dict[str, Any]],
    calls: Sequence[dict[str, Any]],
    *,
    not_applicable: dict[tuple[str, str], Any],
) -> tuple[float | None, str]:
    if not manifest["pilot"] and manifest["completed"] is True:
        return 0.0, "completed full run"
    remaining = 0.0
    full_estimate = manifest.get("full_estimate", manifest["estimate"])
    workload = _remaining_workload(manifest, records, not_applicable=not_applicable)
    if not manifest["pilot"] and not any(workload.values()):
        return None, "incomplete full run: recorded workload covered but completion unconfirmed"
    if any(call["model"] == "text-embedding-3-small" and call["usd"] is None for call in calls):
        return (
            full_estimate["usd"],
            "conservative recorded uncached estimate fallback: applicable embedding attempts have unknown "
            "usage; unknown usage is not zero; not guaranteed upper bound",
        )
    for catalog in manifest.get("full_catalogs", manifest["catalogs"]):
        for arm in DIRECT_ARMS[1:]:
            if (arm, catalog["source"]) in not_applicable:
                continue
            for variant, phase in (("positive", "warm"), ("negative", "negative")):
                own = [
                    r
                    for r in records
                    if r["catalog"] == catalog["source"]
                    and r["arm"] == arm
                    and r["variant"] == variant
                    and r["error"] is None
                    and r["abstained"] is not None
                ]
                count = workload[(catalog["source"], arm, variant)]
                real = [r for r in own if r["phase"] == phase and r["provider_calls"]]
                if not count:
                    continue
                if not real:
                    return (
                        full_estimate["usd"],
                        "conservative recorded uncached estimate fallback: an applicable source/arm has no "
                        "measured phase; whole-workload reference, not guaranteed upper bound",
                    )
                if any(call["usd"] is None for r in real for call in r["provider_calls"]):
                    return (
                        full_estimate["usd"],
                        "conservative recorded uncached estimate fallback: an applicable phase has unpriced "
                        "attempts; unknown usage is not zero; not guaranteed upper bound",
                    )
                paid = sum(call["usd"] for r in real for call in r["provider_calls"])
                remaining += paid / len(real) * count
    full_embedding = sum(line["usd"] for line in full_estimate["lines"] if line["model"] == "text-embedding-3-small")
    paid_embedding = sum(call["usd"] or 0 for call in calls if call["model"] == "text-embedding-3-small")
    remaining += max(0.0, full_embedding - paid_embedding)
    return (
        remaining,
        "observed projection: per-source warm-positive and negative applicable priced rates; "
        "successful requests already covered; embedding reference; not a guaranteed upper bound",
    )


def _markdown(summary: dict[str, Any]) -> str:
    def interval(value: dict[str, Any] | None) -> str:
        return f"{value['value']:.3f} ({value['ci95'][0]:.3f} to {value['ci95'][1]:.3f})" if value else "n/a"

    def money(value: float | None, *, unpriced: int = 0) -> str:
        return "unknown" if unpriced else (f"${value:.4f}" if value is not None else "n/a")

    lines = [
        "# ToolRet direct choice",
        "",
        "Single-turn selection of a relevant tool from real source catalogs.",
        "",
        "Primary pooled comparison uses the same common catalogs in every arm: "
        + ", ".join(summary["applicability"]["common_catalogs"])
        + ".",
        "Secondary all_catalogs rows include every selected catalog and are not a five-arm comparison.",
        f"Original run completed: {summary['manifest']['completed']}; original stop reason: "
        f"{summary['manifest'].get('stop_reason') or 'none'}. Applicability does not change this historical status.",
        "",
        "95% intervals resample tasks together, 2,000 resamples, seed 0. Abstentions and errors are misses in the "
        "positive-request headline. None-option metrics use parsed requests; errors are listed separately.",
        "",
        "| Arm | Catalog | Relevant picks / positives | Paired Δ vs hybrid | Correct / wrong / abstained | "
        "Negatives | Errors |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["rows"]:
        if row["status"] == "not applicable":
            lines.append(
                f"| {row['arm']} | {row['catalog']} (not applicable) | n/a | n/a | n/a | n/a | "
                f"{row['excluded_rejected_attempts']} rejected attempts, excluded |"
            )
            continue
        none = row["none_option"]
        delta = "—" if row["arm"] == "hybrid@20" else interval(row["delta_vs_hybrid"])
        mix = f"{none['negative_share']:.1%}" if none["negative_share"] is not None else "n/a"
        lines.append(
            f"| {row['arm']} | {row['catalog']} | {interval(row['relevant_pick_rate'])} | "
            f"{delta} | {none['correct']} / {none['wrong']} / {none['abstained']} | "
            f"{none['negative_requests']}/{none['requests']} ({mix}) | {none['errors']} |"
        )
    lines += [
        "",
        "## Observed provider cost",
        "",
        "Replay responses do not contribute usage or timing. Search "
        "cost and latency are included for strategies that search. Negatives are priced separately.",
        "",
        "| Arm | Catalog | Cold billed | Warm billed / 1,000 | Warm list / 1,000 | Warm cache share | "
        "Warm latency p50 / p95 ms | Negative billed / 1,000 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["rows"]:
        if row["status"] == "not applicable":
            lines.append(f"| {row['arm']} | {row['catalog']} (not applicable) | n/a | n/a | n/a | n/a | n/a | n/a |")
            continue
        cold, warm = row["positive_cost"]["cold"], row["positive_cost"]["warm"]
        negative = row["negative_cost"]
        cache = f"{warm['cache_share']:.1%}" if warm["cache_share"] is not None else "n/a"
        latency = (
            f"{warm['latency_p50_ms']:.1f} / {warm['latency_p95_ms']:.1f}"
            if warm["latency_p50_ms"] is not None
            else "n/a"
        )
        lines.append(
            f"| {row['arm']} | {row['catalog']} | {money(cold['billed_usd'], unpriced=cold['unpriced_attempts'])} | "
            f"{money(warm['billed_usd_per_1000'], unpriced=warm['unpriced_attempts'])} | "
            f"{money(warm['list_usd_per_1000'], unpriced=warm['unpriced_attempts'])} | {cache} | {latency} | "
            f"{money(negative['billed_usd_per_1000'], unpriced=negative['unpriced_attempts'])} |"
        )
    exclusions = [row for row in summary["rows"] if row["status"] == "not applicable"]
    if exclusions:
        lines += ["", "## Non-applicable pairs and historical rejections", ""]
        for row in exclusions:
            lines.append(
                f"- {row['arm']} / {row['catalog']}: {row['reason']}. "
                f"Evidence: {row['evidence']['source']} ({row['evidence']['date']}); "
                f"{row['excluded_requests']} historical requests excluded, "
                f"{row['excluded_rejected_attempts']} rejected attempts."
            )
        lines += [
            "",
            "| Arm | Catalog | Historical phase | Rejected provider attempts | Billed | List | "
            "Billed / 1,000 | List / 1,000 |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
        for row in exclusions:
            for phase, cost in row["excluded_attempt_cost"].items():
                if cost["provider_calls"]:
                    unknown = cost["unpriced_attempts"]
                    lines.append(
                        f"| {row['arm']} | {row['catalog']} | {phase} | {cost['provider_calls']} | "
                        f"{money(cost['billed_usd'], unpriced=unknown)} | "
                        f"{money(cost['list_usd'], unpriced=unknown)} | "
                        f"{money(cost['billed_usd_per_1000'], unpriced=unknown)} | "
                        f"{money(cost['list_usd_per_1000'], unpriced=unknown)} |"
                    )
    lines += ["", "## Caveats", ""] + [f"- {caveat}" for caveat in summary["caveats"]]
    lines += [
        "",
        "## Reproduce",
        "",
        "```shell",
        "uv run toolhunch-bench direct --dry-run",
        "uv run toolhunch-bench direct --pilot",
        "uv run toolhunch-bench direct",
        "uv run toolhunch-bench direct-report bench/runs/RUN --out bench/results/2026-09-toolret-direct/",
        "```",
        "",
        f"Verified provider usage in this run: {money(summary['run_cost']['verified_usd'])}.",
        f"Guarded charges, including all failed/excluded attempts: {money(summary['run_cost']['budget_charge_usd'])}; "
        f"{summary['run_cost']['unpriced_attempts']} attempts have unknown usage.",
        f"Projected remaining spend: {money(summary['run_cost']['projected_remaining_usd'])}; "
        f"cumulative P1 including historical prior and all guarded charges: "
        f"{money(summary['run_cost']['projected_p1_total_usd'])}.",
        f"Method: {summary['run_cost']['projection_method']}.",
        f"Separate recorded uncached budget reference for remaining workload: "
        f"{money(summary['run_cost']['uncached_remaining_budget_reference_usd'])}. "
        f"{summary['run_cost']['budget_reference_method']}",
        "",
        "Not affiliated with TypeSafe, OpenAI or Pydantic.",
        "",
    ]
    return "\n".join(lines)
