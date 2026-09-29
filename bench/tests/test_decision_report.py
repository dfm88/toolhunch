import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest
from genai_prices import Usage, calc_price
from typer.testing import CliRunner

from toolhunch_bench import cli
from toolhunch_bench.decision import CLM_DEPLOYMENT
from toolhunch_bench.decision_report import build_decision_report
from toolhunch_bench.metrics import THRESHOLD_GRID, bootstrap_ci, cluster_bootstrap_ci

JEV_ID = "jev-1.13.0@api.typesafe.ai"
CLM_HOST = "acme-workspace--clm-serve.modal.run"
CLM_ID = f"clm-latest@{CLM_HOST}"
LOGPROB_ID = "gpt-4.1-mini-2025-04-14@api.openai.com"
JEV_KEY = f"{JEV_ID}|tool-choice-v1|1111111111111111|choice"
CLM_KEY = f"{CLM_ID}|tool-choice-v1|2222222222222222|choice"
LOGPROB_KEY = f"{LOGPROB_ID}|tool-choice-v1+letters-v1|3333333333333333|choice"
SHAPE = {"candidates": 3, "questions": [[4]], "reserved_option": True, "max_detail": "FULL"}
# The runner's deployment record at another GPU price: the report prices CLM from the run's own manifest.
GPU_HOUR_USD = 1.2
DEPLOYMENT = CLM_DEPLOYMENT | {"usd_per_gpu_hour": GPU_HOUR_USD}


def arm(k: int, decider: str | None = None, model_id: str | None = None, **config: Any) -> dict[str, Any]:
    """An arm's manifest entry, as `build_decision_arms` writes it."""
    limits = None
    if decider is not None:
        price = config.pop("price_input_per_mtok", 0.042 if decider == "jev" else None)
        limits = {
            "price_input_per_mtok": price,
            "price_output_per_mtok": None,
            "source": "test",
            "checked": "2026-09-29",
        }
    return {
        "k": k,
        "decider": decider,
        "model_id": model_id,
        "model": None if model_id is None else f"FakeModel(base_url='https://{model_id.partition('@')[2]}/v1')",
        "limits": limits,
        "prompt_version": None if decider is None else "tool-choice-v1",
        "max_detail": None if decider is None else "FULL",
        "reserved_option": None if decider is None else True,
    } | config


def candidates_of(*, gold: bool, retrieval_right: bool) -> list[str]:
    """Three candidate ids in retrieval order; the gold tool is `g`, first only when retrieval ranked it first."""
    if not gold:
        return ["x", "y", "z"]
    return ["g", "x", "y"] if retrieval_right else ["x", "g", "y"]


def common(
    arm_name: str, task: str, *, variant: str, source: str, repeat: int, fallback: bool | None
) -> dict[str, Any]:
    return {
        "record": "search",
        "arm": arm_name,
        "k": 3,
        "source": source,
        "variant": variant,
        "repeat": repeat,
        "task": task,
        "queries": [f"query of {task}"],
        "fallback": fallback if source == "model" else None,
        "context": f"request of {task}",
        "relevant": ["g"],
        "retrieval_seconds": 0.25,
    }


def retrieved(
    arm_name: str,
    task: str,
    *,
    gold: bool = True,
    retrieval_right: bool = False,
    variant: str = "positive",
    source: str = "plain",
    repeat: int = 0,
    fallback: bool | None = None,
) -> dict[str, Any]:
    """A search record of a retrieval-only arm: it ranks by retrieval and decides nothing."""
    candidates = candidates_of(gold=gold, retrieval_right=retrieval_right)
    return common(arm_name, task, variant=variant, source=source, repeat=repeat, fallback=fallback) | {
        "decider": None,
        "candidates": candidates,
        "gold_in_candidates": gold,
        "ranked": candidates,
        "probabilities": {},
        "none_probability": None,
        "abstained": None,
        "key": None,
        "shape": None,
        "state_cut": None,
        "exchanges": None,
        "usage": None,
        "decision_seconds": None,
        "server_seconds": None,
        "error": None,
    }


def decided(
    arm_name: str,
    task: str,
    *,
    best: float,
    none: float | None,
    first_right: bool = False,
    gold: bool = True,
    retrieval_right: bool = False,
    decider: str = "jev",
    key: str = JEV_KEY,
    variant: str = "positive",
    source: str = "plain",
    repeat: int = 0,
    fallback: bool | None = None,
    probabilities: dict[str, float] | None = None,
    seconds: float = 0.4,
    server_seconds: float | None = 0.04,
    usage: tuple[int, int, int] = (1, 100, 1),
    exchanges: Sequence[tuple[int, int]] = (),
) -> dict[str, Any]:
    """A decided search: the decider's first card holds `best`, the reserved option `none`.

    The report reads only the first card's probability and the reserved option's, so the other cards get 0.01.
    """
    candidates = candidates_of(gold=gold, retrieval_right=retrieval_right)
    first = "g" if first_right else "x"
    if probabilities is None:
        probabilities = {first: best} | {card: 0.01 for card in candidates if card != first}
    ranked = sorted(candidates, key=lambda card: -probabilities.get(card, 0.0))
    requests, input_tokens, output_tokens = usage
    return common(arm_name, task, variant=variant, source=source, repeat=repeat, fallback=fallback) | {
        "decider": decider,
        "candidates": candidates,
        "gold_in_candidates": gold,
        "ranked": ranked,
        "probabilities": probabilities,
        "none_probability": none,
        "abstained": none is not None and none >= probabilities[ranked[0]],
        "key": key,
        "shape": SHAPE,
        "state_cut": False,
        "exchanges": [
            {
                "round": 1,
                "detail": "FULL",
                "estimated_input_tokens": estimated,
                "input_tokens": reported,
                "output_tokens": 1,
                "seconds": seconds,
                "server_seconds": server_seconds,
            }
            for estimated, reported in exchanges
        ],
        "usage": {"requests": requests, "input_tokens": input_tokens, "output_tokens": output_tokens},
        "decision_seconds": seconds,
        "server_seconds": server_seconds,
        "error": None,
    }


def failed(arm_name: str, task: str, *, decider: str, message: str, variant: str = "positive") -> dict[str, Any]:
    """A decider search that raised `DecisionError`: its candidates are kept, and it has no ranking."""
    record = retrieved(arm_name, task, variant=variant, retrieval_right=True)
    return record | {"decider": decider, "ranked": None, "error": message}


def write_run(
    root: Path,
    run_id: str,
    records: Sequence[dict[str, Any]],
    *,
    split: str,
    arms: dict[str, dict[str, Any]],
    repeat: int = 1,
    sources: Sequence[str] = ("plain",),
    arm_records: dict[str, dict[str, Any]] | None = None,
) -> Path:
    """A run directory as `run_decisions` writes it: the manifest, then each arm's searches and its arm record."""
    tasks = {record["task"] for record in records}
    manifest = {
        "run_id": run_id,
        "started": "2026-10-01T10:00:00+00:00",
        "git": {"commit": "0123456789abcdef0123", "dirty": False},
        "versions": {"python": "3.14.3", "toolhunch": "0.1.0"},
        "dataset": {
            "name": "mteb/ToolRetrieval",
            "revision": "76d45e560059754e289ea202462865a585679619",
            "tools": 44_453,
            "catalog_fingerprint": "sha256:7d56b70f3415aaaa",
        },
        "tasks": {"file": f"bench/tasks/{run_id}.json", "sha256": "ab" * 32, "count": len(tasks)},
        "split": split,
        "sources": list(sources),
        "negatives": True,
        "repeat": repeat,
        "model_queries": None,
        "embedding_model": "text-embedding-3-small",
        "clm_deployment": DEPLOYMENT if any(a["decider"] == "clm" for a in arms.values()) else None,
        "arms": arms,
    }
    lines: list[dict[str, Any]] = []
    for name in arms:
        own = [record | {"split": split} for record in records if record["arm"] == name]
        lines += own
        summary = {
            "wall_seconds": 10.0,
            "cache_hits": None if arms[name]["decider"] is None else 0,
            "cache_misses": None if arms[name]["decider"] is None else len(own),
        } | (arm_records or {}).get(name, {})
        lines.append(
            {"record": "arm", "arm": name, "searches": len(own), "errors": sum(r["error"] is not None for r in own)}
            | summary
        )
    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (run_dir / "run.jsonl").write_text("".join(json.dumps(line) + "\n" for line in lines))
    return run_dir


def report(out_dir: Path) -> tuple[dict[str, Any], str]:
    return json.loads((out_dir / "summary.json").read_text()), (out_dir / "README.md").read_text()


def rows(summary: dict[str, Any], arm_name: str) -> dict[str, dict[str, Any]]:
    """The held-out rows of one arm, by source."""
    return {row["source"]: row for row in summary["heldout"]["rows"] if row["arm"] == arm_name}


def test_report_freezes_tau_on_dev_and_applies_it_to_heldout(tmp_path: Path) -> None:
    jev = "hybrid+jev@3"
    dev_arms = {"hybrid@3": arm(3), jev: arm(3, "jev", JEV_ID)}
    # Dev, one threshold key, both sources pooled. Per tau, U = (correct - wrong) / 6:
    #   t1 right at 0.80, t2 wrong at 0.45 (gold second), t3 a negative answered at 0.40, t5 right at exactly 0.50;
    #   t4 (reserved 0.50 over 0.30) and t6 (a tie at 0.60) abstain through the reserved option at every tau.
    # U is 0 up to 0.40, 1/6 at 0.45, 2/6 at 0.50, 1/6 from 0.55 to 0.80 and 0 above: tau = 0.50, since the best
    # card answers when its probability equals tau.
    dev_searches = [
        decided(jev, "t1", best=0.8, none=0.1, first_right=True),
        decided(jev, "t2", best=0.45, none=0.1, source="model", fallback=False),
        decided(jev, "t3", best=0.4, none=0.2, gold=False, variant="negative"),
        decided(jev, "t4", best=0.3, none=0.5, gold=False, variant="negative"),
        decided(jev, "t5", best=0.5, none=0.1, first_right=True, source="model", fallback=False),
        decided(jev, "t6", best=0.6, none=0.6, gold=False, variant="negative"),
    ]
    main = write_run(
        tmp_path,
        "dev-main",
        [*dev_searches, retrieved("hybrid@3", "t1")],
        split="dev",
        arms=dev_arms,
        sources=("plain", "model"),
    )
    # A rerun that repeats two of those searches: each search counts once.
    rerun = write_run(
        tmp_path, "dev-rerun", dev_searches[1:3], split="dev", arms={jev: dev_arms[jev]}, sources=("plain", "model")
    )
    # The determinism run repeats searches three times; counted, its right answers at 0.30 would pull tau to 0.
    repeated = [
        decided(jev, task, best=0.3, none=0.1, first_right=True, repeat=index)
        for task in ("t7", "t8")
        for index in range(3)
    ]
    determinism = write_run(tmp_path, "dev-determinism", repeated, split="dev", arms={jev: dev_arms[jev]}, repeat=3)

    # Held-out, same key. At the dev tau of 0.50, h1 answers right and the negative h3 wrong; h2 (0.40) and h4
    # (0.20) abstain. Held-out's own U would peak at 0.25 (h1, h2 right, h3 wrong): the report must not use it.
    # h5 failed. The logprob arm's key has no dev searches, so it gets no tau: the reserved rule alone decides.
    # There, h1 was asked without the reserved option, and h4 had no candidate, so nothing was asked and its
    # recorded abstention stands.
    logprob = "hybrid+logprob@3"
    nothing_asked: dict[str, Any] = {"candidates": [], "ranked": [], "probabilities": {}, "abstained": True}
    heldout_searches = [
        decided(jev, "h1", best=0.9, none=0.05, first_right=True, retrieval_right=True),
        decided(jev, "h2", best=0.4, none=0.05, first_right=True),
        decided(jev, "h3", best=0.55, none=0.1, gold=False, variant="negative"),
        decided(jev, "h4", best=0.2, none=0.1, gold=False),  # the gold tool was not retrieved
        failed(jev, "h5", decider="jev", message=f"{JEV_ID}: HTTP 500"),
        retrieved("hybrid@3", "h1", retrieval_right=True),
        retrieved("hybrid@3", "h2"),
        retrieved("hybrid@3", "h3", gold=False, variant="negative"),
        retrieved("hybrid@3", "h4", gold=False),
        retrieved("hybrid@3", "h5", retrieval_right=True),
        decided(logprob, "h1", best=0.3, none=None, decider="logprob", key=LOGPROB_KEY, first_right=True),
        decided(logprob, "h3", best=0.7, none=0.2, decider="logprob", key=LOGPROB_KEY, gold=False, variant="negative"),
        decided(logprob, "h4", best=0.1, none=None, decider="logprob", key=LOGPROB_KEY, gold=False) | nothing_asked,
    ]
    heldout_arms = dev_arms | {logprob: arm(3, "logprob", LOGPROB_ID)}
    heldout = write_run(tmp_path, "heldout-main", heldout_searches, split="heldout", arms=heldout_arms)

    arguments = ["decision-report", "--dev", str(main), "--dev", str(rerun), "--dev", str(determinism)]
    result = CliRunner().invoke(cli.app, [*arguments, "--heldout", str(heldout), "--out", str(tmp_path / "report")])

    assert result.exit_code == 0, result.output
    summary, readme = report(tmp_path / "report")
    [threshold] = summary["dev"]["thresholds"]
    assert (threshold["key"], threshold["tau"]) == (JEV_KEY, 0.5)
    assert summary["heldout"]["repeats"] is None
    assert "Run-to-run variation" not in readme
    assert (threshold["searches"], threshold["positives"], threshold["negatives"]) == (6, 3, 3)
    expected_u = [0.0] * 9 + [1 / 6, 2 / 6] + [1 / 6] * 6 + [0.0] * 3
    assert [point["tau"] for point in threshold["grid"]] == list(THRESHOLD_GRID)
    assert [point["utility"] for point in threshold["grid"]] == pytest.approx(expected_u)
    escaped = JEV_KEY.replace("|", "\\|")  # a bare pipe would split the table cell
    assert f"| `{escaped}` |" in readme
    # P@1 per dev run, arm and source, side by side; the repeated run is left out.
    assert [(e["run_id"], e["arm"], e["source"], e["positives"], e["p_at_1"]) for e in summary["dev"]["precision"]] == [
        ("dev-main", "hybrid@3", "plain", 1, 0.0),
        ("dev-main", jev, "plain", 1, 1.0),
        ("dev-main", jev, "model", 2, 0.5),
        ("dev-main", jev, "model (searched)", 2, 0.5),
        ("dev-rerun", jev, "model", 1, 0.0),
        ("dev-rerun", jev, "model (searched)", 1, 0.0),
    ]

    jev_row = rows(summary, jev)["plain"]
    assert (jev_row["searches"], jev_row["errors"], jev_row["positives"]) == (4, 1, 3)
    # P@1 on the positives h1, h2 and h4, abstentions ignored; hybrid's own first card is right on h1 only.
    assert (jev_row["p_at_1"], jev_row["hybrid_p_at_1"], jev_row["ceiling"]) == pytest.approx((2 / 3, 1 / 3, 2 / 3))
    # Paired, task by task: h1 both right (0), h2 the decider right where retrieval was wrong (+1), h4 both wrong (0).
    assert jev_row["delta_p_at_1"] == pytest.approx(1 / 3)
    assert jev_row["delta_p_at_1"] == pytest.approx(jev_row["p_at_1"] - jev_row["hybrid_p_at_1"])
    reserved, frozen = jev_row["rules"]["reserved"], jev_row["rules"]["reserved_and_dev_tau"]
    assert (reserved["coverage"], reserved["selective_accuracy"], reserved["wrong_tool_rate"]) == (1.0, 0.5, 0.5)
    assert (reserved["abstention_precision"], reserved["abstention_recall"]) == (None, 0.0)
    assert (frozen["taus"], frozen["without_dev_tau"]) == ([0.5], 0)
    assert (frozen["coverage"], frozen["selective_accuracy"], frozen["wrong_tool_rate"]) == (0.5, 0.5, 0.25)
    assert (frozen["abstention_precision"], frozen["abstention_recall"]) == (0.5, 0.5)
    # The held-out curve is published, and its own best tau is not the one applied.
    [curve] = [c for c in summary["heldout"]["risk_coverage"] if c["arm"] == jev]
    utilities = {point["tau"]: point["utility"] for point in curve["grid"]}
    assert curve["dev_tau"] == 0.5
    assert (utilities[0.25], utilities[0.5]) == (0.25, 0.0)

    # The retrieval-only arm answers every search, the failed decider search included.
    baseline = rows(summary, "hybrid@3")["plain"]
    assert (baseline["searches"], baseline["errors"], baseline["p_at_1"]) == (5, 0, 0.5)
    assert (baseline["delta_p_at_1"], baseline["delta_p_at_1_ci95"]) == (None, None)  # nothing to compare with itself
    assert baseline["rules"]["reserved_and_dev_tau"]["wrong_tool_rate"] == 3 / 5

    logprob_row = rows(summary, logprob)["plain"]
    no_tau = logprob_row["rules"]["reserved_and_dev_tau"]
    assert (no_tau["taus"], no_tau["without_dev_tau"]) == ([], 3)
    for rule in (logprob_row["rules"]["reserved"], no_tau):  # h1 right, h3 wrong, h4 abstains
        assert (rule["coverage"], rule["selective_accuracy"], rule["wrong_tool_rate"]) == pytest.approx(
            (2 / 3, 1 / 2, 1 / 3)
        )
        assert (rule["abstention_precision"], rule["abstention_recall"]) == (1.0, 0.5)
    assert logprob_row["p_at_1"] == 0.5  # h4 ranked nothing
    assert summary["heldout"]["keys_without_dev_tau"] == [{"key": LOGPROB_KEY, "arms": [logprob], "searches": 3}]
    assert LOGPROB_KEY in readme
    assert "no dev" in readme.lower()
    assert [(e["run_id"], e["arm"], e["errors"]) for e in summary["dev"]["errors"] if e["errors"]] == []
    assert [(a["arm"], a["errors"]) for a in summary["heldout"]["arms"]] == [
        ("hybrid@3", 0),
        (jev, 1),
        (logprob, 0),
    ]


def test_heldout_rows_per_source_with_fallbacks_and_intervals(tmp_path: Path) -> None:
    jev = "hybrid+jev@3"
    arms = {jev: arm(3, "jev", JEV_ID)}
    dev = write_run(tmp_path / "dev", "dev", [decided(jev, "d1", best=0.9, none=0.0)], split="dev", arms=arms)
    # Plain: 6 positives, 4 right. Model: the same tasks, where m5 and m6 fell back to the plain request.
    plain = [decided(jev, f"m{i}", best=0.9, none=0.05, first_right=i <= 4) for i in range(1, 7)]
    negatives = [decided(jev, f"m{i}", best=0.9, none=0.05, gold=False, variant="negative") for i in range(1, 7)]
    model = [
        decided(jev, f"m{i}", best=0.9, none=0.05, first_right=i in (1, 5, 6), source="model", fallback=i >= 5)
        for i in range(1, 7)
    ]
    searches = [*plain, *negatives, *model]

    def heldout_report(name: str, repeat: int) -> dict[str, Any]:
        # A run that makes every search `repeat` times, with the same answer each time.
        records = [search | {"repeat": index} for search in searches for index in range(repeat)]
        run = write_run(
            tmp_path / name, name, records, split="heldout", arms=arms, repeat=repeat, sources=("plain", "model")
        )
        build_decision_report([dev], run, out_dir=tmp_path / name / "report")
        return report(tmp_path / name / "report")[0]

    summary = heldout_report("once", 1)

    by_source = rows(summary, jev)
    assert list(by_source) == ["plain", "model", "model (searched)"]
    assert [(row["searches"], row["positives"], row["fallbacks"]) for row in by_source.values()] == [
        (12, 6, None),
        (6, 6, 2),
        (4, 4, 0),
    ]
    assert [row["p_at_1"] for row in by_source.values()] == pytest.approx([4 / 6, 3 / 6, 1 / 4])
    # 95% intervals: 2,000 resamples of the tasks, seed 0, stated in the summary.
    assert summary["bootstrap"] == {"resamples": 2000, "seed": 0, "level": 0.95, "unit": "task"}
    hits = [1.0] * 4 + [0.0] * 2
    assert by_source["plain"]["p_at_1_ci95"] == list(bootstrap_ci(hits, resamples=2000, seed=0))
    # plain: 2 wrong positives and 6 answered negatives, each task's two searches drawn together
    wrong = [[0.0, 1.0]] * 4 + [[1.0, 1.0]] * 2
    assert by_source["plain"]["rules"]["reserved"]["wrong_tool_rate_ci95"] == list(
        cluster_bootstrap_ci(wrong, resamples=2000, seed=0)
    )
    # The same runs give the same files, and a run given as the wrong split is refused.
    build_decision_report([dev], tmp_path / "once" / "once", out_dir=tmp_path / "again")
    for name in ("summary.json", "README.md"):
        assert (tmp_path / "again" / name).read_bytes() == (tmp_path / "once" / "report" / name).read_bytes()
    with pytest.raises(ValueError, match="holds a heldout run, given as a dev run"):
        build_decision_report([tmp_path / "once" / "once"], None, out_dir=tmp_path / "wrong")
    # A search's repeats are resampled together: the same searches made twice give the same intervals.
    twice_summary = heldout_report("twice", 2)
    twice = rows(twice_summary, jev)
    assert (twice["plain"]["searches"], twice["plain"]["records"]) == (12, 24)
    assert twice["plain"]["p_at_1_ci95"] == by_source["plain"]["p_at_1_ci95"]
    assert (
        twice["plain"]["rules"]["reserved"]["wrong_tool_rate_ci95"]
        == by_source["plain"]["rules"]["reserved"]["wrong_tool_rate_ci95"]
    )
    # Searches are counted once, records once per repeat, in the rows and in the arm entries alike.
    assert [(row["searches"], row["records"]) for row in by_source.values()] == [(12, 12), (6, 6), (4, 4)]
    assert [(entry["searches"], entry["records"]) for entry in summary["heldout"]["arms"]] == [(18, 18)]
    assert [(entry["searches"], entry["records"]) for entry in twice_summary["heldout"]["arms"]] == [(18, 36)]
    note = "Rates are over records"
    assert note not in (tmp_path / "once" / "report" / "README.md").read_text()
    assert note in (tmp_path / "twice" / "report" / "README.md").read_text()


def one_arm_report(
    root: Path, searches: Sequence[dict[str, Any]], *, dev_searches: Sequence[dict[str, Any]] = ()
) -> Any:
    """Build the report of a dev run and a held-out run of Jev alone, and read it back."""
    arms = {JEV: arm(3, "jev", JEV_ID)}
    dev = write_run(root, "dev", dev_searches or [decided(JEV, "d1", best=0.9, none=0.0)], split="dev", arms=arms)
    heldout = write_run(root, "heldout", searches, split="heldout", arms=arms)
    build_decision_report([dev], heldout, out_dir=root / "report")
    return report(root / "report")


def test_paired_difference_interval_comes_from_paired_differences(tmp_path: Path) -> None:
    # Jev is right on exactly the tasks where retrieval is right (p1, p2) and wrong on the others.
    searches = [
        decided(JEV, task, best=0.9, none=0.05, first_right=right, retrieval_right=right)
        for task, right in (("p1", True), ("p2", True), ("p3", False), ("p4", False))
    ]

    summary, readme = one_arm_report(tmp_path, searches)

    row = rows(summary, JEV)["plain"]
    assert (row["p_at_1"], row["hybrid_p_at_1"]) == (0.5, 0.5)
    # Every paired difference is 0, so the interval has no width; two marginal intervals of 0.5 could not say so.
    assert (row["delta_p_at_1"], row["delta_p_at_1_ci95"]) == (0.0, [0.0, 0.0])
    low, high = row["p_at_1_ci95"]
    assert high - low >= 0.5
    assert "| arm | source | positives | P@1 (95% CI) | hybrid P@1 | Δ vs hybrid (95% CI) | ceiling |" in readme
    assert f"| {JEV} | plain | 4 | 0.500 ({low:.2f}-{high:.2f}) | 0.500 | +0.000 (0.00-0.00) | 1.000 |" in readme
    assert "the decider's P@1 minus retrieval's own P@1 on the same positives" in readme


def test_paired_difference_is_signed_and_its_interval_follows_the_pairs(tmp_path: Path) -> None:
    # Two tasks the decider gains (+1), three it loses (-1) and one both get wrong (0).
    kinds = {"g1": (True, False), "g2": (True, False), "l1": (False, True), "l2": (False, True), "l3": (False, True)}
    searches = [
        decided(JEV, task, best=0.9, none=0.05, first_right=won, retrieval_right=retrieval)
        for task, (won, retrieval) in kinds.items()
    ] + [decided(JEV, "n1", best=0.9, none=0.05)]

    summary, readme = one_arm_report(tmp_path, searches)

    row = rows(summary, JEV)["plain"]
    assert row["delta_p_at_1"] == pytest.approx(-1 / 6)
    assert row["delta_p_at_1_ci95"] == list(
        cluster_bootstrap_ci([[1.0], [1.0], [-1.0], [-1.0], [-1.0], [0.0]], resamples=2000, seed=0)
    )
    low, high = row["delta_p_at_1_ci95"]
    assert f"| -0.167 ({low:.2f}-{high:.2f}) |" in readme


def test_intervals_resample_tasks(tmp_path: Path) -> None:
    # n1 to n3 have no gold tool among their candidates: the negative repeats the positive, and both are wrong.
    # r1 to r3 have one: the positive is right and its negative is wrong.
    searches = [
        decided(JEV, task, best=0.9, none=0.05, gold=False, variant=variant)
        for task in ("n1", "n2", "n3")
        for variant in ("positive", "negative")
    ] + [
        decided(JEV, task, best=0.9, none=0.05, first_right=variant == "positive", gold=variant == "positive")
        for task in ("r1", "r2", "r3")
        for variant in ("positive", "negative")
    ]

    summary, _ = one_arm_report(tmp_path, searches)

    published = rows(summary, JEV)["plain"]["rules"]["reserved"]["wrong_tool_rate_ci95"]
    by_task = [[1.0, 1.0]] * 3 + [[0.0, 1.0]] * 3
    by_search = [[value] for cluster in by_task for value in cluster]
    resampling = {"resamples": 2000, "seed": 0}
    assert summary["bootstrap"] == resampling | {"level": 0.95, "unit": "task"}
    assert cluster_bootstrap_ci(by_search, **resampling) != cluster_bootstrap_ci(by_task, **resampling)
    assert published == list(cluster_bootstrap_ci(by_task, **resampling))


@pytest.mark.parametrize("case", ["fallback-in-the-pool", "no-fallback", "fallback-only-in-a-repeated-run"])
def test_notes_disclose_fallback_duplicates_only_when_the_pool_holds_one(tmp_path: Path, case: str) -> None:
    arms = {JEV: arm(3, "jev", JEV_ID)}

    def model_search(task: str, *, fallback: bool, repeat: int = 0) -> dict[str, Any]:
        return decided(
            JEV, task, best=0.9, none=0.05, first_right=True, source="model", fallback=fallback, repeat=repeat
        )

    pooled = [model_search("d1", fallback=case == "fallback-in-the-pool"), decided(JEV, "d1", best=0.9, none=0.05)]
    dev = write_run(tmp_path, "dev", pooled, split="dev", arms=arms, sources=("plain", "model"))
    # A run that repeats its searches does not feed the threshold choice, so its fallback does not count.
    repeated = [model_search("d2", fallback=True, repeat=index) for index in range(2)]
    determinism = write_run(tmp_path, "dev-repeat", repeated, split="dev", arms=arms, sources=("model",), repeat=2)
    dev_runs = [dev, *([determinism] if case == "fallback-only-in-a-repeated-run" else [])]

    build_decision_report(dev_runs, None, out_dir=tmp_path / "report")

    _, readme = report(tmp_path / "report")
    sentence = "count twice in U(τ) and in the searches counts of the thresholds table"
    assert (sentence in readme) is (case == "fallback-in-the-pool")


def cost_run(root: Path) -> Path:
    """A held-out run of the three deciders, two searches each, with known usage and timing."""
    jev, clm, logprob = "hybrid+jev@3", "hybrid+clm@3", "hybrid+logprob@3"
    arms = {
        "hybrid@3": arm(3),
        jev: arm(3, "jev", JEV_ID, price_input_per_mtok=0.05),
        clm: arm(3, "clm", CLM_ID),
        logprob: arm(3, "logprob", LOGPROB_ID, prompt_version="tool-choice-v1+letters-v1"),
    }
    records = [
        retrieved("hybrid@3", "h1"),
        retrieved("hybrid@3", "h2"),
        decided(jev, "h1", best=0.9, none=0.1, seconds=0.2, server_seconds=0.1, usage=(1, 1_000, 1)),
        decided(jev, "h2", best=0.9, none=0.1, seconds=0.8, server_seconds=0.3, usage=(1, 3_000, 1)),
        decided(clm, "h1", best=0.9, none=0.1, decider="clm", key=CLM_KEY, seconds=1.5, server_seconds=0.5),
        decided(clm, "h2", best=0.9, none=0.1, decider="clm", key=CLM_KEY, seconds=2.5, server_seconds=0.7),
        decided(clm, "h3", best=0.9, none=0.1, decider="clm", key=CLM_KEY, seconds=2.0, server_seconds=0.6),
        failed(clm, "h4", decider="clm", message=f"{CLM_ID}: HTTP 502 from https://{CLM_HOST}/v1/systemone"),
        decided(logprob, "h1", best=0.9, none=0.1, decider="logprob", key=LOGPROB_KEY, usage=(2, 4_000, 2)),
        decided(logprob, "h2", best=0.9, none=0.1, decider="logprob", key=LOGPROB_KEY, usage=(2, 6_000, 2)),
    ]
    arm_records = {clm: {"wall_seconds": 4.25, "cache_hits": 1, "cache_misses": 2}}
    return write_run(root, "heldout", records, split="heldout", arms=arms, arm_records=arm_records)


def test_clm_cost_has_busy_and_wall_figures(tmp_path: Path) -> None:
    dev = write_run(tmp_path, "dev", [], split="dev", arms={"hybrid@3": arm(3)})
    build_decision_report([dev], cost_run(tmp_path), out_dir=tmp_path / "report")
    summary, readme = report(tmp_path / "report")

    [row] = rows(summary, "hybrid+clm@3").values()
    assert (row["searches"], row["errors"]) == (3, 1)  # the failed search is left out of the figures
    per_1000 = 1000 * GPU_HOUR_USD / 3600
    # Busy: the server's own seconds per search. Wall: the calls' seconds per search, replays at the latency of
    # the call they replay.
    assert row["cost"]["clm_busy_usd_per_1000_searches"] == pytest.approx((0.5 + 0.7 + 0.6) / 3 * per_1000)
    assert row["cost"]["clm_wall_usd_per_1000_searches"] == pytest.approx((1.5 + 2.5 + 2.0) / 3 * per_1000)
    assert row["cost"]["usd_per_1000_searches"] is None  # Modal credits, not a billed price
    [clm_arm] = [a for a in summary["heldout"]["arms"] if a["arm"] == "hybrid+clm@3"]
    assert (clm_arm["wall_seconds"], clm_arm["cache_hits"], clm_arm["cache_misses"]) == (4.25, 1, 2)
    assert "busy" in readme.lower()
    assert "wall" in readme.lower()
    assert "4.2" in readme  # the arm's own clock time, next to the wall figure


def test_published_files_name_the_clm_host_modal(tmp_path: Path) -> None:
    clm = "hybrid+clm@3"
    dev_arms = {clm: arm(3, "clm", CLM_ID)}
    dev = write_run(
        tmp_path / "dev",
        "dev",
        [decided(clm, "d1", best=0.9, none=0.1, decider="clm", key=CLM_KEY)],
        split="dev",
        arms=dev_arms,
    )

    build_decision_report([dev], cost_run(tmp_path), out_dir=tmp_path / "report")

    for published in (tmp_path / "report").iterdir():
        text = published.read_text()
        assert CLM_HOST not in text, published.name
        assert "clm-latest@modal" in text, published.name
    summary, readme = report(tmp_path / "report")
    assert summary["dev"]["thresholds"][0]["key"] == CLM_KEY.replace(CLM_HOST, "modal")
    assert "bench/deploy/clm_modal.py" in readme
    assert "provisional" in readme  # the CLM figures wait for its authors to confirm the deployment


def test_decider_cost_and_latency_per_1000_searches(tmp_path: Path) -> None:
    dev = write_run(tmp_path, "dev", [], split="dev", arms={"hybrid@3": arm(3)})
    build_decision_report([dev], cost_run(tmp_path), out_dir=tmp_path / "report")
    summary, _ = report(tmp_path / "report")

    [jev] = rows(summary, "hybrid+jev@3").values()
    # 2,000 reported input tokens per search at the price the run's manifest declares, $0.05 per million.
    assert jev["cost"]["input_tokens_per_search"] == 2_000
    assert jev["cost"]["usd_per_1000_searches"] == pytest.approx(2_000 * 1000 * 0.05 / 1_000_000)
    [logprob] = rows(summary, "hybrid+logprob@3").values()
    price = calc_price(
        Usage(input_tokens=10_000, output_tokens=4), model_ref="gpt-4.1-mini-2025-04-14", provider_id="openai"
    )
    assert logprob["cost"]["asks_per_search"] == 2
    assert logprob["cost"]["usd_per_1000_searches"] == pytest.approx(
        float(price.total_price) / 2 * 1000  # pyright: ignore[reportUnknownMemberType]
    )
    [baseline] = rows(summary, "hybrid@3").values()
    assert (baseline["cost"]["usd_per_1000_searches"], baseline["latency_ms"]["decision"]) == (0.0, None)
    # Two searches: 200 and 800 ms of decision, 100 and 300 ms on the server; retrieval took 250 ms each time.
    assert jev["latency_ms"]["decision"] == pytest.approx({"p50": 500.0, "p95": 770.0})
    assert jev["latency_ms"]["server"] == pytest.approx({"p50": 200.0, "p95": 290.0})
    assert jev["latency_ms"]["retrieval"] == pytest.approx({"p50": 250.0, "p95": 250.0})


def test_report_checks_determinism_and_the_jev_tokenizer(tmp_path: Path) -> None:
    jev, clm, logprob = "hybrid+jev@3", "hybrid+clm@3", "hybrid+logprob@3"
    arms = {
        jev: arm(3, "jev", JEV_ID),
        clm: arm(3, "clm", CLM_ID),
        logprob: arm(3, "logprob", LOGPROB_ID),
    }

    def repeats(
        arm_name: str, decider: str, key: str, answers: Sequence[tuple[dict[str, float], float | None]]
    ) -> list[dict[str, Any]]:
        return [
            decided(
                arm_name, "t1", best=0.0, none=none, probabilities=probabilities, decider=decider, key=key, repeat=index
            )
            for index, (probabilities, none) in enumerate(answers)
        ]

    # Jev moves the reserved option most, by 0.03. Logprob leaves card y out of its final question once, which
    # ranks it at 0.0 against 0.5. CLM moves by 0.01, which is within the tolerance even though 0.6 - 0.59 is
    # 0.010000000000000009 in floating point; its failed repeat is left out.
    determinism = [
        *repeats(
            jev,
            "jev",
            JEV_KEY,
            [({"g": 0.70, "x": 0.20}, 0.10), ({"g": 0.69, "x": 0.21}, 0.10), ({"g": 0.70, "x": 0.20}, 0.13)],
        ),
        *repeats(
            logprob,
            "logprob",
            LOGPROB_KEY,
            [({"g": 0.5, "y": 0.5}, 0.0), ({"g": 0.5, "y": 0.5}, 0.0), ({"g": 0.55}, 0.0)],
        ),
        *repeats(clm, "clm", CLM_KEY, [({"g": 0.6, "x": 0.4}, 0.0), ({"g": 0.59, "x": 0.41}, 0.0)]),
        failed(clm, "t1", decider="clm", message="timeout") | {"repeat": 2},
    ]
    determinism_run = write_run(tmp_path, "dev-determinism", determinism, split="dev", arms=arms, repeat=3)
    # Jev's reported input tokens are 300 + 0.5 x the heuristic's estimate; CLM's exchanges are not Jev's.
    tokens = [
        decided(jev, "t1", best=0.9, none=0.1, exchanges=[(100, 350), (200, 400)]),
        decided(jev, "t2", best=0.9, none=0.1, exchanges=[(400, 500)]),
        decided(jev, "t3", best=0.9, none=0.1, exchanges=[(1_000, 800)]),
        decided(clm, "t1", best=0.9, none=0.1, decider="clm", key=CLM_KEY, exchanges=[(100, 5_000)]),
    ]
    main = write_run(tmp_path, "dev-main", tokens, split="dev", arms=arms)

    build_decision_report([main, determinism_run], None, out_dir=tmp_path / "report")
    summary, readme = report(tmp_path / "report")

    checked = {entry["arm"]: entry for entry in summary["dev"]["determinism"]}
    assert {name: entry["run_id"] for name, entry in checked.items()} == dict.fromkeys(arms, "dev-determinism")
    assert checked[jev]["max_abs_delta_p"] == pytest.approx(0.03)
    assert checked[logprob]["max_abs_delta_p"] == pytest.approx(0.5)
    assert checked[clm]["max_abs_delta_p"] == pytest.approx(0.01)
    assert [checked[name]["within_tolerance"] for name in (jev, logprob, clm)] == [False, False, True]
    assert [(e["run_id"], e["arm"], e["errors"]) for e in summary["dev"]["errors"] if e["errors"]] == [
        ("dev-determinism", clm, 1)
    ]

    [fit] = summary["tokenizer"]
    assert fit["split"] == "dev"
    assert (fit["exchanges"], fit["intercept"], fit["slope"]) == (4, pytest.approx(300.0), pytest.approx(0.5))
    # The reported / estimated ratios are 3.5, 2.0, 1.25 and 0.8.
    assert (fit["median_ratio"], fit["share_under_1"]) == (pytest.approx(1.625), 0.25)
    assert summary["heldout"] is None
    assert "held-out" in readme.lower()


JEV = "hybrid+jev@3"
LOGPROB = "hybrid+logprob@3"
# What a run started with --bypass-cache records for each decider arm's model.
BYPASSED = "CachedDecisionModel(FakeModel(), bypass=True)"


def edit_manifest(run_dir: Path, edit: Callable[[dict[str, Any]], object]) -> None:
    path = run_dir / "manifest.json"
    manifest: dict[str, Any] = json.loads(path.read_text())
    edit(manifest)
    path.write_text(json.dumps(manifest))


def dev_and_main(root: Path) -> tuple[Path, Path]:
    """A dev run whose only threshold is tau = 0.5 for Jev's key, and a held-out run that asks each search once."""
    arms = {JEV: arm(3, "jev", JEV_ID)}
    # A right answer at 0.80 and a negative answered at 0.45: U is 0 up to 0.45 and 1/2 from 0.50 to 0.80.
    dev_searches = [
        decided(JEV, "d1", best=0.8, none=0.1, first_right=True),
        decided(JEV, "d2", best=0.45, none=0.1, gold=False, variant="negative"),
    ]
    main_searches = [
        decided(JEV, "h1", best=0.9, none=0.05, first_right=True, exchanges=[(100, 350)]),
        decided(JEV, "h2", best=0.9, none=0.05),
        decided(JEV, "h3", best=0.9, none=0.05, gold=False, variant="negative"),
    ]
    dev = write_run(root, "dev", dev_searches, split="dev", arms=arms)
    return dev, write_run(root, "heldout-main", main_searches, split="heldout", arms=arms)


def repeats_run(root: Path, *, split: str = "heldout", repeat: int = 3, bypassed: str = BYPASSED) -> Path:
    """A run that makes each search three times: Jev on h1, h2 and a negative h3, logprob on h1, and a failed h4."""

    def jev(task: str, index: int, probabilities: dict[str, float], *, none: float, negative: bool = False) -> Any:
        variant = "negative" if negative else "positive"
        return decided(
            JEV,
            task,
            best=0.0,
            none=none,
            probabilities=probabilities,
            gold=not negative,
            variant=variant,
            repeat=index,
            exchanges=[(400, 500)],
        )

    def logprob(index: int, probabilities: dict[str, float]) -> Any:
        return decided(
            LOGPROB,
            "h1",
            best=0.0,
            none=None,
            probabilities=probabilities,
            decider="logprob",
            key=LOGPROB_KEY,
            repeat=index,
        )

    # Only the first card's probability and the reserved option's decide an outcome; the others make the spreads.
    h1 = [{"g": 0.80, "x": 0.10, "y": 0.05}, {"g": 0.70, "x": 0.20, "y": 0.05}, {"g": 0.40, "x": 0.45, "y": 0.10}]
    h2 = [{"g": 0.55, "x": 0.30, "y": 0.10}, {"g": 0.30, "x": 0.60, "y": 0.05}, {"g": 0.30, "x": 0.55, "y": 0.10}]
    h3 = [{"x": 0.80, "y": 0.05, "z": 0.05}, {"x": 0.20, "y": 0.05, "z": 0.05}, {"x": 0.50, "y": 0.05, "z": 0.05}]
    answers = [{"g": 0.30, "x": 0.20, "y": 0.10}, {"g": 0.30, "x": 0.20, "y": 0.10}, {"g": 0.25, "x": 0.35, "y": 0.10}]
    records = [
        *(jev("h1", index, probabilities, none=0.05) for index, probabilities in enumerate(h1)),
        *(jev("h2", index, probabilities, none=0.05) for index, probabilities in enumerate(h2)),
        *(jev("h3", index, probabilities, none=0.10, negative=True) for index, probabilities in enumerate(h3)),
        failed(JEV, "h4", decider="jev", message=f"{JEV_ID}: HTTP 500") | {"repeat": 1},
        *(logprob(index, probabilities) for index, probabilities in enumerate(answers)),
    ]
    arms = {
        JEV: arm(3, "jev", JEV_ID, model=bypassed),
        LOGPROB: arm(3, "logprob", LOGPROB_ID, model=bypassed),
    }
    records.reverse()  # the report orders repeats by their index, not by where their records sit in the file
    return write_run(root, "heldout-repeats", records, split=split, arms=arms, repeat=repeat)


def test_heldout_repeats_report_variation_per_repeat(tmp_path: Path) -> None:
    dev, main = dev_and_main(tmp_path)
    repeats = repeats_run(tmp_path)

    build_decision_report([dev], main, out_dir=tmp_path / "report", heldout_repeats=repeats)
    build_decision_report([dev], main, out_dir=tmp_path / "alone")
    summary, readme = report(tmp_path / "report")

    variation = summary["heldout"]["repeats"]
    assert (variation["run_id"], variation["repeat"]) == ("heldout-repeats", 3)
    jev_row, logprob_row = variation["rows"]
    assert (jev_row["arm"], jev_row["decider"], jev_row["k"], jev_row["source"]) == (JEV, "jev", 3, "plain")
    # Three searches made three times, and a failed one that is in neither count.
    assert (jev_row["searches"], jev_row["records"], jev_row["errors"]) == (3, 9, 1)
    # At the dev tau of 0.50, best card and outcome per repeat (h1, h2, h3):
    #   0: right and answered, right and answered, a negative answered at 0.80: U = (2 - 1) / 3
    #   1: right, wrong (x leads at 0.60), a negative at 0.20 that abstains:    U = (1 - 1) / 3
    #   2: h1 abstains at 0.45, h2 wrong, the negative answered at exactly 0.50: U = (0 - 2) / 3
    per_repeat = jev_row["per_repeat"]
    assert [entry["repeat"] for entry in per_repeat] == [0, 1, 2]
    assert [entry["p_at_1"] for entry in per_repeat] == [1.0, 0.5, 0.0]
    frozen = [entry["reserved_and_dev_tau"] for entry in per_repeat]
    assert [rule["coverage"] for rule in frozen] == pytest.approx([1.0, 2 / 3, 2 / 3])
    assert [rule["wrong_tool_rate"] for rule in frozen] == pytest.approx([1 / 3, 1 / 3, 2 / 3])
    assert [rule["utility"] for rule in frozen] == pytest.approx([1 / 3, 0.0, -2 / 3])
    assert jev_row["p_at_1_range"] == [0.0, 1.0]
    assert jev_row["utility_range"] == pytest.approx([-2 / 3, 1 / 3])
    # The negative h3 moves most: card x goes 0.80, 0.20, 0.50.
    assert jev_row["largest_abs_dp"] == pytest.approx(0.6)
    assert jev_row["at"] == {"task": "h3", "variant": "negative"}
    assert jev_row["main_p_at_1"] == rows(summary, JEV)["plain"]["p_at_1"] == 0.5

    # Logprob has no dev tau, so every answer stands (a tau of 0.5 would have abstained on all of them), and the
    # main run has no logprob arm to compare with.
    assert [entry["p_at_1"] for entry in logprob_row["per_repeat"]] == [1.0, 1.0, 0.0]
    assert [entry["reserved_and_dev_tau"]["utility"] for entry in logprob_row["per_repeat"]] == [1.0, 1.0, -1.0]
    assert logprob_row["utility_range"] == [-1.0, 1.0]
    assert logprob_row["largest_abs_dp"] == pytest.approx(0.15)
    assert logprob_row["at"] == {"task": "h1", "variant": "positive"}
    assert logprob_row["main_p_at_1"] is None

    # The main tables, and the token check of Jev's exchanges, still come from the main run alone.
    assert summary["heldout"] | {"repeats": None} == report(tmp_path / "alone")[0]["heldout"]
    assert summary["tokenizer"] == report(tmp_path / "alone")[0]["tokenizer"]
    assert [entry["exchanges"] for entry in summary["tokenizer"]] == [0, 1]
    assert [(entry["split"], entry["role"], entry["run_id"]) for entry in summary["runs"]] == [
        ("dev", "main", "dev"),
        ("heldout", "main", "heldout-main"),
        ("heldout", "repeats", "heldout-repeats"),
    ]
    assert "| heldout | repeats | `heldout-repeats` |" in readme
    assert (
        readme.index("### Cost per 1,000 searches") < readme.index("### Run-to-run variation") < readme.index("## Dev")
    )
    assert "The repeats run `heldout-repeats` makes each search 3 times with the decision cache bypassed." in readme
    assert "| hybrid+jev@3 | plain | 3 | 1.000 / 0.500 / 0.000 | 0.500 | 0.333 / 0.000 / -0.667 | 0.6000 |" in readme
    assert "| hybrid+logprob@3 | plain | 1 | 1.000 / 1.000 / 0.000 | n/a | 1.000 / 1.000 / -1.000 | 0.1500 |" in readme


@pytest.mark.parametrize(
    "case", ["no-heldout-run", "dev-split", "single-repeat", "other-tasks", "other-dataset", "cache-not-bypassed"]
)
def test_heldout_repeats_rejects_mismatched_runs(tmp_path: Path, case: str) -> None:
    dev, main = dev_and_main(tmp_path)
    repeats = repeats_run(
        tmp_path,
        split="dev" if case == "dev-split" else "heldout",
        repeat=1 if case == "single-repeat" else 3,
        bypassed=BYPASSED.replace("True", "False") if case == "cache-not-bypassed" else BYPASSED,
    )
    if case == "other-tasks":
        edit_manifest(repeats, lambda manifest: manifest["tasks"].update(sha256="cd" * 32))
    if case == "other-dataset":
        edit_manifest(repeats, lambda manifest: manifest["dataset"].update(revision="0" * 40))

    with pytest.raises(ValueError, match="heldout-repeats") as error:
        build_decision_report(
            [dev], None if case == "no-heldout-run" else main, out_dir=tmp_path / "report", heldout_repeats=repeats
        )

    assert str(repeats) in str(error.value)
    assert not (tmp_path / "report").exists()  # nothing is written for a refused run


def test_decision_report_command_takes_a_repeats_run(tmp_path: Path) -> None:
    dev, main = dev_and_main(tmp_path)
    repeats = repeats_run(tmp_path)
    arguments = ["decision-report", "--dev", str(dev), "--heldout", str(main), "--out", str(tmp_path / "report")]

    result = CliRunner().invoke(cli.app, [*arguments, "--heldout-repeats", str(repeats)])

    assert result.exit_code == 0, result.output
    assert report(tmp_path / "report")[0]["heldout"]["repeats"]["run_id"] == "heldout-repeats"
    missing = CliRunner().invoke(cli.app, [*arguments, "--heldout-repeats", str(tmp_path / "nowhere")])
    assert missing.exit_code == 2  # the option is checked like the other run directories
