# AGENTS.md

Guidance for coding agents (and humans) working in this repository. `CLAUDE.md` is a symlink to this
file.

## What this is

`toolhunch` is a Python library that ranks an agent's tools for a request: a cheap retriever for
recall, then an optional System-1 decision model (TypeSafe's Jev or a compatible endpoint), a
reranker or an LLM for precision, with abstention. It plugs into Pydantic AI and ships a benchmark
that measures prompt caching, not just tokens.

Status: alpha.

## Commands

```shell
uv sync --all-packages              # core + bench + dev tools
uv run pytest                       # unit tests (no network)
uv run pytest -m live               # live tests against paid APIs; need keys in .env
uv run ruff format . && uv run ruff check --fix .
uv run pyright                      # strict
uv run --group docs mkdocs serve    # docs
```

Pre-commit runs ruff and pyright: `uvx pre-commit install` once.

## Layout

```text
src/toolhunch/        the library (published)
bench/                benchmark harness, a uv workspace member (never published to PyPI)
tests/                library tests
docs/                 public user documentation (mkdocs)
```

Create a module when its code lands; do not add empty placeholders.

## Architecture rules

- **Core stays framework-free.** Nothing under `src/toolhunch/` outside `integrations/` imports a
  framework. Integrations depend on core, never the reverse.
- **Core dependencies are `pydantic`, `httpx2` and `anyio` only.** Anything else is an extra, imported
  lazily, with an error naming the extra.
- **Ranking and disclosure are separate**: pipelines rank; reveal/preselect/proxy decide how tools
  reach the model.
- **Model limits are declared data** with a source and a date, overridable per instance. Never
  hard-code a vendor limit inside planner logic.
- **Card text is untrusted.** Fence it as data; never splice it into instructions.
- **Probabilities are only comparable within one question.** Thresholds belong to one model × prompt
  version × payload shape; record that key wherever a threshold is applied.
- I/O is async and anyio-compatible.

## Code conventions

- Python ≥ 3.12 syntax: `list[str]`, `str | None`, PEP 695 generics where they help.
- Keyword arguments at call sites; public functions take keyword-only parameters after the first
  obvious one (`*`).
- Pydantic models at boundaries (wire payloads, configuration); frozen `dataclass(slots=True)` for
  internal values on hot paths.
- Group functions that share parameters into a class instead of a module of loose functions. Keep
  single-use transformations next to their caller; extract a helper only when it earns it.
- Docstrings (Google style) on public API; inline comments only for non-obvious decisions.
- Reuse existing types before adding new ones; search `src/` first.

## Testing

- Every library change comes with tests. `tests/` never touches the network: HTTP adapters are tested
  with `httpx2.MockTransport` and recorded payloads.
- Live tests carry `@pytest.mark.live` and skip when their key is missing.
- `inline-snapshot` for structured outputs; update snapshots deliberately
  (`--inline-snapshot=fix`), never blindly.

## Benchmark rules

- Pilot (50 tasks) before any full run; report cost before scaling.
- Pin model versions, prompt versions, payload shapes, catalog fingerprints in every result.
- Raw runs go to `bench/runs/` (git-ignored); curated results to `bench/results/`. Never hand-edit a
  result file. Publish results that contradict the hypotheses too.
- Every paid run appends to the cost ledger.

## Secrets

Keys live in `.env` (git-ignored); variable names are in `.env.example`. Never print, log or commit
key values, and never read `.env` into a transcript — load it in code.

## Documentation

Repository language is English.

## Git

Conventional commits (`feat:`, `fix:`, `docs:`, `bench:`, `chore:`). Commit or push only when asked.

## Local instructions

Maintainers may keep extra instructions in a git-ignored `AGENTS.local.md` at the repository root.
If it exists and is not already in your context, read it before starting and treat it as an
extension of this file; if it does not exist, ignore this section. Claude Code loads it on its own
when `CLAUDE.local.md` is a symlink to it.
