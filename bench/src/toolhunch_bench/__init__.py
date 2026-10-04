"""Benchmark harness for toolhunch (never published): datasets, retrieval runs and reports, `toolhunch-bench`."""

from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parents[2]
"""The `bench/` directory of the repository: tasks, results and git-ignored runs live under it."""


def without_local_root(text: str) -> str:
    """`text` with this checkout's absolute root removed, so a published file names repository paths only.

    A local server's health answer can name its files by absolute path, which holds the maintainer's home directory.
    """
    return text.replace(f"{BENCH_DIR.parent}/", "")
