from importlib.metadata import version

import toolhunch


def test_version_matches_distribution_metadata() -> None:
    assert toolhunch.__version__ == version("toolhunch")
