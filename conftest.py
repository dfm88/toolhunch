"""Rules every test session follows, `tests/` and `bench/tests/` alike: a paid live test needs an explicit opt-in.

Keep this file to the standard library and pytest: `tests/test_network_guard.py` copies it verbatim into an inner
pytest session, where nothing else of this repository is importable.
"""

import os

import pytest

LIVE_OPT_IN = "TOOLHUNCH_LIVE"
LIVE_SKIP_REASON = f"paid live test: set {LIVE_OPT_IN}=1 in the shell to run it"

# Read when pytest imports this file, before any test module: code that loads `.env` later (a fixture, or a module
# imported during collection) cannot opt a session in, so a value written there never counts.
OPTED_IN = os.environ.get(LIVE_OPT_IN) == "1"


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Give live tests the network, and skip them unless the shell opted in.

    Without the opt-in, `-m live` still skips every live test: keys alone never spend money.
    """
    for item in items:
        if item.get_closest_marker("live") is None:
            continue
        item.add_marker(pytest.mark.enable_socket)  # offline tests run under --disable-socket
        if not OPTED_IN:
            item.add_marker(pytest.mark.skip(reason=LIVE_SKIP_REASON))
