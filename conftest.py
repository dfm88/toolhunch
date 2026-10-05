"""Rules every test session follows, `tests/` and `bench/tests/` alike: a paid live test needs an explicit opt-in."""

import os

import pytest

LIVE_OPT_IN = "TOOLHUNCH_LIVE"
LIVE_SKIP_REASON = f"paid live test: set {LIVE_OPT_IN}=1 in the shell to run it"


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Give live tests the network, and skip them unless the shell opted in.

    The opt-in is read from the process environment at collection, before any fixture loads `.env`, so a value
    written there never counts. Without it, `-m live` still skips every live test: keys alone never spend money.
    """
    opted_in = os.environ.get(LIVE_OPT_IN) == "1"
    for item in items:
        if item.get_closest_marker("live") is None:
            continue
        item.add_marker(pytest.mark.enable_socket)  # offline tests run under --disable-socket
        if not opted_in:
            item.add_marker(pytest.mark.skip(reason=LIVE_SKIP_REASON))
