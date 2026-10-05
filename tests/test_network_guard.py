import socket
from pathlib import Path

import pytest
from pytest_socket import SocketBlockedError

pytest_plugins = ["pytester"]

ROOT_CONFTEST = Path(__file__).resolve().parents[1] / "conftest.py"


@pytest.mark.filterwarnings("ignore:A test tried to use socket")  # pytest-socket warns as it raises
def test_offline_tests_cannot_open_a_network_socket() -> None:
    # The block fires when the socket is created, so nothing is sent to port 9 (discard).
    with pytest.raises(SocketBlockedError):
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(("127.0.0.1", 9))


def test_live_tests_run_only_with_the_opt_in_from_the_shell(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The inner session runs the repository's own conftest.py against a live test that does nothing remote.
    pytester.makeconftest(ROOT_CONFTEST.read_text())
    pytester.makeini("[pytest]\nmarkers =\n    live: paid live test\n")
    pytester.makepyfile(
        """
        import pytest

        @pytest.mark.live
        def test_live(request):
            assert request.node.get_closest_marker("enable_socket") is not None
        """
    )

    monkeypatch.delenv("TOOLHUNCH_LIVE", raising=False)
    skipped = pytester.runpytest_subprocess("-m", "live", "-rs")
    skipped.assert_outcomes(skipped=1)
    skipped.stdout.fnmatch_lines(["SKIPPED*paid live test: set TOOLHUNCH_LIVE=1 in the shell to run it"])

    monkeypatch.setenv("TOOLHUNCH_LIVE", "1")  # reaches only the inner session's subprocess
    pytester.runpytest_subprocess("-m", "live").assert_outcomes(passed=1)
