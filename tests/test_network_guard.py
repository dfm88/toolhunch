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
    pytestconfig: pytest.Config, pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    # This session loads the repository's root conftest.py: it is what covers `tests/` and `bench/tests/`.
    assert any(
        Path(getattr(plugin, "__file__", "")).resolve() == ROOT_CONFTEST
        for plugin in pytestconfig.pluginmanager.get_plugins()
    )

    # The inner session runs a copy of that file under the same socket options, against an offline test that must
    # stay blocked and a live test that does nothing remote: it only creates (never connects) a network socket.
    pytester.makeconftest(ROOT_CONFTEST.read_text())
    pytester.makeini("[pytest]\nmarkers =\n    live: paid live test\n")
    pytester.makepyfile(
        """
        import socket

        import pytest
        from pytest_socket import SocketBlockedError

        def test_offline():
            with pytest.raises(SocketBlockedError):
                socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        @pytest.mark.live
        def test_live():
            socket.socket(socket.AF_INET, socket.SOCK_STREAM).close()
        """
    )
    options = ("--disable-socket", "--allow-unix-socket")

    monkeypatch.delenv("TOOLHUNCH_LIVE", raising=False)
    skipped = pytester.runpytest_subprocess(*options, "-m", "live", "-rs", timeout=60)
    skipped.assert_outcomes(skipped=1)
    skipped.stdout.fnmatch_lines(["SKIPPED*paid live test: set TOOLHUNCH_LIVE=1 in the shell to run it"])

    monkeypatch.setenv("TOOLHUNCH_LIVE", "1")  # reaches only the inner session's subprocess
    pytester.runpytest_subprocess(*options, timeout=60).assert_outcomes(passed=2)
