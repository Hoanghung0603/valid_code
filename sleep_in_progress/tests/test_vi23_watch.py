"""The watchdog must stop only its own job after consecutive failed probes."""
import importlib.util
from pathlib import Path
from unittest.mock import Mock


spec = importlib.util.spec_from_file_location(
    "vi23_watch", Path(__file__).parents[1] / "deploy" / "vi23_watch.py")
watch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(watch)


def test_repeated_slow_probe_stops_child(monkeypatch):
    child = Mock(pid=4321)
    child.poll.return_value = None
    monkeypatch.setattr(watch.subprocess, "Popen", Mock(return_value=child))
    run = Mock()
    monkeypatch.setattr(watch.subprocess, "run", run)
    monkeypatch.setattr(watch, "probe", Mock(side_effect=[True, False, False]))
    clock = iter([0, 31, 31, 62])
    monkeypatch.setattr(watch.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(watch.time, "sleep", lambda _: None)
    monkeypatch.setattr("sys.argv", ["vi23_watch.py", "--", "python", "job.py"])
    child.wait.side_effect = lambda: setattr(child.poll, "return_value", 1)
    assert watch.main() == 2
    run.assert_called_once_with(["taskkill", "/PID", "4321", "/T", "/F"], check=False)


def test_initial_probe_failure_does_not_launch(monkeypatch):
    monkeypatch.setattr(watch, "probe", Mock(return_value=False))
    monkeypatch.setattr(watch.time, "sleep", lambda _: None)
    launch = Mock()
    monkeypatch.setattr(watch.subprocess, "Popen", launch)
    monkeypatch.setattr("sys.argv", ["vi23_watch.py", "--", "python", "job.py"])
    assert watch.main() == 2
    launch.assert_not_called()
