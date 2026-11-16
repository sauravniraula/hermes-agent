"""Default display preferences must agree across startup and config loaders."""

from types import SimpleNamespace

import pytest

from hermes_cli.config import atomic_config_write, load_config
from hermes_cli.cli_config_load import _cli_config_defaults
from hermes_cli import main


@pytest.mark.parametrize("override", [None, {"interface": "cli", "compact": False, "skin": "default"}])
def test_display_defaults_and_explicit_overrides(tmp_path, monkeypatch, override):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_TUI", raising=False)
    monkeypatch.setattr(main, "_EARLY_INTERFACE_CACHE", None)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    if override is not None:
        atomic_config_write(tmp_path / "config.yaml", {"display": override})
    expected = override or {"interface": "tui", "compact": True, "skin": "mono"}
    config = load_config()
    cli_display = _cli_config_defaults()["display"]
    for key, value in expected.items():
        assert config["display"][key] == value
        if override is None:
            assert cli_display[key] == value
    assert main._config_default_interface_early() == expected["interface"]
    args = SimpleNamespace(cli=False, tui=False)
    assert main._resolve_use_tui(args) is (expected["interface"] == "tui")
    assert main._wants_tui_early([]) is (expected["interface"] == "tui")
    assert main._resolve_use_tui(SimpleNamespace(cli=True, tui=False)) is False
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    assert main._resolve_use_tui(args) is False
    assert main._wants_tui_early([]) is False
