"""Management surfaces report skill availability without widening configured allowlists."""

from io import StringIO

import pytest
from rich.console import Console

from hermes_cli.config import atomic_config_replace


@pytest.fixture
def catalog(tmp_path, monkeypatch):
    from tools import skills_tool

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_PLATFORM", raising=False)
    monkeypatch.delenv("HERMES_SESSION_PLATFORM", raising=False)
    monkeypatch.setattr(skills_tool, "SKILLS_DIR", tmp_path / "skills")
    for name in ("alpha", "beta", "hermes-agent"):
        directory = tmp_path / "skills" / name
        directory.mkdir(parents=True)
        (directory / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: Test.\n---\n{name} instructions\n", encoding="utf-8",
        )
    atomic_config_replace(tmp_path / "config.yaml", {"skills": {"enabled": ["alpha"]}})
    return tmp_path


def test_cli_enabled_only_listing_honors_allowlist(catalog):
    from hermes_cli.skills_hub import do_list

    sink = StringIO()
    do_list(enabled_only=True, console=Console(file=sink, force_terminal=False))
    output = sink.getvalue()
    assert "alpha" in output and "hermes-agent" in output and "beta" not in output


def test_native_menu_explicit_platform_applies_allowlist(catalog):
    from hermes_cli.commands_platforms import _iter_gateway_skills

    atomic_config_replace(catalog / "config.yaml", {
        "skills": {"platform_enabled": {"telegram": ["alpha"], "discord": ["beta"]}},
    })
    assert {row[1]["name"] for row in _iter_gateway_skills("telegram")} == {"alpha", "hermes-agent"}
    assert {row[1]["name"] for row in _iter_gateway_skills("discord")} == {"beta", "hermes-agent"}


def test_dashboard_marks_excluded_skill_disabled_and_refuses_reenable(catalog):
    from starlette.testclient import TestClient
    from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

    client = TestClient(app)
    client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    rows = client.get("/api/skills").json()
    assert {row["name"] for row in rows if row["enabled"]} == {"alpha", "hermes-agent"}
    before = (catalog / "config.yaml").read_bytes()
    response = client.put("/api/skills/toggle", json={"name": "beta", "enabled": True})
    assert response.status_code == 409
    assert "skills.enabled" in response.json()["detail"]
    assert (catalog / "config.yaml").read_bytes() == before
    assert client.put("/api/skills/toggle", json={"name": "alpha", "enabled": False}).status_code == 200
    assert client.put("/api/skills/toggle", json={"name": "alpha", "enabled": True}).status_code == 200


def test_cli_toggle_excludes_skills_outside_allowlist(catalog, monkeypatch):
    from hermes_cli import curses_ui, skills_config

    monkeypatch.setattr(skills_config, "_select_platform", lambda: None)
    monkeypatch.setattr("builtins.input", lambda _: "1")
    labels = []

    def choose(_title, items, selected, **_kwargs):
        labels.extend(items)
        return selected

    monkeypatch.setattr(curses_ui, "curses_checklist", choose)
    before = (catalog / "config.yaml").read_bytes()
    skills_config.skills_command()
    assert any("alpha" in label for label in labels)
    assert not any("beta" in label for label in labels)
    assert (catalog / "config.yaml").read_bytes() == before
