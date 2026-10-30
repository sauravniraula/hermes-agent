"""Profile-scoped capability listings and estimated token costs."""

import json
import os
from pathlib import Path

from hermes_cli.config import atomic_config_write


def test_tools_breakdown_partitions_active_tools_and_reports_token_sizes(tmp_path, monkeypatch, capsys):
    from hermes_cli.main import _build_cli_parser

    monkeypatch.chdir(tmp_path)
    home = Path(os.environ["HERMES_HOME"])
    atomic_config_write(home / "config.yaml", {
        "agent": {"enabled_toolsets": ["file", "skills"]},
        "tools": {"tool_search": {"eager_toolsets": ["file"]}},
    })
    parser, _ = _build_cli_parser()
    args = parser.parse_args(["tools", "breakdown", "--json"])
    assert args.func(args) == 0
    data = json.loads(capsys.readouterr().out)
    rows = {row["name"]: row for row in data["tools"]}
    assert rows["read_file"]["disclosure"] == "direct"
    assert rows["skill_view"]["disclosure"] == "deferred"
    assert rows["read_file"]["tokens"] > 0
    assert rows["skill_view"]["tokens"] > 0
    assert rows["skill_view"]["upfront_tokens"] == 0
    assert "terminal" not in rows
    assert data["counts"]["active"] == len(rows)
    assert data["counts"]["active"] == data["counts"]["direct"] + data["counts"]["deferred"]
    assert data["token_method"] == "estimate_tokens_rough"


def test_skills_breakdown_filters_platform_allowlists_and_sizes_each_skill(tmp_path, monkeypatch, capsys):
    from hermes_cli.main import _build_cli_parser

    monkeypatch.chdir(tmp_path)
    home = Path(os.environ["HERMES_HOME"])
    atomic_config_write(home / "config.yaml", {
        "agent": {"enabled_toolsets": ["skills"]},
        "skills": {"enabled": ["alpha", "beta", "gamma"], "disabled": ["gamma"],
                   "platform_enabled": {"whatsapp": ["beta"]}},
    })
    for name in ("alpha", "beta", "gamma"):
        path = home / "skills" / name / "SKILL.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"---\nname: {name}\ndescription: Workflow for {name}.\n---\n\n# {name}\n" + "Steps. " * 30)
    parser, _ = _build_cli_parser()
    args = parser.parse_args(["skills", "breakdown", "--platform", "whatsapp", "--json"])
    assert args.func(args) == 0
    data = json.loads(capsys.readouterr().out)
    assert [r["name"] for r in data["skills"]] == ["beta"]
    assert data["count"] == 1
    assert data["skills"][0]["index_tokens"] > 0
    assert data["skills"][0]["file_tokens"] > data["skills"][0]["index_tokens"]
    assert data["token_method"] == "estimate_tokens_rough"


def test_tool_breakdown_includes_agent_injected_direct_tools(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from hermes_cli.capability_breakdown import compute_tools_breakdown

    monkeypatch.chdir(tmp_path)
    atomic_config_write(Path(os.environ["HERMES_HOME"]) / "config.yaml", {"agent": {"enabled_toolsets": []}})
    closed = []
    injected = {"type": "function", "function": {"name": "provider_memory_lookup", "description": "Search notes.",
                "parameters": {"type": "object", "properties": {}}}}
    monkeypatch.setattr("hermes_cli.prompt_size._build_inspection_agent", lambda _: SimpleNamespace(
        tools=[injected], close=lambda: closed.append(True)))
    data = compute_tools_breakdown()
    assert len(data["tools"]) == 1
    row = data["tools"][0]
    assert row["name"] == "provider_memory_lookup" and row["disclosure"] == "direct"
    assert row["upfront_tokens"] == row["tokens"] > 0
    assert closed == [True]


def test_disabled_skills_toolset_has_no_upfront_index_cost(tmp_path, monkeypatch):
    from hermes_cli.capability_breakdown import compute_skills_breakdown, _render_skills

    monkeypatch.chdir(tmp_path)
    home = Path(os.environ["HERMES_HOME"])
    atomic_config_write(home / "config.yaml", {
        "agent": {"enabled_toolsets": []}, "skills": {"enabled": ["alpha"]},
    })
    skill = home / "skills/alpha/SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text("---\nname: alpha\ndescription: Alpha workflow.\n---\n# Alpha\nSteps.")
    data = compute_skills_breakdown()
    assert data["skills_toolset_enabled"] is False
    assert data["index_total_tokens"] == 0
    assert data["skills"][0]["index_tokens"] == 0
    assert data["skills"][0]["file_tokens"] > 0
    rendered = _render_skills(data)
    assert "estimated tokens" in rendered and "alpha" in rendered


def test_tools_breakdown_with_search_off_has_no_deferred_tools(tmp_path, monkeypatch):
    from hermes_cli.capability_breakdown import compute_tools_breakdown, _render_tools

    monkeypatch.chdir(tmp_path)
    atomic_config_write(Path(os.environ["HERMES_HOME"]) / "config.yaml", {
        "agent": {"enabled_toolsets": ["file"]}, "tools": {"tool_search": {"enabled": "off"}},
    })
    data = compute_tools_breakdown()
    assert data["counts"]["deferred"] == 0
    assert data["counts"]["direct"] == data["counts"]["active"]
    assert all(r["upfront_tokens"] == r["tokens"] > 0 for r in data["tools"])
    rendered = _render_tools(data)
    assert "estimated tokens" in rendered and "read_file" in rendered


def test_platform_scope_restores_previous_platform_on_failure(monkeypatch):
    import pytest
    from hermes_cli.inspection import platform_scope

    monkeypatch.setenv("HERMES_PLATFORM", "tui")
    with pytest.raises(RuntimeError, match="stop"):
        with platform_scope("cli"):
            assert os.environ["HERMES_PLATFORM"] == "cli"
            raise RuntimeError("stop")
    assert os.environ["HERMES_PLATFORM"] == "tui"
