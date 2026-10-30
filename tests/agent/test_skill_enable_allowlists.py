"""Profile/platform allowlists gate discovery and explicit skill activation alike."""

import json

import pytest

from hermes_cli.config import atomic_config_replace


def _write_skill(root, name, *, directory=None, body=None):
    skill_dir = root / (directory or name)
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Test {name}.\n---\n{body or name + ' content'}\n",
        encoding="utf-8",
    )
    reference = skill_dir / "references" / "guide.md"
    reference.parent.mkdir(exist_ok=True)
    reference.write_text(f"{name} reference", encoding="utf-8")
    return skill_dir


@pytest.fixture
def skill_env(tmp_path, monkeypatch):
    from tools import skills_tool

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_PLATFORM", raising=False)
    monkeypatch.delenv("HERMES_SESSION_PLATFORM", raising=False)
    monkeypatch.setattr(skills_tool, "SKILLS_DIR", tmp_path / "skills")
    for name in ("alpha", "beta", "gamma", "hermes-agent"):
        _write_skill(tmp_path / "skills", name)
    skills_tool.clear_skills_cache()

    def configure(**skills):
        atomic_config_replace(tmp_path / "config.yaml", {"skills": skills})

    configure()
    return tmp_path, configure


def _listed():
    from tools.skills_tool import skills_list

    return {entry["name"] for entry in json.loads(skills_list())["skills"]}


@pytest.mark.parametrize("settings,expected", [
    ({}, {"alpha", "beta", "gamma", "hermes-agent"}),
    ({"enabled": None}, {"alpha", "beta", "gamma", "hermes-agent"}),
    ({"enabled": []}, {"hermes-agent"}),
    ({"enabled": ["alpha"]}, {"alpha", "hermes-agent"}),
    ({"enabled": ["ALPHA", " beta ", "gam*"]}, {"hermes-agent"}),
    ({"enabled": ["alpha"], "disabled": ["alpha", "hermes-agent"]}, {"hermes-agent"}),
    ({"platform_enabled": {"telegram": ["beta"]}}, {"beta", "hermes-agent"}),
    ({"enabled": ["alpha", "beta"], "platform_enabled": {"telegram": ["beta", "gamma"]}}, {"beta", "hermes-agent"}),
    ({"enabled": ["alpha"], "platform_enabled": {"telegram": None}}, {"alpha", "hermes-agent"}),
    ({"platform_enabled": {"telegram": []}}, {"hermes-agent"}),
    ({"platform_enabled": {"discord": []}}, {"alpha", "beta", "gamma", "hermes-agent"}),
    ({"enabled": ["alpha"], "platform_disabled": {"telegram": ["alpha"]}}, {"hermes-agent"}),
])
def test_selection_applies_to_list_prompt_and_explicit_load(skill_env, monkeypatch, settings, expected):
    from agent.prompt_builder import build_skills_system_prompt
    from agent.skill_commands import build_preloaded_skills_prompt, get_skill_commands
    from tools.skills_tool import skill_view

    _, configure = skill_env
    configure(**settings)
    monkeypatch.setenv("HERMES_PLATFORM", "telegram")
    assert _listed() == expected
    prompt = build_skills_system_prompt()
    commands = get_skill_commands()
    for name in ("alpha", "beta", "gamma", "hermes-agent"):
        assert (f"- {name}:" in prompt) == (name in expected)
        assert (f"/{name}" in commands) == (name in expected)
        for file_path in (None, "references/guide.md"):
            assert json.loads(skill_view(name, file_path=file_path))["success"] == (name in expected)
    _, loaded, missing = build_preloaded_skills_prompt(["alpha", "beta", "gamma"])
    assert set(loaded) == expected - {"hermes-agent"}
    assert set(missing) == {"alpha", "beta", "gamma"} - expected


@pytest.mark.parametrize("settings", [
    {"enabled": "alpha"}, {"enabled": False}, {"enabled": {"alpha": True}},
    {"enabled": ["alpha", 7]}, {"platform_enabled": ["alpha"]},
    {"platform_enabled": {"telegram": "alpha"}},
])
def test_malformed_allowlists_fail_closed(skill_env, monkeypatch, caplog, settings):
    _, configure = skill_env
    configure(**settings)
    monkeypatch.setenv("HERMES_PLATFORM", "telegram")
    assert _listed() == {"hermes-agent"}
    assert "skills." in caplog.text


def test_allowlist_config_change_refreshes_warm_catalog_and_commands(skill_env):
    from agent.prompt_builder import build_skills_system_prompt
    from agent.skill_commands import get_skill_commands

    _, configure = skill_env
    configure(enabled=["alpha"])
    assert _listed() == {"alpha", "hermes-agent"}
    assert "/alpha" in get_skill_commands()
    assert "- alpha:" in build_skills_system_prompt()
    configure(enabled=["beta"])
    assert _listed() == {"beta", "hermes-agent"}
    assert "/alpha" not in get_skill_commands() and "/beta" in get_skill_commands()
    prompt = build_skills_system_prompt()
    assert "- alpha:" not in prompt and "- beta:" in prompt


def test_session_platform_and_profile_scopes_do_not_leak(skill_env, monkeypatch, tmp_path):
    from gateway.session_context import _SESSION_PLATFORM
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    _, configure = skill_env
    configure(platform_enabled={"telegram": ["alpha"], "discord": ["beta"]})
    token = _SESSION_PLATFORM.set("telegram")
    try:
        assert _listed() == {"alpha", "hermes-agent"}
        _SESSION_PLATFORM.set("discord")
        assert _listed() == {"beta", "hermes-agent"}
    finally:
        _SESSION_PLATFORM.reset(token)
    other = tmp_path / "other-profile"
    other.mkdir()
    atomic_config_replace(other / "config.yaml", {"skills": {"enabled": []}})
    token = set_hermes_home_override(str(other))
    try:
        assert _listed() == {"hermes-agent"}
    finally:
        reset_hermes_home_override(token)
    assert _listed() == {"alpha", "beta", "gamma", "hermes-agent"}


def test_duplicate_exact_load_names_and_aliases_share_policy(skill_env):
    from agent.prompt_builder import build_skills_system_prompt
    from tools.skills_tool import skill_view

    root, configure = skill_env
    _write_skill(root / "skills", "duplicate", directory="one/copy", body="FIRST COPY")
    _write_skill(root / "skills", "duplicate", directory="two/copy", body="SECOND COPY")
    _write_skill(root / "skills", "canonical", directory="different-directory")
    configure(enabled=["one/copy", "canonical"])
    assert _listed() == {"one/copy", "canonical", "hermes-agent"}
    prompt = build_skills_system_prompt()
    assert "one/copy" in prompt and "two/copy" not in prompt
    assert json.loads(skill_view("one/copy"))["success"] is True
    assert json.loads(skill_view("two/copy"))["success"] is False
    assert json.loads(skill_view("different-directory"))["success"] is True
    # An unadvertised directory alias is not a second way to enable a unique skill.
    configure(enabled=["different-directory"])
    assert json.loads(skill_view("canonical"))["success"] is False
    assert json.loads(skill_view("different-directory"))["success"] is False


def test_auto_load_and_bundles_cannot_restore_excluded_skills(skill_env):
    from agent.skill_bundles import build_bundle_invocation_message, save_bundle
    from agent.skill_commands import build_auto_load_prompt

    _, configure = skill_env
    configure(enabled=["alpha"], auto_load=["alpha", "beta"])
    prompt, loaded, missing = build_auto_load_prompt()
    assert loaded == ["alpha"] and missing == ["beta"]
    assert "alpha content" in prompt and "beta content" not in prompt
    save_bundle("selection-test", ["alpha", "beta"])
    prompt, loaded, missing = build_bundle_invocation_message("/selection-test")
    assert loaded == ["alpha"] and missing == ["beta"]
    assert "beta content" not in prompt


def test_bundle_explicit_platform_narrows_ambient_selection(skill_env):
    from agent.skill_bundles import build_bundle_invocation_message, save_bundle

    _, configure = skill_env
    configure(platform_enabled={"telegram": ["alpha"]})
    save_bundle("platform-selection", ["alpha", "beta"])
    prompt, loaded, _ = build_bundle_invocation_message("/platform-selection", platform="telegram")
    assert loaded == ["alpha"] and "beta content" not in prompt


def test_plugin_qualified_names_gate_all_surfaces_and_supporting_files(skill_env, monkeypatch):
    from agent.prompt_builder import build_skills_system_prompt
    from agent.skill_commands import get_plugin_skill_commands
    from hermes_cli import plugins
    from tools.registry import registry

    root, configure = skill_env
    manager = plugins.PluginManager()
    monkeypatch.setattr(plugins, "_plugin_manager", manager)
    monkeypatch.setattr(plugins, "discover_plugins", lambda: None)
    for name in ("alpha", "beta"):
        directory = _write_skill(root / "plugin-skills", name)
        manager._plugin_skills[f"demo:{name}"] = {
            "path": directory / "SKILL.md", "plugin": "demo", "bare_name": name,
            "description": f"Plugin {name}.",
        }
    configure(enabled=["demo:alpha"])
    assert _listed() == {"demo:alpha", "hermes-agent"}
    commands = get_plugin_skill_commands()
    assert "/demo:alpha" in commands and "/demo:beta" not in commands
    prompt = build_skills_system_prompt()
    assert "demo:alpha" in prompt and "demo:beta" not in prompt
    handler = registry.get_entry("skill_view").handler
    assert json.loads(handler({"name": "demo:alpha"}))["success"] is True
    assert json.loads(handler({"name": "demo:beta"}))["success"] is False
    assert json.loads(handler({"name": "demo:beta", "file_path": "references/guide.md"}))["success"] is False
    configure(enabled=["alpha"])
    assert json.loads(handler({"name": "demo:alpha"}))["success"] is False
    # The projection's metadata cache is warm: a config-only allowlist change must still apply.
    assert "/demo:alpha" not in get_plugin_skill_commands()


def test_external_skills_follow_selection(skill_env, tmp_path):
    from agent.prompt_builder import build_skills_system_prompt
    from tools.skills_tool import skill_view

    _, configure = skill_env
    external = tmp_path / "external"
    _write_skill(external, "external-alpha")
    _write_skill(external, "external-beta")
    configure(external_dirs=[str(external)], enabled=["external-alpha"])
    assert _listed() == {"external-alpha", "hermes-agent"}
    assert "external-alpha" in build_skills_system_prompt()
    assert json.loads(skill_view("external-beta"))["success"] is False


def test_explicit_home_override_uses_own_allowlist_for_index_and_auto_load(skill_env, tmp_path):
    from agent.prompt_builder import build_skills_system_prompt
    from agent.skill_commands import build_auto_load_prompt

    _, configure = skill_env
    configure(enabled=["alpha"])
    other = tmp_path / "scoped-profile"
    _write_skill(other / "skills", "scoped-alpha")
    _write_skill(other / "skills", "scoped-beta")
    atomic_config_replace(other / "config.yaml", {
        "skills": {"enabled": ["scoped-alpha"], "auto_load": ["scoped-alpha", "scoped-beta"]},
    })
    prompt = build_skills_system_prompt(skills_dir_override=other / "skills")
    assert "scoped-alpha" in prompt and "scoped-beta" not in prompt
    # skill_view's production root must follow the bound home, not the test fixture's patched root.
    from tools import skills_tool
    original = skills_tool.SKILLS_DIR
    skills_tool.SKILLS_DIR = skills_tool._SKILLS_DIR_AT_IMPORT
    try:
        prompt, loaded, missing = build_auto_load_prompt(home_override=other)
    finally:
        skills_tool.SKILLS_DIR = original
    assert loaded == ["scoped-alpha"] and missing == ["scoped-beta"]
    assert "scoped-beta content" not in prompt
