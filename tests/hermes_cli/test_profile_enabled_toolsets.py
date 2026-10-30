"""A profile-wide toolset allowlist overrides platform defaults without widening session grants."""

import json
from types import SimpleNamespace

import pytest

from hermes_cli.config import atomic_config_write, load_config_readonly
from hermes_cli.tools_config import _get_platform_tools, _platform_toolsets_explicitly_saved
from tools.registry import registry


@pytest.mark.parametrize("platform", ["cli", "tui", "desktop", "telegram", "discord", "slack", "cron"])
@pytest.mark.parametrize("allowed", [[], ["terminal", "file"]])
def test_profile_selection_overrides_platform_defaults(platform, allowed):
    config = {
        "agent": {"enabled_toolsets": allowed},
        "platform_toolsets": {"cli": ["web"], "telegram": []},
        "mcp_servers": {"unlisted": {"command": "unused"}},
        "context": {"engine": "lossless"},
        "toolsets": ["kanban"],
    }
    assert _get_platform_tools(config, platform) == set(allowed)
    assert _platform_toolsets_explicitly_saved(config, platform)


def test_profile_selection_retains_platform_restrictions_and_disabled_precedence():
    config = {"agent": {
        "enabled_toolsets": ["discord", "file", "terminal"],
        "disabled_toolsets": ["terminal"],
    }}
    assert _get_platform_tools(config, "telegram") == {"file"}
    assert _get_platform_tools(config, "discord") == {"discord", "file"}


@pytest.mark.parametrize("raw", ["terminal", 1, {"terminal": True}, [1]])
def test_malformed_allowlist_is_loud_and_selects_no_tools(raw, caplog):
    assert _get_platform_tools({"agent": {"enabled_toolsets": raw}}, "cli") == set()
    assert "agent.enabled_toolsets" in caplog.text


def test_null_allowlist_preserves_platform_selection():
    config = {"platform_toolsets": {"cli": ["terminal", "file", "no_mcp"]}}
    assert _get_platform_tools({**config, "agent": {"enabled_toolsets": None}}, "cli") == _get_platform_tools(config, "cli")


@pytest.fixture
def policy_config(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    def save(allowed, disabled=None, **extra):
        atomic_config_write(tmp_path / "config.yaml", {
            "agent": {"enabled_toolsets": allowed, "disabled_toolsets": disabled or []},
            "tools": {"tool_search": {"enabled": "off"}},
            **extra,
        })
        return load_config_readonly()

    return save


@pytest.mark.parametrize("selection", [None, ["hermes-cli"], ["coding"], ["file", "terminal"]])
def test_real_yaml_loader_caps_composites_and_caller_selections(policy_config, selection):
    import model_tools

    config = policy_config(["file", "terminal"], disabled=["terminal"])
    assert _get_platform_tools(config, "cli") == {"file"}
    defs = model_tools.get_tool_definitions(enabled_toolsets=selection, quiet_mode=True)
    assert {td["function"]["name"] for td in defs} == {"read_file", "write_file", "patch", "search_files"}
    # A narrower caller selection must not be widened to the whole profile list.
    assert model_tools.get_tool_definitions(enabled_toolsets=["terminal"], quiet_mode=True) == []


def test_empty_profile_allowlist_blocks_even_explicit_caller_toolsets(policy_config):
    import model_tools

    policy_config([])
    assert model_tools.get_tool_definitions(enabled_toolsets=["hermes-cli"], quiet_mode=True) == []
    assert model_tools.get_tool_definitions(quiet_mode=True) == []


def test_profile_allowlist_is_scoped_to_the_current_home(tmp_path, monkeypatch):
    import model_tools

    for name, selected in [("a", ["file"]), ("b", [])]:
        home = tmp_path / name
        home.mkdir()
        atomic_config_write(home / "config.yaml", {"agent": {"enabled_toolsets": selected}})
    for name, expected in [("a", {"read_file", "write_file", "patch", "search_files"}), ("b", set()), ("a", {"read_file", "write_file", "patch", "search_files"})]:
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / name))
        defs = model_tools.get_tool_definitions(enabled_toolsets=["hermes-cli"], quiet_mode=True)
        assert {td["function"]["name"] for td in defs} == expected


def test_late_plugin_and_mcp_registration_cannot_escape_allowlist(policy_config):
    import model_tools

    policy_config(["file"], tools={"tool_search": {"enabled": "auto", "eager_toolsets": []}})
    for name, toolset in [("profile_policy_plugin", "profile-policy-plugin"), ("mcp__profile_policy__fetch", "mcp-profile-policy")]:
        registry.register(
            name=name, toolset=toolset,
            schema={"name": name, "description": "Fetch a policy record.", "parameters": {"type": "object", "properties": {}}},
            handler=lambda args, **kwargs: json.dumps({"ok": True}),
        )
    try:
        defs = model_tools.get_tool_definitions(
            enabled_toolsets=["file", "profile-policy-plugin", "mcp-profile-policy"],
            quiet_mode=True, skip_tool_search_assembly=True,
        )
        assert {td["function"]["name"] for td in defs} == {"read_file", "write_file", "patch", "search_files"}
        for name in ["profile_policy_plugin", "mcp__profile_policy__fetch"]:
            blocked = json.loads(model_tools.handle_function_call(
                "tool_call", {"name": name, "arguments": {}},
                enabled_toolsets=["file", "profile-policy-plugin", "mcp-profile-policy"],
            ))
            assert blocked["error"]
    finally:
        registry.deregister("profile_policy_plugin")
        registry.deregister("mcp__profile_policy__fetch")


def test_memory_provider_gate_honors_profile_allowlist(policy_config):
    from agent.memory_manager_tools import memory_provider_tools_enabled

    policy_config(["file"])
    assert not memory_provider_tools_enabled(None)
    assert not memory_provider_tools_enabled(["memory"], memory_tool_present=True)


@pytest.mark.parametrize("server_enabled", [False, True])
def test_explicit_plugin_and_mcp_opt_in_respects_availability(policy_config, server_enabled):
    import model_tools

    rows = [("policy_plugin_call", "policy-plugin", None),
            ("mcp__policy_notes__fetch", "mcp-policy-notes", None),
            ("policy_unavailable_call", "policy-plugin", lambda: False)]
    for name, toolset, check in rows:
        registry.register(
            name=name, toolset=toolset, check_fn=check,
            schema={"name": name, "description": "Read a policy record.", "parameters": {"type": "object", "properties": {}}},
            handler=lambda args, **kwargs: json.dumps({"ok": True}),
        )
    registry.register_toolset_alias("policy-notes", "mcp-policy-notes")
    try:
        policy_config(["policy-plugin", "policy-notes"], mcp_servers={
            "policy-notes": {"command": "unused", "enabled": server_enabled},
        })
        defs = model_tools.get_tool_definitions(quiet_mode=True)
        expected = {"policy_plugin_call", "mcp__policy_notes__fetch"} if server_enabled else {"policy_plugin_call"}
        assert {td["function"]["name"] for td in defs} == expected
    finally:
        for name, _, _ in rows:
            registry.deregister(name)


@pytest.mark.parametrize("name", ["policy-notes", "mcp-policy-notes"])
@pytest.mark.parametrize("server_enabled", [False, True])
def test_only_explicit_enabled_mcp_server_is_selected(name, server_enabled):
    config = {"agent": {"enabled_toolsets": ["file", name]}, "mcp_servers": {
        "policy-notes": {"command": "unused", "enabled": server_enabled},
        "unlisted": {"command": "unused"},
    }}
    assert _get_platform_tools(config, "telegram") == ({"file", "policy-notes"} if server_enabled else {"file"})


@pytest.mark.parametrize("allowed", [["file"], ["context_engine"]])
def test_context_engine_injection_and_refresh_honor_allowlist(policy_config, allowed):
    from agent.agent_init import _inject_context_engine_tools
    from tools.mcp_tool_agent import _reinject_post_build_tools

    policy_config(allowed)
    schema = {"name": "policy_context_lookup", "description": "Look up context.", "parameters": {"type": "object", "properties": {}}}
    agent = SimpleNamespace(
        tools=[], valid_tool_names=set(), enabled_toolsets=["context_engine"], disabled_toolsets=[],
        session_id="policy-context", platform="cli",
        context_compressor=SimpleNamespace(get_tool_schemas=lambda: [schema], on_session_start=lambda *args, **kwargs: None),
    )
    _inject_context_engine_tools(agent)
    expected = {"policy_context_lookup"} if "context_engine" in allowed else set()
    assert agent.valid_tool_names == expected
    defs, names = [], set()
    assert _reinject_post_build_tools(agent, defs, names) == expected
    assert names == expected
