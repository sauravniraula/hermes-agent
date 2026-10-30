"""An explicit eager list defers every other enabled tool, without widening scope."""

import json
from types import SimpleNamespace

import pytest

from tools.registry import discover_builtin_tools, registry
from tools.tool_search import (
    BRIDGE_TOOL_NAMES,
    ToolSearchConfig,
    assemble_tool_defs,
    dispatch_tool_describe,
    dispatch_tool_search,
    load_config_readonly,
    resolve_underlying_call,
    scoped_deferrable_names,
)


@pytest.fixture
def eager_tools():
    discover_builtin_tools()
    names = ("mcp__eager_test__fetch", "eager_test_plugin", "eager_test_gui")
    for name, toolset in zip(names, ("mcp-eager-test", "eager-test-plugin", "desktop_ui")):
        registry.register(
            name=name,
            toolset=toolset,
            schema={
                "name": name,
                "description": "Fetch an eager policy test record.",
                "parameters": {"type": "object", "properties": {}},
            },
            handler=lambda args, **kwargs: json.dumps({"ok": True}),
        )
    yield names
    for name in names:
        registry.deregister(name)


def _definitions(names):
    return [{"type": "function", "function": {**registry.get_entry(name).schema, "name": name}}
            for name in names]


@pytest.mark.parametrize(
    "raw,visible",
    [
        ({}, {"terminal", "read_file", "eager_test_gui"}),
        ({"eager": None}, {"terminal", "read_file", "eager_test_gui"}),
        ({"eager": "terminal"}, {"terminal", "read_file", "eager_test_gui"}),
        ({"eager": 42}, {"terminal", "read_file", "eager_test_gui"}),
        ({"eager": {"terminal"}}, {"terminal", "read_file", "eager_test_gui"}),
        ({"eager": ("terminal",)}, {"terminal", "read_file", "eager_test_gui"}),
        ({"eager": []}, set()),
        ({"eager": [" terminal "]}, set()),
        ({"eager": ["Terminal"]}, set()),
        ({"eager": ["mcp__eager_test__*"]}, set()),
        ({"eager": [" terminal ", "mcp__eager_test__fetch", "", "terminal"]},
         {"terminal", "mcp__eager_test__fetch"}),
        ({"eager": ["computer_use"], "defer": ["computer_use"]}, {"computer_use"}),
        ({"eager": ["disabled_tool"]}, set()),
        ({"eager": ["terminal", "read_file", "computer_use", "mcp__eager_test__fetch",
                    "eager_test_plugin", "eager_test_gui"]},
         {"terminal", "read_file", "computer_use", "mcp__eager_test__fetch",
          "eager_test_plugin", "eager_test_gui"}),
    ],
)
def test_eager_selection_and_legacy_fallback(eager_tools, raw, visible, caplog):
    names = {"terminal", "read_file", "computer_use", *eager_tools}
    config = ToolSearchConfig.from_raw(raw)
    result = assemble_tool_defs(_definitions(sorted(names)), config=config)

    exposed = {td["function"]["name"] for td in result.tool_defs}
    assert result.activated == bool(names - visible)
    assert exposed == visible | (BRIDGE_TOOL_NAMES if names - visible else set())
    assert result.deferred_count == len(names - visible)
    if raw.get("eager") is not None and not isinstance(raw["eager"], list):
        assert "tools.tool_search.eager" in caplog.text


@pytest.mark.parametrize("eager", [["terminal", "mcp__eager_test__fetch", "eager_test_gui"], []])
def test_eager_policy_flows_through_live_config_and_bridge(tmp_path, monkeypatch, eager_tools, eager):
    """The real YAML loaders, search/describe and dispatcher must agree on the policy."""
    import model_tools
    from agent.tool_executor import _unwrap_tool_search_call
    from hermes_cli.config import atomic_config_write

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    atomic_config_write(tmp_path / "config.yaml", {
        "tools": {"tool_search": {"enabled": "auto", "eager": eager}},
    })
    config = load_config_readonly()
    # Keep the GUI fixture out of this scope, even if it is named in the eager list.
    enabled = ["terminal", "file", "mcp-eager-test", "eager-test-plugin"]
    defs = model_tools.get_tool_definitions(
        enabled_toolsets=enabled, quiet_mode=True, skip_tool_search_assembly=True,
    )
    names = {td["function"]["name"] for td in defs}
    visible = names & set(eager)
    deferred = names - visible
    result = assemble_tool_defs(defs, config=config)
    assert {td["function"]["name"] for td in result.tool_defs} == visible | BRIDGE_TOOL_NAMES
    assert scoped_deferrable_names(defs) == deferred

    search = json.loads(dispatch_tool_search({"queries": ["read file"]}, current_tool_defs=defs))
    assert search["total_available"] == len(deferred)
    assert set(search["tools"]) <= deferred
    assert "read_file" in search["tools"]

    described = json.loads(dispatch_tool_describe(
        {"names": ["read_file", "mcp__eager_test__fetch", "eager_test_gui"]},
        current_tool_defs=defs,
    ))
    assert "read_file" in described["tools"]
    assert "eager_test_gui" not in described["tools"]
    if eager:
        assert "mcp__eager_test__fetch" in described["errors"]
    else:
        assert "mcp__eager_test__fetch" in described["tools"]

    for name in visible:
        _, _, error = resolve_underlying_call({"name": name, "arguments": {}}, frozenset(names))
        assert "directly" in error
    blocked = model_tools.handle_function_call(
        function_name="tool_call",
        function_args={"name": "eager_test_gui", "arguments": {}},
        enabled_toolsets=enabled,
    )
    assert json.loads(blocked)["error"]

    target = tmp_path / "record.txt"
    target.write_text("eager allowlist bridge execution proof", encoding="utf-8")
    agent = SimpleNamespace(enabled_toolsets=enabled, disabled_toolsets=[])
    arguments = {"path": str(target)}
    assert _unwrap_tool_search_call(
        agent, "tool_call", {"name": "read_file", "arguments": arguments},
    ) == ("read_file", arguments, None)
    _, _, invalid = _unwrap_tool_search_call(
        agent, "tool_call", {"name": "read_file", "arguments": {}},
    )
    assert invalid
    name, _, _ = _unwrap_tool_search_call(
        agent, "tool_call", {"name": "eager_test_gui", "arguments": {}},
    )
    assert name == "tool_call"
    output = model_tools.handle_function_call(
        function_name="tool_call",
        function_args={"name": "read_file", "arguments": {"path": str(target)}},
        enabled_toolsets=enabled,
    )
    assert "eager allowlist bridge execution proof" in output


def test_disabled_search_ignores_eager_allowlist(eager_tools):
    defs = _definitions(["terminal", "read_file", *eager_tools])
    result = assemble_tool_defs(defs, config=ToolSearchConfig.from_raw({"enabled": "off", "eager": []}))
    assert not result.activated
    assert result.tool_defs == defs
