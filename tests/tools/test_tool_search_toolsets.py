"""Disclosure policies expand toolsets, including newly registered tools and composites."""

import json

import pytest

from hermes_cli.config import atomic_config_write
from tools.registry import registry
from tools.tool_search import BRIDGE_TOOL_NAMES, ToolSearchConfig, assemble_tool_defs


def _names(result):
    return {td["function"]["name"] for td in result.tool_defs}


@pytest.mark.parametrize(
    "selection,visible",
    [
        ({"eager_toolsets": ["file"]}, {"read_file", "write_file", "patch", "search_files"}),
        ({"defer_toolsets": ["file"]}, {"terminal"}),
        ({"eager_toolsets": ["file"], "defer_toolsets": ["file"]}, {"read_file", "write_file", "patch", "search_files"}),
        ({"eager_toolsets": []}, set()),
    ],
)
def test_whole_file_toolset_changes_disclosure(selection, visible):
    import model_tools

    defs = model_tools.get_tool_definitions(
        enabled_toolsets=["file", "terminal"], quiet_mode=True, skip_tool_search_assembly=True,
    )
    result = assemble_tool_defs(defs, config=ToolSearchConfig.from_raw(selection))
    assert result.activated
    assert _names(result) == visible | BRIDGE_TOOL_NAMES


@pytest.mark.parametrize("policy_key", ["eager_toolsets", "defer_toolsets"])
def test_late_members_of_plugin_toolset_follow_the_same_policy(policy_key):
    import model_tools

    config = ToolSearchConfig.from_raw({policy_key: ["disclosure-plugin"]})
    registered = []
    try:
        for name in ["disclosure_read", "disclosure_write"]:
            registry.register(
                name=name, toolset="disclosure-plugin",
                schema={"name": name, "description": "Read or write a record.", "parameters": {"type": "object", "properties": {}}},
                handler=lambda args, **kwargs: json.dumps({"ok": True}),
            )
            registered.append(name)
            defs = model_tools.get_tool_definitions(
                enabled_toolsets=["file", "disclosure-plugin"], quiet_mode=True, skip_tool_search_assembly=True,
            )
            result = assemble_tool_defs(defs, config=config)
            visible = _names(result)
            for member in registered:
                assert (member in visible) == (policy_key == "eager_toolsets")
    finally:
        for name in registered:
            registry.deregister(name)


def test_disclosure_toolsets_never_enable_globally_disabled_toolsets(tmp_path, monkeypatch):
    import model_tools

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    atomic_config_write(tmp_path / "config.yaml", {
        "agent": {"enabled_toolsets": ["file"], "disabled_toolsets": ["file"]},
        "tools": {"tool_search": {"eager_toolsets": ["file", "terminal"]}},
    })
    assert model_tools.get_tool_definitions(enabled_toolsets=["file", "terminal"], quiet_mode=True) == []


@pytest.mark.parametrize("key", ["eager_toolsets", "defer_toolsets"])
def test_composite_includes_apply_to_every_member(monkeypatch, key):
    import model_tools
    import toolsets

    monkeypatch.setitem(toolsets.TOOLSETS, "disclosure-bundle", {
        "description": "File tools via an included toolset", "tools": [], "includes": ["file"],
    })
    defs = model_tools.get_tool_definitions(
        enabled_toolsets=["file", "terminal"], quiet_mode=True, skip_tool_search_assembly=True,
    )
    result = assemble_tool_defs(defs, config=ToolSearchConfig.from_raw({key: ["disclosure-bundle"]}))
    expected = {"read_file", "write_file", "patch", "search_files"} if key == "eager_toolsets" else {"terminal"}
    assert _names(result) == expected | BRIDGE_TOOL_NAMES


@pytest.mark.parametrize("bad", ["file", [1], {"file"}, ("file",)])
def test_invalid_defer_toolset_policy_warns_and_preserves_legacy_behavior(bad, caplog):
    import model_tools

    defs = model_tools.get_tool_definitions(
        enabled_toolsets=["file", "terminal"], quiet_mode=True, skip_tool_search_assembly=True,
    )
    result = assemble_tool_defs(defs, config=ToolSearchConfig.from_raw({"defer_toolsets": bad}))
    baseline = assemble_tool_defs(defs, config=ToolSearchConfig.from_raw(None))
    assert _names(result) == _names(baseline)
    assert "tools.tool_search.defer_toolsets" in caplog.text
