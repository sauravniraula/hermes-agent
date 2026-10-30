"""Profile-wide toolset selection and the runtime ceiling shared by every surface."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def profile_enabled_toolsets(config: dict) -> list[str] | None:
    """None preserves platform defaults; [] explicitly selects no toolsets.

    A malformed restriction fails closed rather than widening to defaults. Names
    are literal; do not repair typos, whitespace, or non-string identifiers.
    """
    agent = config.get("agent") or {}
    raw = agent.get("enabled_toolsets") if isinstance(agent, dict) else None
    if raw is None:
        return None
    if not isinstance(raw, list) or any(not isinstance(name, str) or not name for name in raw):
        logger.warning("agent.enabled_toolsets must be a YAML list of toolset names or null; selecting no toolsets")
        return []
    return list(dict.fromkeys(raw))


def _profile_tool_names(config: dict, selected: list[str]) -> set[str]:
    from agent.skill_utils import parse_config_string_list
    from model_tools import _apply_toolset_selection
    from tools.mcp_tool_common import mcp_server_enabled
    from tools.registry import registry

    tools: set[str] = set()
    _apply_toolset_selection(tools, selected, quiet_mode=True, disable=False)
    _apply_toolset_selection(
        tools, parse_config_string_list((config.get("agent") or {}).get("disabled_toolsets")),
        quiet_mode=True, disable=True,
    )
    # A still-registered MCP schema must not bypass its server's explicit off switch.
    servers = config.get("mcp_servers") or {}
    disabled_mcp = {f"mcp-{name}" for name, cfg in servers.items()
                    if isinstance(cfg, dict) and not mcp_server_enabled(cfg)}
    return {name for name in tools if (entry := registry.get_entry(name)) is None or entry.toolset not in disabled_mcp}


def restrict_profile_tool_names(tools: set[str]) -> set[str]:
    """Intersect, never widen, a caller's grant with the active profile's allowlist."""
    from hermes_cli.config import load_config_readonly

    config = load_config_readonly()
    selected = profile_enabled_toolsets(config)
    return tools if selected is None else tools & _profile_tool_names(config, selected)


def profile_allows_toolset(name: str) -> bool:
    """Gate post-registry tool families (memory providers and context engines) too."""
    from hermes_cli.config import load_config_readonly
    from toolsets import resolve_toolset

    config = load_config_readonly()
    selected = profile_enabled_toolsets(config)
    if selected is None:
        return True
    tools = set(resolve_toolset(name))
    if tools:
        return bool(tools & _profile_tool_names(config, selected))
    return name in selected and name not in ((config.get("agent") or {}).get("disabled_toolsets") or [])


def resolve_profile_toolsets(config: dict, platform: str, selected: list[str]) -> set[str]:
    """Reuse platform expansion, then remove implicit plugin/MCP/native additions.

    Resolve with the allowlist unset in a copy to avoid recursion and preserve the
    caller's document. Tool membership, not just labels, handles composite bundles.
    """
    from hermes_cli import tools_config as tc
    from toolsets import resolve_toolset

    if not selected:
        return set()
    requested = [name for name in selected if tc._toolset_allowed_for_platform(name, platform)]
    scoped = {**config, "agent": {**(config.get("agent") or {}), "enabled_toolsets": None},
              "platform_toolsets": {platform: requested}}
    resolved = tc._get_platform_tools(scoped, platform, include_default_mcp_servers=False)
    allowed = _profile_tool_names(config, requested)
    # A configured MCP server's canonical mcp-<name> and bare alias select the same server,
    # including before discovery has registered any schemas.
    known_servers = tc.enabled_mcp_server_names(config) | set(config.get("mcp_servers") or {})
    explicit = set(requested) | {name[4:] for name in requested
                                 if name.startswith("mcp-") and name not in known_servers and name[4:] in known_servers}
    return {name for name in resolved if name in explicit
            or ((tools := set(resolve_toolset(name))) and tools <= allowed)}
