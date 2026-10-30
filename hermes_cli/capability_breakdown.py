"""Profile/platform capability listings with explicitly approximate token costs."""

from contextlib import redirect_stdout
import json
import logging
from pathlib import Path
import sys
from typing import Any

from hermes_cli.inspection import platform_scope


def _tokens(value: Any) -> int:
    from agent.model_metadata import estimate_tokens_rough

    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return estimate_tokens_rough(text)


def compute_tools_breakdown(platform: str = "cli") -> dict:
    from hermes_cli.config import load_config
    from hermes_cli.mcp_startup import ensure_mcp_discovery_before_agent_build
    from hermes_cli.tools_config import _get_platform_tools
    from model_tools import get_tool_definitions
    from hermes_cli.prompt_size import _build_inspection_agent

    with platform_scope(platform):
        cfg = load_config()
        selected = sorted(_get_platform_tools(cfg, platform))
        ensure_mcp_discovery_before_agent_build(logger=logging.getLogger(__name__), single_query=True)
        agent = _build_inspection_agent(platform)
        try:
            # The agent also owns post-registry memory/context-engine injections.
            direct = list(agent.tools or [])
            raw = get_tool_definitions(enabled_toolsets=selected, quiet_mode=True, skip_tool_search_assembly=True)
        finally:
            agent.close()
        rows = _tool_rows(raw, direct)
        direct_count = sum(row["disclosure"] == "direct" for row in rows)
        return {"platform": platform, "token_method": "estimate_tokens_rough", "estimated": True,
                "counts": {"active": len(rows), "direct": direct_count, "deferred": len(rows) - direct_count},
                "direct_schema_tokens": _tokens(direct), "tools": rows}


def _tool_rows(raw: list, direct_definitions: list) -> list[dict]:
    from tools.registry import registry
    from tools.tool_search_catalog import BRIDGE_TOOL_NAMES

    direct = {td["function"]["name"]: td for td in direct_definitions}
    active = {td["function"]["name"]: td for td in raw}
    active.update(direct)
    rows = []
    for name, schema in sorted(active.items()):
        entry = registry.get_entry(name)
        toolset = entry.toolset if entry else ("tool_search" if name in BRIDGE_TOOL_NAMES else "agent-injected")
        cost = _tokens(schema)
        rows.append({"name": name, "toolset": toolset,
                     "disclosure": "direct" if name in direct else "deferred", "tokens": cost,
                     "upfront_tokens": cost if name in direct else 0})
    return rows


def _render_tools(data: dict) -> str:
    counts = data["counts"]
    lines = [f"Tools breakdown (platform={data['platform']}; estimated tokens, not billed usage)",
             f"Active: {counts['active']} | Direct: {counts['direct']} | Deferred: {counts['deferred']}",
             f"Direct schema payload: ~{data['direct_schema_tokens']:,} tokens",
             "Deferred full-schema sizes are on-demand costs; their catalog is included in the direct bridge.",
             "", f"{'Tool':<40} {'Disclosure':<10} {'Schema tokens':>14} {'Upfront':>9}"]
    lines.extend(f"{r['name']:<40} {r['disclosure']:<10} {r['tokens']:>14,} {r['upfront_tokens']:>9,}"
                 for r in data["tools"])
    return "\n".join(lines)


def cmd_tools_breakdown(args: Any) -> int:
    from agent.redact import redact_sensitive_text

    try:
        with redirect_stdout(sys.stderr):
            data = compute_tools_breakdown(args.platform)
        print(json.dumps(data, ensure_ascii=False, indent=2) if args.json else _render_tools(data))
        return 0
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        print(redact_sensitive_text(f"Could not inspect tools: {exc}", force=True,
                                    redact_url_credentials=True), file=sys.stderr)
        return 1


def _skill_file_tokens(path: Path | None) -> int | None:
    if path is None:
        return None
    try:
        return _tokens(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError):
        return None


def _skill_index_tokens(index: str) -> dict[str, int]:
    """Per-entry costs exclude shared category/header text; total index includes it."""
    costs = {}
    for line in index.splitlines(keepends=True):
        stripped = line.lstrip()
        if stripped.startswith("- "):
            name = stripped[2:].partition(": ")[0].strip()
            costs[name] = _tokens(line)
        elif " [names only]: " in line:
            prefix, _, listing = line.partition(" [names only]: ")
            names = [name.strip() for name in listing.split(",") if name.strip()]
            shared = _tokens(prefix + " [names only]: ")
            for offset, name in enumerate(names):
                costs[name] = _tokens(name) + shared // len(names) + (offset < shared % len(names))
    return costs


def compute_skills_breakdown(platform: str = "cli") -> dict:
    from agent.coding_context import coding_compact_skill_categories
    from agent.prompt_builder import build_skills_system_prompt
    from agent.runtime_cwd import resolve_context_cwd
    from hermes_cli.config import load_config
    from hermes_cli.plugins import get_plugin_manager
    from hermes_cli.tools_config import _get_platform_tools
    from tools.skills_tool import _skill_catalog, skills_list
    from toolsets import resolve_toolset

    with platform_scope(platform):
        listing = json.loads(skills_list())
        if not listing.get("success"):
            raise ValueError(listing.get("error", "Skill discovery failed"))
        selected = _get_platform_tools(load_config(), platform)
        names = {name for toolset in selected for name in resolve_toolset(toolset)}
        tools_enabled = bool(names & {"skills_list", "skill_view", "skill_manage"})
        index = build_skills_system_prompt(
            available_tools=names, available_toolsets=selected,
            compact_categories=coding_compact_skill_categories(platform=platform, cwd=resolve_context_cwd()) or None,
        ) if tools_enabled else ""
        index_costs = _skill_index_tokens(index)
        paths = {s["load_name"]: s["path"] for s in _skill_catalog() if s["load_name"]}
        manager = get_plugin_manager()
        rows = []
        for skill in listing["skills"]:
            name = skill["name"]
            path = paths.get(name) or manager.find_plugin_skill(name)
            rows.append({"name": name, "category": skill.get("category", ""),
                         "index_tokens": index_costs.get(name, 0), "file_tokens": _skill_file_tokens(path),
                         "path": str(path) if path else None})
        rows.sort(key=lambda row: row["name"])
        return {"platform": platform, "token_method": "estimate_tokens_rough", "estimated": True,
                "skills_toolset_enabled": tools_enabled, "count": len(rows),
                "index_total_tokens": _tokens(index), "skills": rows}


def _render_skills(data: dict) -> str:
    lines = [f"Skills breakdown (platform={data['platform']}; estimated tokens, not billed usage)",
             f"Enabled: {data['count']} | Complete index: ~{data['index_total_tokens']:,} tokens",
             "Index = advertised entry; file = raw SKILL.md if read, not a sum paid upfront.",
             "Shared index headings are included only in the complete-index total.",
             "", f"{'Skill':<45} {'Index tokens':>13} {'File tokens':>13}"]
    for row in data["skills"]:
        file_cost = f"{row['file_tokens']:,}" if row["file_tokens"] is not None else "n/a"
        lines.append(f"{row['name']:<45} {row['index_tokens']:>13,} {file_cost:>13}")
    return "\n".join(lines)


def cmd_skills_breakdown(args: Any) -> int:
    from agent.redact import redact_sensitive_text

    try:
        with redirect_stdout(sys.stderr):
            data = compute_skills_breakdown(args.platform)
        print(json.dumps(data, ensure_ascii=False, indent=2) if args.json else _render_skills(data))
        return 0
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        print(redact_sensitive_text(f"Could not inspect skills: {exc}", force=True,
                                    redact_url_credentials=True), file=sys.stderr)
        return 1
