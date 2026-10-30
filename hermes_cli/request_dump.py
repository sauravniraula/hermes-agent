"""Build a fresh provider-native request for inspection, never invoking inference."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Any
from hermes_cli.inspection import platform_scope


def _build_agent(platform: str, model: str | None, provider: str | None):
    from agent.skill_utils import parse_config_string_list
    from hermes_cli.config import load_config
    from hermes_cli.mcp_startup import ensure_mcp_discovery_before_agent_build
    from hermes_cli.oneshot import _resolve_model_and_provider
    from hermes_cli.runtime_provider import resolve_runtime_provider
    from hermes_cli.tools_config import _get_platform_tools
    from hermes_constants import resolve_reasoning_config
    from run_agent import AIAgent
    import logging

    cfg = load_config()
    choice = _resolve_model_and_provider(cfg, model, provider)
    if not choice.model:
        raise ValueError("No model configured; run `hermes model` or pass --model")
    runtime = resolve_runtime_provider(
        requested=choice.provider, target_model=choice.model,
        explicit_base_url=choice.base_url, explicit_api_key=choice.api_key,
    )
    ensure_mcp_discovery_before_agent_build(logger=logging.getLogger(__name__), single_query=True)
    return AIAgent(
        model=choice.model, provider=runtime.get("provider"),
        requested_provider=runtime.get("requested_provider"), api_mode=runtime.get("api_mode"),
        api_key=runtime.get("api_key"), base_url=runtime.get("base_url"),
        credential_pool=runtime.get("credential_pool"), request_overrides=runtime.get("request_overrides"),
        reasoning_config=resolve_reasoning_config(cfg, choice.model),
        enabled_toolsets=sorted(_get_platform_tools(cfg, platform)),
        disabled_toolsets=parse_config_string_list((cfg.get("agent") or {}).get("disabled_toolsets")) or None,
        platform=platform, quiet_mode=True, save_trajectories=False,
    )


def _request_body(agent: Any, prompt: str) -> dict[str, Any]:
    messages = [{"role": "system", "content": agent._build_system_prompt()},
                {"role": "user", "content": prompt}]
    # Same provider-native builder used by the turn loop; no conversation, hooks, tools or API call.
    kwargs = agent._build_api_kwargs(messages)
    if agent.api_mode == "codex_responses":
        from agent.codex_responses_adapter import drops_replay_item_ids

        kwargs = agent._get_transport().preflight_kwargs(
            kwargs, allow_stream=False, drop_replay_item_ids=drops_replay_item_ids(agent.base_url),
            sanitize_harmony_tokens=agent._is_codex_backend(),
        )
    body = {key: value for key, value in kwargs.items()
            if key not in {"timeout", "http_client", "extra_headers", "extra_query", "extra_body"}
            and not key.startswith("_")}
    body.update(kwargs.get("extra_body") or {})  # SDK extra_body fields override standard fields on the wire.
    if agent.api_mode == "codex_responses" and agent._is_codex_backend():
        body["stream"] = True
    return body


def _redact_body(value: Any) -> Any:
    from agent.redact import is_secret_field_name, redact_sensitive_text

    if isinstance(value, str):
        return redact_sensitive_text(value, force=True, redact_url_credentials=True)
    if isinstance(value, list):
        return [_redact_body(item) for item in value]
    if isinstance(value, dict):
        return {key: "[REDACTED]" if is_secret_field_name(key) and isinstance(item, str)
                else _redact_body(item) for key, item in value.items()}
    return value


def build_request_dump(platform: str = "cli", prompt: str = "", *,
                       model: str | None = None, provider: str | None = None) -> dict[str, Any]:
    """Fresh profile/CWD request, not a capture of any running session or its history."""
    with platform_scope(platform):
        agent = _build_agent(platform, model, provider)
        try:
            body = _request_body(agent, prompt)
            # Scrub string values before serialization: text regexes can consume JSON delimiters.
            return _redact_body(json.loads(json.dumps(body, ensure_ascii=False)))
        finally:
            agent.close()


def run_request_dump(args: Any) -> int:
    """Write only to a new, private file; failures never print unredacted exception text."""
    from agent.redact import redact_sensitive_text

    output = Path(args.output).expanduser()
    try:
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"Output already exists: {output}; choose a new file")
        body = build_request_dump(
            args.platform, args.prompt, model=getattr(args, "model", None), provider=getattr(args, "provider", None),
        )
        serialized = json.dumps(body, ensure_ascii=False, indent=2) + "\n"
        fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(serialized)
    except (OSError, ValueError, RuntimeError, TypeError, ImportError) as exc:
        print(redact_sensitive_text(f"Could not export request: {exc}", force=True,
                                    redact_url_credentials=True), file=sys.stderr)
        return 1
    print(f"Request written to {output} (platform={args.platform}; secrets redacted; no inference sent)")
    return 0
