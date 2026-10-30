"""Shared profile-aware scope for read-only capability/request inspection."""

from contextlib import contextmanager
import os


def validate_platform(platform: str) -> None:
    from agent.prompt_builder import PLATFORM_HINTS
    from gateway.platform_registry import platform_registry
    from hermes_cli.plugins import discover_plugins

    if platform in PLATFORM_HINTS:
        return
    discover_plugins()
    if platform_registry.get(platform) is None:
        raise ValueError(f"Unknown platform {platform!r}; use cli, tui, or a registered messaging platform")


@contextmanager
def platform_scope(platform: str):
    # Skill selection consults this before the session ContextVar; restore for embedded callers.
    validate_platform(platform)
    previous = os.environ.get("HERMES_PLATFORM")
    os.environ["HERMES_PLATFORM"] = platform
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("HERMES_PLATFORM", None)
        else:
            os.environ["HERMES_PLATFORM"] = previous
