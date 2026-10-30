"""Local session policies retain the profile-wide toolset ceiling on create and resume."""

import json

import pytest

from gateway.session_policy import build_policy
from gateway.session_policy_refresh import refreshed_policy


@pytest.mark.parametrize("source", ["cli", "tui", "gui", "oneshot"])
def test_new_policy_carries_allowlist_and_resume_can_replace_it(tmp_path, source):
    before = {"agent": {"enabled_toolsets": ["terminal"]}, "platform_toolsets": {"cli": ["web"]}}
    policy = build_policy({"source": source, "cwd": str(tmp_path)}, before)
    assert json.loads(policy.config_json)["agent"]["enabled_toolsets"] == ["terminal"]
    assert "terminal" in policy.toolsets
    assert "web" not in policy.toolsets

    after = {"agent": {"enabled_toolsets": []}, "platform_toolsets": {"cli": ["web"]}}
    refreshed, _, _ = refreshed_policy(policy, after, json.loads(policy.terminal_json))
    assert json.loads(refreshed.config_json)["agent"]["enabled_toolsets"] == []
    assert "terminal" not in refreshed.toolsets
    assert "web" not in refreshed.toolsets
    # A policy refresh must not mutate the previous conversation's snapshot.
    assert json.loads(policy.config_json)["agent"]["enabled_toolsets"] == ["terminal"]
