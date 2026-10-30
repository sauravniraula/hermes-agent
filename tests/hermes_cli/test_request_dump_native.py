"""Native provider payloads and safety boundaries of request-dump."""

import json
import os
from pathlib import Path
import stat
import subprocess
import sys

import pytest

from hermes_cli.config import atomic_config_replace
from hermes_cli.config_backend import read_config_doc
from tests.e2e.core.history._helpers import OFFLINE_CONFIG, NO_BACKGROUND_REVIEW
from tests.fakes.fake_llm_provider import FakeLLMServer, write_hermes_home


@pytest.fixture
def profile(tmp_path, monkeypatch):
    home = Path(os.environ["HERMES_HOME"])
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-e2e")
    with FakeLLMServer() as server:
        write_hermes_home(home, server.base_url, extra_config=OFFLINE_CONFIG + NO_BACKGROUND_REVIEW)
        cfg = read_config_doc(home / "config.yaml")
        cfg["agent"].update(enabled_toolsets=["file", "skills"], environment_probe=False)
        atomic_config_replace(home / "config.yaml", cfg)
        (home / "SOUL.md").write_text("PROFILE NATIVE SOUL")
        yield home, server


@pytest.mark.parametrize("api_mode,provider,model,base_url", [
    ("codex_responses", "openai-codex", "gpt-5.1-codex", "https://chatgpt.com/backend-api/codex"),
    ("anthropic_messages", "anthropic", "claude-sonnet-4-5", "https://api.anthropic.com"),
])
def test_real_agent_native_formats_without_inference(profile, monkeypatch, tmp_path, api_mode, provider, model, base_url):
    from hermes_cli.main import _build_cli_parser
    import httpx

    home, _server = profile
    cfg = read_config_doc(home / "config.yaml")
    cfg["model"].update(default=model, provider=provider, base_url=base_url)
    atomic_config_replace(home / "config.yaml", cfg)
    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", lambda **_: {
        "provider": provider, "requested_provider": provider, "api_mode": api_mode,
        "base_url": base_url, "api_key": "sk-fake-e2e",
    })

    def no_network(*_args, **_kwargs):
        raise AssertionError("request-dump must not send inference")

    monkeypatch.setattr(httpx.Client, "send", no_network)
    parser, _ = _build_cli_parser()
    output = tmp_path / f"{api_mode}.json"
    args = parser.parse_args(["request-dump", str(output), "--platform", "whatsapp"])
    assert args.func(args) == 0
    body = json.loads(output.read_text())
    assert body["model"] == model
    assert "extra_headers" not in body and "timeout" not in body
    assert body["tools"]
    if api_mode == "codex_responses":
        assert "PROFILE NATIVE SOUL" in body["instructions"]
        assert "You are on WhatsApp." in body["instructions"]
        assert body["input"][-1]["content"] == [{"type": "input_text", "text": ""}]
        assert body["stream"] is True and body["store"] is False
    else:
        system = body["system"]
        system_text = system if isinstance(system, str) else "\n".join(b.get("text", "") for b in system)
        assert "PROFILE NATIVE SOUL" in system_text and "You are on WhatsApp." in system_text
        # Anthropic rejects blank user content; its real converter uses this placeholder.
        assert body["messages"][-1]["content"] == "(empty message)"


def test_export_redacts_even_when_normal_redaction_disabled(profile, monkeypatch, tmp_path):
    from hermes_cli.main import _build_cli_parser

    _home, server = profile
    monkeypatch.setenv("HERMES_REDACT_SECRETS", "false")
    secret = "sk-proj-" + "a" * 80
    parser, _ = _build_cli_parser()
    output = tmp_path / "redacted.json"
    args = parser.parse_args(["request-dump", str(output), "--prompt", f"OPENAI_API_KEY={secret}"])
    assert args.func(args) == 0
    assert secret not in output.read_text()
    assert server.requests == []


def test_existing_file_is_preserved_before_agent_initialization(tmp_path, monkeypatch):
    from hermes_cli.main import _build_cli_parser
    import hermes_cli.request_dump as dump

    output = tmp_path / "existing.json"
    output.write_text("KEEP EXISTING CONTENT")

    def unexpected_build(*_args, **_kwargs):
        raise AssertionError("must refuse an existing output before provider initialization")

    monkeypatch.setattr(dump, "_build_agent", unexpected_build)
    parser, _ = _build_cli_parser()
    args = parser.parse_args(["request-dump", str(output)])
    assert args.func(args) == 1
    assert output.read_text() == "KEEP EXISTING CONTENT"


def test_unknown_platform_does_not_create_output(tmp_path):
    from hermes_cli.main import _build_cli_parser

    parser, _ = _build_cli_parser()
    output = tmp_path / "bad.json"
    args = parser.parse_args(["request-dump", str(output), "--platform", "not-a-platform"])
    assert args.func(args) == 1
    assert not output.exists()


@pytest.mark.platforms("posix")
def test_dump_file_is_private(profile, tmp_path):
    from hermes_cli.main import _build_cli_parser

    parser, _ = _build_cli_parser()
    output = tmp_path / "private.json"
    args = parser.parse_args(["request-dump", str(output)])
    assert args.func(args) == 0
    assert stat.S_IMODE(output.stat().st_mode) == 0o600


def test_model_override_survives_subparser_defaults(tmp_path):
    from hermes_cli.main import _build_cli_parser

    parser, _ = _build_cli_parser()
    args = parser.parse_args(["--model", "chosen-model", "request-dump", str(tmp_path / "request.json")])
    assert args.model == "chosen-model"


def test_named_profile_commands_use_the_selected_home(tmp_path, monkeypatch):
    """Real CLI subprocesses: global -p, request building and both breakdown handlers."""
    root = Path(__file__).resolve().parents[2]
    monkeypatch.chdir(tmp_path)
    home = tmp_path / ".hermes/profiles/work"
    with FakeLLMServer() as server:
        write_hermes_home(home, server.base_url, extra_config=OFFLINE_CONFIG + NO_BACKGROUND_REVIEW)
        cfg = read_config_doc(home / "config.yaml")
        cfg["agent"].update(enabled_toolsets=["file", "skills"], environment_probe=False)
        cfg["skills"]["enabled"] = ["work-skill"]
        atomic_config_replace(home / "config.yaml", cfg)
        (home / "SOUL.md").write_text("SELECTED WORK PROFILE SOUL")
        skill = home / "skills/work-skill/SKILL.md"
        skill.parent.mkdir(parents=True)
        skill.write_text("---\nname: work-skill\ndescription: Work instructions.\n---\n# Work\nDo work.")
        env = {**os.environ, "HOME": str(tmp_path), "HERMES_HOME": str(tmp_path / ".hermes"),
               "PYTHONPATH": str(root), "OPENAI_API_KEY": "sk-fake-e2e"}

        def invoke(*args):
            result = subprocess.run([sys.executable, "-m", "hermes_cli.main", "-p", "work", *args],
                                    env=env, cwd=tmp_path, capture_output=True, text=True, timeout=45, check=False)
            assert result.returncode == 0, result.stderr
            return result

        output = tmp_path / "work.json"
        invoke("request-dump", str(output), "--platform", "whatsapp")
        body = json.loads(output.read_text())
        assert "SELECTED WORK PROFILE SOUL" in body["messages"][0]["content"]
        assert body["messages"][-1]["content"] == ""
        tools = json.loads(invoke("tools", "breakdown", "--json").stdout)
        assert "terminal" not in {r["name"] for r in tools["tools"]}
        skills = json.loads(invoke("skills", "breakdown", "--platform", "whatsapp", "--json").stdout)
        assert "work-skill" in {r["name"] for r in skills["skills"]}
        assert server.requests == []
