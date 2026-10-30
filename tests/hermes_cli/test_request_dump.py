"""Request-dump CLI contracts: native requests, profile ownership and safe export."""

from pathlib import Path


def test_request_dump_parser_defaults_to_cli_and_empty_prompt(tmp_path):
    from hermes_cli.main import _build_cli_parser

    parser, _ = _build_cli_parser()
    args = parser.parse_args(["request-dump", str(tmp_path / "request.json")])
    assert args.platform == "cli"
    assert args.prompt == ""
    assert Path(args.output).name == "request.json"
    assert callable(args.func)


def test_exports_real_profile_request_without_calling_model(tmp_path, monkeypatch):
    import json
    import os
    from hermes_cli.config import atomic_config_replace
    from hermes_cli.config_backend import read_config_doc
    from hermes_cli.main import _build_cli_parser
    from tests.e2e.core.history._helpers import OFFLINE_CONFIG, NO_BACKGROUND_REVIEW
    from tests.fakes.fake_llm_provider import FakeLLMServer, write_hermes_home

    home = Path(os.environ["HERMES_HOME"])
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-e2e")
    with FakeLLMServer() as server:
        write_hermes_home(home, server.base_url, extra_config=OFFLINE_CONFIG + NO_BACKGROUND_REVIEW)
        cfg = read_config_doc(home / "config.yaml")
        cfg["agent"].update(enabled_toolsets=["file", "memory", "skills"], environment_probe=False)
        atomic_config_replace(home / "config.yaml", cfg)
        (home / "SOUL.md").write_text("PROFILE SOUL FROM DISK")
        (home / "memories").mkdir(exist_ok=True)
        (home / "memories/USER.md").write_text("PROFILE USER FACT")
        output = tmp_path / "request.json"
        parser, _ = _build_cli_parser()
        args = parser.parse_args(["request-dump", str(output)])
        assert args.func(args) == 0
        body = json.loads(output.read_text())
        assert body["model"] == "fake-model"
        assert body["messages"][-1] == {"role": "user", "content": ""}
        prompt = body["messages"][0]["content"]
        assert "PROFILE SOUL FROM DISK" in prompt
        assert "PROFILE USER FACT" in prompt
        assert "You are in a plain terminal (CLI)." in prompt
        assert "read_file" in {t["function"]["name"] for t in body["tools"]}
        assert server.requests == []
