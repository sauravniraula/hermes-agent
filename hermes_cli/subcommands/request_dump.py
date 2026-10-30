"""``hermes request-dump`` parser and lazy command dispatch."""

from __future__ import annotations

import argparse


def cmd_request_dump(args: argparse.Namespace) -> int:
    from hermes_cli.request_dump import run_request_dump

    return run_request_dump(args)


def build_request_dump_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "request-dump", help="Export a redacted provider-native request without running inference",
        description="Build a fresh request from the active profile's model, prompt and tools. "
                    "No inference or tool execution. The user prompt is empty unless --prompt is given. "
                    "Provider authentication/tool discovery may contact services during initialization.",
    )
    parser.add_argument("output", metavar="FILE", help="New JSON file to create (existing files are never overwritten)")
    parser.add_argument("--platform", default="cli", help="Platform to simulate (cli, tui, whatsapp, telegram, ...)")
    parser.add_argument("--prompt", default="", help="User message to include (default: empty)")
    parser.add_argument("--model", default=argparse.SUPPRESS, help="Model override for this export")
    parser.add_argument("--provider", default=argparse.SUPPRESS, help="Provider override for this export")
    parser.set_defaults(func=cmd_request_dump)
