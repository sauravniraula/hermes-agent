"""Long capability identifiers must not shift the token columns."""

import pytest

from hermes_cli.capability_breakdown import _render_skills, _render_tools


@pytest.mark.parametrize("names", [[], ["read_file"], [
    "read_file", "mcp__neon__delete_storage_objects_by_prefix",
    "mcp__neon__delete_functions_custom_domain_with_a_long_suffix",
]])
def test_tool_columns_align_without_truncating_names(names):
    rows = [{"name": name, "disclosure": "deferred", "tokens": 1234, "upfront_tokens": 0}
            for name in names]
    output = _render_tools({"platform": "cli", "counts": {
        "active": len(rows), "direct": 0, "deferred": len(rows)},
        "direct_schema_tokens": 0, "tools": rows})
    header, *lines = output.splitlines()[5:]
    for name, line in zip(names, lines):
        assert line.startswith(name + " ")
        assert line.index("deferred") == header.index("Disclosure")
        assert line.index("1,234") + len("1,234") == header.index("Schema tokens") + len("Schema tokens")
        assert line.index("0", line.index("1,234")) + 1 == len(header)


@pytest.mark.parametrize("names", [[], ["short"], ["short", "plugin:skill-with-a-name-longer-than-the-old-fixed-column-width"]])
def test_skill_columns_align_without_truncating_names(names):
    rows = [{"name": name, "index_tokens": 123, "file_tokens": None} for name in names]
    output = _render_skills({"platform": "cli", "count": len(rows), "index_total_tokens": 123,
                             "skills": rows})
    header, *lines = output.splitlines()[5:]
    for name, line in zip(names, lines):
        assert line.startswith(name + " ")
        assert line.index("123") + 3 == header.index("Index tokens") + len("Index tokens")
        assert line.index("n/a") + 3 == len(header)
