"""Tests for nano-agent. A scripted fake client stands in for the API, so these run offline."""

import copy
import io
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

import nano_agent as na


# -- a fake Anthropic client that replays scripted replies ---------------------

def text(t):
    return NS(type="text", text=t)


def call(id, name, **input):
    return NS(type="tool_use", id=id, name=name, input=input)


def reply(*content, stop="end_turn", tokens=1000):
    if stop == "end_turn" and any(b.type == "tool_use" for b in content):
        stop = "tool_use"
    usage = NS(input_tokens=tokens, output_tokens=50, cache_read_input_tokens=0, cache_creation_input_tokens=0)
    return NS(content=list(content), stop_reason=stop, usage=usage)


class FakeStream:
    def __init__(self, r):
        self.r = r

    def __enter__(self):
        if isinstance(self.r, Exception):
            raise self.r
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        for b in self.r.content:
            if b.type == "text":
                yield NS(type="text", text=b.text)

    def get_final_message(self):
        return self.r


class FakeClient:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.requests = []
        self.messages = NS(stream=self.stream)

    def stream(self, **kwargs):
        # Deep-copy what was sent: the agent keeps mutating its history afterwards.
        self.requests.append(copy.deepcopy(kwargs))
        return FakeStream(self.replies.pop(0))


@pytest.fixture
def project(tmp_path):
    (tmp_path / "app.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "util.py").write_text("TODO = 1\n", encoding="utf-8")
    return tmp_path


def make_agent(root, client, answers=(), yolo=False, **cfg):
    answers = list(answers)
    perms = na.Permissions(yolo=yolo, ask=lambda _q: answers.pop(0))
    config = na.Config(root=root, yolo=yolo, **cfg)
    return na.Agent(client=client, cfg=config, ws=na.Workspace(root), perms=perms,
                    system="test system", ui=na.UI(quiet=True))


# -- the loop --------------------------------------------------------------

def test_plain_answer_and_request_shape(project):
    client = FakeClient(reply(text("hello")))
    agent = make_agent(project, client)
    assert agent.run("hi") == "hello"
    req = client.requests[0]
    assert req["model"] == "claude-opus-4-5"
    assert req["thinking"] == {"type": "enabled", "budget_tokens": 8_000}
    assert req["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert all(t["eager_input_streaming"] for t in req["tools"])


def test_read_then_edit_fixes_the_bug(project):
    client = FakeClient(
        reply(call("t1", "read_file", path="app.py")),
        reply(call("t2", "edit_file", path="app.py", old="return a - b", new="return a + b")),
        reply(text("fixed")),
    )
    agent = make_agent(project, client, answers=["y"])
    assert agent.run("fix add") == "fixed"
    assert "return a + b" in (project / "app.py").read_text()
    # every tool result went back, and the request prefix (system + tools) never changed
    assert client.requests[1]["messages"][-1]["content"][0]["content"].startswith("     1\tdef add")
    assert len({str(r["tools"]) + str(r["system"]) for r in client.requests}) == 1


def test_declined_permission_leaves_file_alone(project):
    client = FakeClient(
        reply(call("t1", "read_file", path="app.py")),
        reply(call("t2", "write_file", path="app.py", content="oops")),
        reply(text("ok")),
    )
    make_agent(project, client, answers=["n"]).run("rewrite it")
    assert (project / "app.py").read_text().startswith("def add")
    result = client.requests[2]["messages"][-1]["content"][0]
    assert result["is_error"] and "declined" in result["content"]


def test_always_answer_is_remembered_per_tool(project):
    client = FakeClient(
        reply(call("t1", "bash", command="echo one"), call("t2", "bash", command="echo two")),
        reply(text("done")),
    )
    make_agent(project, client, answers=["a"]).run("go")  # only one answer available: second call must not ask
    results = client.requests[1]["messages"][-1]["content"]
    assert [r.get("is_error", False) for r in results] == [False, False]
    assert "one" in results[0]["content"] and "two" in results[1]["content"]


def test_catastrophic_commands_are_refused_even_in_yolo(project):
    client = FakeClient(reply(call("t1", "bash", command="rm -rf /")), reply(text("ok")))
    make_agent(project, client, yolo=True).run("clean up")
    result = client.requests[1]["messages"][-1]["content"][0]
    assert result["is_error"] and "never-run" in result["content"]


def test_memory_is_loaded_into_the_system_prompt(project):
    (project / "NANO.md").write_text("- tests run with `pytest -q`\n")
    system = na.build_system(na.Config(root=project))
    assert "tests run with `pytest -q`" in system
    na.remember(na.Workspace(project), "use tabs", "NANO.md")
    assert (project / "NANO.md").read_text().endswith("- use tabs\n")


# -- workspace safety ------------------------------------------------------


def test_edits_require_a_fresh_read(project):
    ws = na.Workspace(project)
    with pytest.raises(na.ToolError, match="read app.py before editing"):
        na.edit_file(ws, "app.py", "a - b", "a + b")
    na.read_file(ws, "app.py")
    (project / "app.py").write_text("changed underneath\n")
    import os
    os.utime(project / "app.py", ns=(1, 1))
    with pytest.raises(na.ToolError, match="changed since you last read"):
        na.edit_file(ws, "app.py", "changed", "x")


def test_ambiguous_edit_is_rejected(project):
    ws = na.Workspace(project)
    (project / "dup.py").write_text("x = 1\nx = 1\n")
    na.read_file(ws, "dup.py")
    with pytest.raises(na.ToolError, match="occurs 2 times"):
        na.edit_file(ws, "dup.py", "x = 1", "x = 2")
    assert na.edit_file(ws, "dup.py", "x = 1", "x = 2", replace_all=True).endswith("(2 replacements)")

def test_parallel_reads_come_back_in_one_message_in_order(project):
    client = FakeClient(
        reply(call("a", "read_file", path="src/util.py"), call("b", "grep", pattern="def "), call("c", "glob", pattern="*.py")),
        reply(text("ok")),
    )
    make_agent(project, client).run("look around")
    results = client.requests[1]["messages"][-1]["content"]
    assert [r["tool_use_id"] for r in results] == ["a", "b", "c"]
    assert "TODO = 1" in results[0]["content"]
    assert "app.py:1: def add(a, b):" in results[1]["content"]
    assert results[2]["content"].splitlines() == ["app.py", "src/util.py"]


def test_paths_cannot_escape_the_project(project):
    ws = na.Workspace(project)
    with pytest.raises(na.ToolError, match="outside the project"):
        na.read_file(ws, "../secret.txt")


def test_read_file_pages_long_files(project):
    (project / "long.txt").write_text("\n".join(f"line {i}" for i in range(1, 11)))
    out = na.read_file(na.Workspace(project), "long.txt", offset=3, limit=2)
    assert out.splitlines()[0] == "     3\tline 3"
    assert "6 more lines (use offset=5)" in out


# -- compaction -------------------------------------------------------------

# -- compaction -------------------------------------------------------------


def test_compaction_replaces_history_and_keeps_the_new_prompt(project):
    client = FakeClient(
        reply(text("first answer"), tokens=200_000),
        reply(text("<summary>user wants add() fixed; nothing changed yet</summary>")),
        reply(text("second answer")),
    )
    agent = make_agent(project, client, compact_at=150_000)
    agent.run("first question")
    assert agent.run("second question") == "second answer"

    summarize = client.requests[1]
    assert summarize["messages"][-1]["content"] == na.SUMMARY_PROMPT
    assert summarize["tools"] == client.requests[0]["tools"], "tools stay identical, even for the summary"

    after = client.requests[2]["messages"]
    assert len(after) == 1, "nothing from before the summary is replayed"
    assert "user wants add() fixed" in after[0]["content"]
    assert after[0]["content"].endswith("New request:\nsecond question")


def test_compaction_never_splits_a_tool_round(project):
    agent = make_agent(project, FakeClient(), compact_at=1)
    agent.usage.last_context = 10
    agent.messages = [{"role": "user", "content": "go"},
                      {"role": "assistant", "content": [call("t1", "read_file", path="app.py")]}]
    assert agent.maybe_compact() is False


# -- subagents ------------------------------------------------------------

# -- subagents ------------------------------------------------------------


def test_subagent_gets_read_only_tools_and_returns_its_answer(project):
    client = FakeClient(
        reply(call("t1", "task", description="find todos", prompt="Where are the TODOs?")),
        reply(call("s1", "grep", pattern="TODO")),     # subagent's first turn
        reply(text("src/util.py:1 has the only TODO")),  # subagent's answer
        reply(text("there is one TODO")),
    )
    agent = make_agent(project, client)
    assert agent.run("any todos?") == "there is one TODO"
    child_req = client.requests[1]
    assert {t["name"] for t in child_req["tools"]} == {"read_file", "grep", "glob"}
    assert len(child_req["messages"]) == 1, "the subagent starts with a fresh context"
    result = client.requests[3]["messages"][-1]["content"][0]
    assert result["content"] == "src/util.py:1 has the only TODO"
    assert agent.usage.output == 200, "subagent usage is counted in the parent's total"


def test_ui_indents_every_line_of_subagent_output():
    out = io.StringIO()
    ui = na.UI(indent="  | ", out=out)
    ui.line("hello")
    ui.stream("text", "streamed ")
    ui.stream("text", "across chunks\nand lines")
    ui.end_stream()
    reset = na.UI.RESET
    assert out.getvalue() == f"  | hello\n  | {reset}streamed across chunks\n  | and lines{reset}\n"
