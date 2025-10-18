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
    assert req["model"] == "claude-sonnet-4-5"
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


# -- workspace safety ------------------------------------------------------

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

# -- subagents ------------------------------------------------------------

def test_ui_indents_every_line_of_subagent_output():
    out = io.StringIO()
    ui = na.UI(indent="  | ", out=out)
    ui.line("hello")
    ui.stream("text", "streamed ")
    ui.stream("text", "across chunks\nand lines")
    ui.end_stream()
    reset = na.UI.RESET
    assert out.getvalue() == f"  | hello\n  | {reset}streamed across chunks\n  | and lines{reset}\n"
