"""
nano-agent: a complete coding agent in one readable file.

The loop is the whole idea: send the conversation, stream the reply, run any
tools it asked for, append the results, repeat until the model stops asking.
Everything else here exists to make that loop safe and able to run for a long time:

  tools         read, write, edit, bash, grep, glob, plus a subagent and memory
  permissions   anything that changes the world asks first, paths stay in the project
  streaming     thinking, text and tool calls print as they are generated
  caching       the prompt prefix never changes mid-session, so it stays cached
  compaction    near the context limit, the history is replaced by one summary
  subagents     a fresh agent with read-only tools explores and reports back

Run it:  python nano_agent.py "fix the failing test in tests/"
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import fnmatch
import json
import os
import platform
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import anthropic

# ----------------------------------------------------------------------------
# Configuration

MODEL = "claude-opus-4-5"
PRICES = {"input": 5.00, "output": 25.00, "cache_read": 0.50, "cache_write": 6.25}  # $/MTok, Opus 4.5
MAX_TOOL_OUTPUT = 30_000  # characters of tool output the model gets back
SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "__pycache__", "dist", "build", ".next", ".mypy_cache"}


@dataclass
class Config:
    root: Path
    model: str = MODEL
    max_tokens: int = 32_000
    compact_at: int = 150_000  # input tokens; compaction runs above this
    yolo: bool = False  # skip permission prompts (catastrophic commands stay blocked)
    memory_file: str = "NANO.md"


# ----------------------------------------------------------------------------
# Workspace: every path the model gives us is untrusted, so it gets resolved
# and confined to the project root before anything touches the disk.


class ToolError(Exception):
    """An error the model should see and recover from, not a crash."""


class Workspace:
    def __init__(self, root: Path):
        self.root = root.resolve()
        # mtime of every file the model has read, so edits to stale files fail.
        self.seen: dict[Path, int] = {}

    def path(self, p: str) -> Path:
        target = (self.root / p).resolve()
        if target != self.root and not target.is_relative_to(self.root):
            raise ToolError(f"{p} is outside the project root {self.root}")
        return target

    def rel(self, p: Path) -> str:
        return p.relative_to(self.root).as_posix() or "."

    def mark_seen(self, p: Path) -> None:
        self.seen[p] = p.stat().st_mtime_ns

    def check_fresh(self, p: Path) -> None:
        if p not in self.seen:
            raise ToolError(f"read {self.rel(p)} before editing it")
        if p.stat().st_mtime_ns != self.seen[p]:
            raise ToolError(f"{self.rel(p)} changed since you last read it; read it again")

    def walk(self, start: Path):
        for dirpath, dirnames, filenames in os.walk(start):
            dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
            for name in sorted(filenames):
                yield Path(dirpath) / name


# ----------------------------------------------------------------------------
# Tools. Each one is a plain function plus a JSON schema the model sees.


def read_file(ws: Workspace, path: str, offset: int = 1, limit: int = 2000) -> str:
    p = ws.path(path)
    if not p.is_file():
        raise ToolError(f"no such file: {path}")
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    ws.mark_seen(p)
    start = max(offset, 1) - 1
    chunk = lines[start : start + limit]
    body = "\n".join(f"{i:>6}\t{line}" for i, line in enumerate(chunk, start + 1))
    more = len(lines) - (start + len(chunk))
    return body + (f"\n... {more} more lines (use offset={start + len(chunk) + 1})" if more > 0 else "") or "(empty file)"


def write_file(ws: Workspace, path: str, content: str) -> str:
    p = ws.path(path)
    if p.exists():
        ws.check_fresh(p)  # overwriting is an edit: you must have seen what you replace
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    ws.mark_seen(p)
    return f"wrote {ws.rel(p)} ({len(content.splitlines())} lines)"


def edit_file(ws: Workspace, path: str, old: str, new: str, replace_all: bool = False) -> str:
    p = ws.path(path)
    if not p.is_file():
        raise ToolError(f"no such file: {path}")
    ws.check_fresh(p)
    text = p.read_text(encoding="utf-8")
    count = text.count(old)
    if count == 0:
        raise ToolError("old text not found; copy it exactly from read_file output, without the line numbers")
    if count > 1 and not replace_all:
        raise ToolError(f"old text occurs {count} times; add surrounding lines to make it unique, or set replace_all")
    p.write_text(text.replace(old, new) if replace_all else text.replace(old, new, 1), encoding="utf-8")
    ws.mark_seen(p)
    return f"edited {ws.rel(p)}" + (f" ({count} replacements)" if replace_all and count > 1 else "")


def bash(ws: Workspace, command: str, timeout: int = 120) -> str:
    try:
        r = subprocess.run(command, shell=True, cwd=ws.root, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=min(timeout, 600))
    except subprocess.TimeoutExpired:
        raise ToolError(f"command timed out after {timeout}s")
    out = (r.stdout + (f"\n[stderr]\n{r.stderr}" if r.stderr.strip() else "")).strip()
    return f"{out}\n[exit code {r.returncode}]".strip()


def grep(ws: Workspace, pattern: str, path: str = ".", glob: str = "*") -> str:
    try:
        rx = re.compile(pattern)
    except re.error as e:
        raise ToolError(f"bad regex: {e}")
    base = ws.path(path)
    files = [base] if base.is_file() else ws.walk(base)
    hits = []
    for f in files:
        if not fnmatch.fnmatch(f.name, glob):
            continue
        try:
            for n, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
                if rx.search(line):
                    hits.append(f"{ws.rel(f)}:{n}: {line.strip()[:200]}")
        except (UnicodeDecodeError, OSError):
            continue  # binary or unreadable
        if len(hits) >= 500:
            hits.append("... (stopped at 500 matches; narrow the pattern or path)")
            break
    return "\n".join(hits) or "no matches"


def glob_files(ws: Workspace, pattern: str) -> str:
    matches = [ws.rel(f) for f in ws.walk(ws.root) if fnmatch.fnmatch(ws.rel(f), pattern)]
    return "\n".join(matches[:1000]) or "no files match"


def remember(ws: Workspace, note: str, memory_file: str) -> str:
    # Saved for the *next* session. Changing the system prompt now would
    # invalidate the prompt cache and the thinking blocks already in the history.
    p = ws.root / memory_file
    with p.open("a", encoding="utf-8") as f:
        f.write(f"- {note.strip()}\n")
    return f"saved to {memory_file}; it will be in your instructions next session"


def tool(name: str, description: str, props: dict, required: list[str]) -> dict:
    return {
        "name": name,
        "description": description,
        "input_schema": {"type": "object", "properties": props, "required": required, "additionalProperties": False},
        # Stream big inputs (file contents) as they are generated. The API no
        # longer validates them for us, so validate_input() below does.
        "eager_input_streaming": True,
    }


S, I, B = {"type": "string"}, {"type": "integer"}, {"type": "boolean"}
TOOL_SPECS = [
    tool("read_file", "Read a text file. Output has line numbers. Read a file before editing it.",
         {"path": S, "offset": {**I, "description": "1-based first line"}, "limit": I}, ["path"]),
    tool("write_file", "Create a file, or replace one you have read, with the given content.",
         {"path": S, "content": S}, ["path", "content"]),
    tool("edit_file", "Replace an exact, unique piece of text in a file you have read. Prefer this over write_file.",
         {"path": S, "old": S, "new": S, "replace_all": B}, ["path", "old", "new"]),
    tool("bash", "Run a shell command in the project root and return its output and exit code.",
         {"command": S, "timeout": {**I, "description": "seconds, max 600"}}, ["command"]),
    tool("grep", "Search file contents with a Python regex. Returns path:line: text.",
         {"pattern": S, "path": S, "glob": {**S, "description": "filename filter, e.g. *.py"}}, ["pattern"]),
    tool("glob", "List files whose project-relative path matches a glob, e.g. src/*.ts or *test*.",
         {"pattern": S}, ["pattern"]),
    tool("task", "Hand a self-contained research question to a subagent with read-only tools and a fresh context. "
                 "Use it to explore a large codebase without filling your own context. Returns its final answer.",
         {"description": {**S, "description": "3-5 word label"}, "prompt": S}, ["description", "prompt"]),
    tool("remember", "Save a durable fact about this project or the user's preferences for future sessions.",
         {"note": S}, ["note"]),
]
READ_ONLY = {"read_file", "grep", "glob", "task"}
SUBAGENT_TOOLS = ["read_file", "grep", "glob"]


# ----------------------------------------------------------------------------
# Permissions: read-only tools just run. Anything else asks, unless --yolo.
# A few commands are refused outright, --yolo or not.

CATASTROPHIC = [
    r"\brm\s+-[a-z]*r[a-z]*f?\s+(/|~|\*|\.)(\s|$)", r"\bmkfs\b", r"\bdd\s+.*of=/dev/", r":\(\)\s*\{.*\};\s*:",
    r"\bgit\s+push\s+.*--force\b", r"\bshutdown\b|\breboot\b", r"\bformat\s+[a-z]:",
]


class Permissions:
    def __init__(self, yolo: bool, ask: Callable[[str], str] = input):
        self.yolo = yolo
        self.ask = ask
        self.always: set[str] = set()

    def check(self, name: str, args: dict) -> str | None:
        """None if allowed, otherwise the reason it was refused."""
        if name in READ_ONLY:
            return None
        if name == "bash" and any(re.search(p, args["command"]) for p in CATASTROPHIC):
            return "refused: this command is on nano-agent's never-run list"
        if self.yolo or name in self.always:
            return None
        answer = self.ask(f"  allow {name}? [y]es / [n]o / [a]lways for {name}: ").strip().lower()
        if answer in ("a", "always"):
            self.always.add(name)
            return None
        if answer in ("y", "yes"):
            return None
        return "the user declined this action. Ask them what they want instead."


# ----------------------------------------------------------------------------
# Terminal output, kept separate so subagents can print indented and tests can stay quiet.


class UI:
    DIM, BOLD, CYAN, RED, RESET = "\033[2m", "\033[1m", "\033[36m", "\033[31m", "\033[0m"

    def __init__(self, indent: str = "", quiet: bool = False, out=sys.stdout):
        self.indent, self.quiet, self.out = indent, quiet, out
        self.mode = None  # what we are printing: "thinking" | "text" | None
        self.at_line_start = True

    def _write(self, s: str) -> None:
        """Write, putting the indent at the start of every line (chunks can end mid-line)."""
        if self.quiet:
            return
        for piece in s.splitlines(keepends=True):
            if self.at_line_start and piece.strip("\n"):
                self.out.write(self.indent)
            self.out.write(piece)
            self.at_line_start = piece.endswith("\n")
        self.out.flush()

    def stream(self, kind: str, chunk: str) -> None:
        if self.mode != kind:
            self._write(("" if self.at_line_start else "\n") + (self.DIM if kind == "thinking" else self.RESET))
            self.mode = kind
        self._write(chunk)

    def end_stream(self) -> None:
        if self.mode:
            self._write(self.RESET + ("" if self.at_line_start else "\n"))
        self.mode = None

    def line(self, s: str, color: str = "") -> None:
        self.end_stream()
        self._write(f"{color}{s}{self.RESET if color else ''}\n")


def summarize_call(name: str, args: dict) -> str:
    key = {"read_file": "path", "write_file": "path", "edit_file": "path", "bash": "command",
           "grep": "pattern", "glob": "pattern", "task": "description", "remember": "note"}.get(name)
    detail = str(args.get(key, "")) if isinstance(args, dict) else ""
    return f"{name}({detail[:100]}{'...' if len(detail) > 100 else ''})"


SUMMARY_PROMPT = (
    "Summarize the transcript inside <summary></summary> tags. Include relevant information in the summary such "
    "that this conversation will be continued by a new context window without needing to redo work or be "
    "reprovided with relevant constraints or context. Be sure to preserve: (1) any difficulties or problems that "
    "came up, and how they were handled or resolved; (2) any possibilities, options, or approaches that were "
    "raised, tried, or set aside, and why; (3) anything that was asked for, decided, agreed, ruled out, or "
    "established as a preference, constraint, or boundary - stated exactly; (4) exactly where things stand now - "
    "what has been covered, settled, or completed so far; (5) anything still open, unresolved, promised, or "
    "expected to happen next; (6) specific details that would be hard to reconstruct - names, numbers, dates, "
    "exact wording, links or references - kept exactly. Be complete on these even at the cost of length; keep "
    "everything else concise. Weight the two voices differently: keep what the user said, asked for, shared, or "
    "established carefully and close to their own words; your own explanations and reasoning can be condensed "
    "much further, to what they concluded or produced - as long as nothing in the six items above is dropped. "
    "Do not call any tools while writing this summary; respond with text only."
)


@dataclass
class Usage:
    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0
    last_context: int = 0  # size of the most recent request: what compaction watches

    def add(self, u: Any) -> None:
        self.input += u.input_tokens or 0
        self.output += u.output_tokens or 0
        self.cache_read += getattr(u, "cache_read_input_tokens", 0) or 0
        self.cache_write += getattr(u, "cache_creation_input_tokens", 0) or 0
        self.last_context = ((u.input_tokens or 0) + (getattr(u, "cache_read_input_tokens", 0) or 0)
                             + (getattr(u, "cache_creation_input_tokens", 0) or 0))

    def cost(self) -> float:
        return (self.input * PRICES["input"] + self.output * PRICES["output"]
                + self.cache_read * PRICES["cache_read"] + self.cache_write * PRICES["cache_write"]) / 1e6

    def __str__(self) -> str:
        return (f"{self.input + self.cache_read + self.cache_write:,} in ({self.cache_read:,} cached) / "
                f"{self.output:,} out, ${self.cost():.2f}")


@dataclass
class Agent:
    client: Any
    cfg: Config
    ws: Workspace
    perms: Permissions
    system: str
    tool_names: list[str] = field(default_factory=lambda: [t["name"] for t in TOOL_SPECS])
    ui: UI = field(default_factory=UI)
    messages: list[dict] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)

    def __post_init__(self) -> None:
        # Fixed for the life of the agent: changing tools or system mid-session
        # would throw away the prompt cache and every thinking block so far.
        self.tools = [t for t in TOOL_SPECS if t["name"] in self.tool_names]
        self.specs = {t["name"]: t for t in self.tools}

    # -- one user turn: loop until the model stops calling tools ------------

    def run(self, prompt: str) -> str:
        self.messages.append({"role": "user", "content": prompt})
        while True:
            self.maybe_compact()
            reply = self.request()
            self.messages.append({"role": "assistant", "content": reply.content})
            calls = [b for b in reply.content if b.type == "tool_use"]

            if reply.stop_reason == "refusal":
                # A refusal can cut a tool call off mid-input: never run it, but
                # answer it so the history stays valid for the next turn.
                if calls:
                    self.messages.append({"role": "user", "content": [self.error(c.id, "not run") for c in calls]})
                self.ui.line("[the model declined this request]", UI.RED)
                return ""
            if not calls:
                return "".join(b.text for b in reply.content if b.type == "text")
            if reply.stop_reason == "max_tokens":
                # A cut-off tool input can still parse as a valid, truncated
                # object. Never run it: tell the model and let it retry smaller.
                results = [self.error(c.id, "your output hit max_tokens and this call was cut off; "
                                            "retry with a smaller input, e.g. several edit_file calls") for c in calls]
            else:
                results = self.run_tools(calls)
            # All results go back in one message, or the model learns to stop calling tools in parallel.
            self.messages.append({"role": "user", "content": results})

    def request(self, extra: list[dict] | None = None) -> Any:
        """Stream one model response, printing it as it arrives."""
        for attempt in range(3):
            try:
                with self.client.messages.stream(
                    model=self.cfg.model,
                    max_tokens=self.cfg.max_tokens,
                    system=[{"type": "text", "text": self.system, "cache_control": {"type": "ephemeral"}}],
                    tools=self.tools,
                    messages=self.messages + (extra or []),
                    thinking={"type": "enabled", "budget_tokens": 8_000},
                ) as stream:
                    for event in stream:
                        if event.type == "thinking":
                            self.ui.stream("thinking", event.thinking)
                        elif event.type == "text":
                            self.ui.stream("text", event.text)
                    reply = stream.get_final_message()
                self.ui.end_stream()
                self.usage.add(reply.usage)
                return reply
            except ValueError:
                # Tool-input JSON the SDK could not parse at all. The block never
                # completed, so there is no id to answer: ask again (bounded).
                self.ui.line("[malformed tool input from the model; retrying]", UI.RED)
                if attempt == 2:
                    raise

    # -- compaction --------------------------------------------------------

    def maybe_compact(self, force: bool = False) -> bool:
        """
        Near the context limit, replace the whole history with one summary.

        This is "simple compaction": nothing from before the summary is
        replayed - no old turns, no old thinking blocks - so nothing carried
        over is tied to a transcript that no longer exists. It never runs in
        the middle of a tool round, and a brand-new user prompt is kept verbatim
        after the summary rather than summarized away.
        """
        if not force and self.usage.last_context < self.cfg.compact_at:
            return False
        last = self.messages[-1] if self.messages else None
        if last is None or (last["role"] == "assistant" and any(
                getattr(b, "type", None) == "tool_use" for b in last["content"])):
            return False  # a tool call is still waiting for its result
        new_prompt = self.messages.pop()["content"] if isinstance(last["content"], str) and last["role"] == "user" else None
        if not self.messages:
            self.messages.append({"role": "user", "content": new_prompt})
            return False  # nothing before it to summarize

        self.ui.line(f"[compacting {self.usage.last_context:,} tokens of history]", UI.DIM)
        quiet, self.ui.quiet = self.ui.quiet, True
        try:
            reply = self.request(extra=[{"role": "user", "content": SUMMARY_PROMPT}])
        finally:
            self.ui.quiet = quiet
        text = "".join(b.text for b in reply.content if b.type == "text")
        match = re.search(r"<summary>(.*)</summary>", text, re.S)
        summary = (match.group(1) if match else text).strip()
        content = f"This session was compacted. Summary of everything so far:\n\n<summary>\n{summary}\n</summary>\n\n"
        content += (f"New request:\n{new_prompt}" if new_prompt is not None
                    else "Continue from where things stand, without redoing finished work.")
        self.messages = [{"role": "user", "content": content}]
        self.usage.last_context = 0
        self.ui.line(f"[compacted to {len(summary):,} characters]", UI.DIM)
        return True

    # -- tools -------------------------------------------------------------

    def run_tools(self, calls: list[Any]) -> list[dict]:
        results: dict[str, dict] = {}
        parallel = []
        for call in calls:
            self.ui.line(f"> {summarize_call(call.name, call.input)}", UI.CYAN)
            denied = self.perms.check(call.name, call.input)
            if denied:
                results[call.id] = self.error(call.id, denied)
            elif call.name in READ_ONLY:
                parallel.append(call)  # safe to run side by side
            else:
                results[call.id] = self.execute(call)  # writes run one at a time, in order

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            for call, result in zip(parallel, pool.map(self.execute, parallel)):
                results[call.id] = result
        return [results[c.id] for c in calls]

    def execute(self, call: Any) -> dict:
        args, ws = call.input, self.ws
        try:
            match call.name:
                case "read_file": out = read_file(ws, **args)
                case "write_file": out = write_file(ws, **args)
                case "edit_file": out = edit_file(ws, **args)
                case "bash": out = bash(ws, **args)
                case "grep": out = grep(ws, **args)
                case "glob": out = glob_files(ws, **args)
                case "task": out = self.subagent(args["description"], args["prompt"])
                case "remember": out = remember(ws, args["note"], self.cfg.memory_file)
                case _: raise ToolError(f"unknown tool {call.name}")
        except ToolError as e:
            self.ui.line(f"  ! {e}", UI.RED)
            return self.error(call.id, str(e))
        except Exception as e:  # a bug in a tool should not kill the session
            return self.error(call.id, f"{type(e).__name__}: {e}")
        if len(out) > MAX_TOOL_OUTPUT:
            out = out[:MAX_TOOL_OUTPUT] + f"\n... [cut: {len(out):,} characters total; narrow your request]"
        return {"type": "tool_result", "tool_use_id": call.id, "content": out}

    @staticmethod
    def error(tool_use_id: str, message: str) -> dict:
        return {"type": "tool_result", "tool_use_id": tool_use_id, "content": message, "is_error": True}

    # -- subagents ---------------------------------------------------------

    def subagent(self, description: str, prompt: str) -> str:
        """A fresh agent with read-only tools. Only its final answer comes back."""
        child = Agent(
            client=self.client,
            cfg=Config(**self.cfg.__dict__),
            ws=self.ws,
            perms=self.perms,
            system=self.system + "\n\nYou are a subagent. Investigate, then reply with a complete, concise report. "
                                 "Your final message is all the caller sees.",
            tool_names=SUBAGENT_TOOLS,
            ui=UI(indent=self.ui.indent + "  | ", quiet=self.ui.quiet, out=self.ui.out),
        )
        self.ui.line(f"  [subagent: {description}]", UI.DIM)
        answer = child.run(prompt)
        self.usage.input += child.usage.input
        self.usage.output += child.usage.output
        self.usage.cache_read += child.usage.cache_read
        self.usage.cache_write += child.usage.cache_write
        return answer or "(the subagent returned no answer)"

def build_system(cfg: Config) -> str:
    memory_path = cfg.root / cfg.memory_file
    memory = memory_path.read_text(encoding="utf-8").strip() if memory_path.exists() else ""
    shell = "cmd.exe" if platform.system() == "Windows" else "sh"
    return f"""You are nano-agent, a coding agent working in a user's project through tools.

Project root: {cfg.root}
Platform: {platform.system()} {platform.release()}; bash runs commands with {shell}
Date: {datetime.date.today().isoformat()}

How to work:
- Explore before changing things: glob and grep to find code, read_file before any edit.
- Make the smallest change that fully solves the task, in the style of the surrounding code.
- Prefer edit_file for changes to existing files. Run the project's tests or build after changing code.
- Call independent read-only tools in parallel in a single turn.
- Use task to delegate broad searches so your own context stays focused.
- Use remember for durable facts worth knowing next session (commands, conventions, preferences).
- If an action is refused, do not try to get around the refusal; ask the user.
- When you finish, say what you changed and how you verified it, plainly. If something failed, say so.
""" + (f"\nProject memory ({cfg.memory_file}):\n{memory}\n" if memory else "")


def repl(agent: Agent) -> None:
    print(f"nano-agent ({agent.cfg.model}) in {agent.cfg.root}")
    print("Commands: /cost  /compact  /exit\n")
    while True:
        try:
            prompt = input(f"{UI.BOLD}> {UI.RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if prompt in ("/exit", "/quit"):
            return
        if prompt == "/cost":
            print(agent.usage)
            continue
        if prompt == "/compact":
            if not agent.maybe_compact(force=True):
                print("nothing to compact yet")
            continue
        if prompt:
            turn(agent, prompt)


def turn(agent: Agent, prompt: str) -> str:
    before = len(agent.messages)
    try:
        return agent.run(prompt)
    except KeyboardInterrupt:
        agent.ui.line("[interrupted]", UI.RED)
        # Leave the history valid: drop the half-finished turn entirely.
        del agent.messages[before:]
        return ""
    except anthropic.AuthenticationError:
        agent.ui.line("[no valid credentials: set ANTHROPIC_API_KEY, or run `ant auth login`]", UI.RED)
        del agent.messages[before:]
        return ""
    except anthropic.APIError as e:  # the SDK already retried 429s, 5xx and dropped connections
        agent.ui.line(f"[API error: {e}]", UI.RED)
        del agent.messages[before:]
        return ""
    finally:
        agent.ui.line(f"[{agent.usage}]", UI.DIM)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="A complete coding agent in one file.")
    ap.add_argument("prompt", nargs="*", help="run one task and exit (omit for an interactive session)")
    ap.add_argument("--root", default=".", help="project directory (default: current)")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--yolo", action="store_true", help="don't ask before writes and commands")
    ap.add_argument("--compact-at", type=int, default=150_000, help="compact above this many input tokens")
    a = ap.parse_args(argv)

    # The SDK finds credentials in env vars or an `ant auth login` profile; fail early if there are none.
    if not (any(os.environ.get(k) for k in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE"))
            or (Path.home() / ".config" / "anthropic").is_dir()):
        print("nano-agent: no Anthropic credentials. Set ANTHROPIC_API_KEY, or run `ant auth login`.", file=sys.stderr)
        return 1

    cfg = Config(root=Path(a.root).resolve(), model=a.model, yolo=a.yolo, compact_at=a.compact_at)
    agent = Agent(client=anthropic.Anthropic(), cfg=cfg, ws=Workspace(cfg.root),
                  perms=Permissions(cfg.yolo), system=build_system(cfg))
    if a.prompt:
        turn(agent, " ".join(a.prompt))
    else:
        repl(agent)
    return 0


if __name__ == "__main__":
    sys.exit(main())
