# nano-agent

A complete coding agent in one readable Python file, about 480 lines of code.

```sh
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...
python nano_agent.py "the test in tests/test_dates.py fails on leap years, fix it"
python nano_agent.py            # interactive session
```

Agent frameworks hide the parts that matter. This is the opposite: one file you can read top to bottom in twenty minutes, that still does what a real agent has to do. It reads and edits code, runs commands, asks before it changes anything, and streams its thinking and tool calls as they happen.

## What's in the file

| Section | What it does |
|---|---|
| **Workspace** | Resolves every model-supplied path and refuses anything outside the project root. |
| **Tools** | `read_file`, `write_file`, `edit_file` (exact string replacement), `bash`, `grep`, `glob`. Each is a plain function plus a JSON schema. |
| **Permissions** | Read-only tools run freely. Writes and commands ask `[y]es / [n]o / [a]lways`. A short list of catastrophic commands (`rm -rf /`, `mkfs`, force-push, ...) is refused even with `--yolo`. |
| **The loop** | Stream a reply, run the tools it asked for, append every result, repeat. About 40 lines. |

## Design notes

These are the decisions worth stealing.

**The prompt prefix never changes mid-session.** The system prompt and tool list are fixed when the agent starts, and both are marked for prompt caching. A long session re-reads them every turn at cache prices.

**Failure is information.** Tool errors and refused permissions come back as `is_error` tool results with a reason the model can act on. Nothing in a tool can crash the session, and Ctrl-C drops the half-finished turn so the history stays valid.

## Options

```
python nano_agent.py [prompt ...] [--root DIR] [--model MODEL] [--yolo]
```

In an interactive session, `/cost` shows tokens and dollars so far, and `/exit` quits.

The default model is `claude-sonnet-4-5`.

## Tests

```sh
pip install -r requirements-dev.txt
pytest -q
```

The tests drive the real loop against a scripted fake client, so they run offline in about 2 seconds. They check the request shape, the read, edit and fix flow, permissions (decline, always, the never-run list), path escapes, tool errors, and the interactive output.

MIT licensed.
