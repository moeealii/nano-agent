# nano-agent

A complete coding agent in one readable Python file.

```sh
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...        # or OPENAI_API_KEY (+ OPENAI_BASE_URL)
python nano_agent.py "the test in tests/test_dates.py fails on leap years, fix it"
python nano_agent.py            # interactive session
```

Agent frameworks hide the parts that matter. This is the opposite: one file you can read top to bottom in twenty minutes, that still does what a real agent has to do. It reads and edits code, runs commands, asks before it changes anything, delegates research to subagents, remembers things between sessions, keeps working when the conversation outgrows the context window, and talks to whichever model endpoint you point it at.

## What's in the file

| Section | What it does |
|---|---|
| **Workspace** | Resolves every model-supplied path and refuses anything outside the project root. Records the mtime of every file the model reads, so an edit to a file that changed since is rejected rather than silently clobbering it. |
| **Tools** | `read_file`, `write_file`, `edit_file` (exact, unique string replacement), `bash`, `grep`, `glob`, `task` (subagent), `remember` (memory). Each is a plain function plus a JSON schema. |
| **Providers** | One normalized transcript, translated to Anthropic's Messages API or to any OpenAI-compatible `/chat/completions` endpoint. The loop, tools, permissions and compaction never learn which one is in use. |
| **Permissions** | Read-only tools run freely. Writes and commands ask `[y]es / [n]o / [a]lways`. A short list of catastrophic commands (`rm -rf /`, `mkfs`, force-push, ...) is refused even with `--yolo`. |
| **The loop** | Stream a reply, run the tools it asked for, append every result in one message, repeat. About 40 lines. |
| **Subagents** | `task` starts a fresh agent with read-only tools and lower effort. Only its final report comes back, so a broad search never floods the main context. |
| **Compaction** | Near the context limit, the whole history becomes one summary. |

## Design notes

These are the decisions worth stealing.

**The prompt prefix never changes mid-session.** The system prompt and tool list are fixed when the agent starts, and both are marked for prompt caching. A long session re-reads them every turn at cache prices. That's also why `remember` writes to `NANO.md` for the *next* session instead of editing the current system prompt: editing it would throw away the cache, and on current Claude models it would also invalidate every thinking block already in the history.

**The agent owns its transcript, not the wire format.** Tool calls, results and thinking are kept in one normalized shape. Each provider adapter translates that to the vendor's streaming API on the way out - the Anthropic adapter sends `tool_use`/`tool_result` blocks, the OpenAI one sends `tool_calls` and `role: "tool"` messages - and normalizes `finish_reason` back, so `max_tokens` and `length` mean the same thing to the loop. Adding a backend is one adapter, not a change to the loop.

**Compaction is "simple compaction".** When the last request was over `--compact-at` tokens (150k by default), the agent asks the model for a structured summary. It then replaces the entire history with that one message, replaying no old turns and no old thinking blocks. Keeping the last few turns verbatim looks smarter but breaks: those turns' thinking was produced alongside history that no longer exists. Compaction only runs between tool rounds, never while a tool call is waiting for its result. A brand-new user request is kept word for word after the summary, not summarised away.

**Tool inputs stream, so they're validated by hand.** Tools set `eager_input_streaming`, so a 400-line `write_file` starts arriving immediately instead of after a long silence. The cost is that the API no longer validates those inputs. `validate_input()` checks every call against its schema before it runs. A call cut off by `max_tokens` is never executed, even when the truncated JSON happens to parse. Bad inputs go back to the model as errors it can fix.

**Parallel where it's safe.** The model can request several tools at once. Read-only ones run in a thread pool. Writes and commands run one at a time, in order, after their permission prompt. All results go back in a single message: splitting them teaches the model to stop calling tools in parallel.

**Failure is information.** Tool errors, refused permissions, invalid input and stale-file edits all come back as `is_error` tool results with a reason the model can act on. Nothing in a tool can crash the session, and Ctrl-C drops the half-finished turn so the history stays valid.

**Refusals fall back.** Requests opt into server-side fallback (`fallbacks="default"`). If a safety classifier declines, the API retries on the model it recommends for that category instead of just stopping.

## Options

```
python nano_agent.py [prompt ...] [--root DIR] [--provider auto|anthropic|openai] [--model MODEL]
                     [--effort low|medium|high|xhigh|max] [--yolo] [--compact-at TOKENS]
```

`--provider auto` (the default) reads `NANO_PROVIDER`, then guesses from the model name: `claude-*` goes to Anthropic, `gpt-*` to the OpenAI-compatible client. `NANO_MODEL` sets the default model. For a self-hosted endpoint, set `OPENAI_BASE_URL` (and `OPENAI_API_KEY` to anything it accepts).

In an interactive session, `/cost` shows tokens and dollars so far, `/compact` compacts now, and `/exit` quits.

The default model is `claude-opus-5-5` on Anthropic and `gpt-5.2` on the OpenAI-compatible client; `--model` overrides either. Subagents run at `medium` effort.

## Tests

```sh
pip install -r requirements-dev.txt
pytest -q
```

The tests drive the real loop against a scripted fake client, so they run offline in about 2 seconds. They check the request shape, the read, edit and fix flow, permissions (decline, always, the never-run list), path escapes, stale and ambiguous edits, invalid and truncated tool input, refusals, parallel result ordering, malformed-JSON retries, compaction (prefix unchanged, nothing replayed, new prompt kept, never mid-round), subagents (read-only tools, fresh context, usage rolled up), memory, and both provider adapters (transcript translation, streamed tool-call fragments, error labelling, finish-reason mapping).

MIT licensed.
