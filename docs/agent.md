# The coding agent

With tools on, the model can work on the project in the current folder. It can read and
search files, edit them, run commands and look things up on the web. Tools are on by default
in chat and controlled by `tools` in `spindle.toml`, `--tools`, or `/tools on|off`.

## How a turn works

1. Your message goes to the model together with the tool descriptions. These are in the
   system prompt, in Qwen's own function-calling format (`<tools>` ... `</tools>`).
2. The model replies. Instead of answering, it can emit one or more calls:

   ```
   <tool_call>
   {"name": "read_file", "arguments": {"path": "src/main.rs", "offset": 120, "limit": 60}}
   </tool_call>
   ```

3. Spindle parses each call with a lenient JSON parser. The parser accepts the usual
   small-model slips: single quotes, trailing commas, `True`/`None`, raw newlines in strings.
   Spindle then checks the call against the guard (see [security.md](security.md)), asks
   you if needed, and runs it.
4. The results go back to the model as `<tool_response>` blocks, and it continues.
5. This repeats until the model answers without a tool call, for at most 24 rounds per
   message.

Tool output is cut down to what fits in the remaining context. Long outputs keep their
beginning and end. If the model repeats the exact same call it has already made twice, it
gets an error instead, which breaks most loops.

The interface shows each call as a row (`▸ Read src/main.rs`, `⌕ Grep "fn main"`). Edits show
a diff and commands show their output. The raw `<tool_call>` text is never printed, even
when a small model spells the tags out token by token.

## Tools

| Tool | What it does | Asks first |
|---|---|---|
| `read_file` | numbered lines, 250 by default; `offset`/`limit` for long files | no |
| `list_dir` | files and folders in a directory | no |
| `glob` | find files by pattern, e.g. `**/*.rs` | no |
| `grep` | search file contents, optionally filtered by file name | no |
| `write_file` | create or overwrite a file | outside the project folder |
| `edit_file` | replace an exact string, which must be unique unless `replace_all`; works with CRLF files and suggests near matches when the string isn't found | outside the project folder |
| `delete_file` | delete a file | always |
| `run_command` | run a PowerShell command in the project folder (60 s default timeout, 600 s max) | for risky commands |
| `code_map` | outline of the project's source files (functions, types, line numbers), ranked by optional focus words | no |
| `recall` | show more of a shortened tool result: a line range, or the lines around a search text | no |
| `web_search` | search the web and return titles and snippets | no |
| `fetch_url` | download a page and return its text | no |

Every write, edit and delete first saves what was there before, so `/undo` can put it back.
For a file the agent created, `/undo` removes it. `/rewind` goes further: it undoes the whole
last turn, every file it changed and the conversation itself, and puts the turn's prompt back
in the input box. The conversation goes back without recomputing anything, because the KV
cache from before the turn is still in memory; only the position moves. If the context window
has slid since that turn, only the files are restored.

## Made for slow hardware

On a CPU the model has to process every token of tool output before it can answer, and on two
PCs that runs at about 25 tok/s. So the agent treats tool output as a budget measured in
seconds:

- **Budget.** One round of tool results gets at most what this machine processes in about 30
  seconds, measured from the session's own prompt speed (`SPINDLE_TOOL_SECS=n` changes it), and
  at most a third of the free context. On a fast small model that is the old limit; on the
  two-PC setup it is about 750 tokens.
- **Vault.** Output that doesn't fit is not thrown away. The model gets a digest and a note like
  `saved as v3`, and `recall {"id": "v3", "lines": "120-180"}` or
  `recall {"id": "v3", "find": "panicked"}` brings back any part of it. The vault holds up to
  16 MB per session.
- **Digests that keep the signal.** For command output: the first lines, every line with an
  error, warning, failure, panic, traceback or `file:line` location (plus one line around
  each), and as much of the end as fits, with `…` where lines were dropped. A 600-line cargo
  build with one type error comes back as a few dozen lines that still name the error, the file
  and the line.
- **Find before reading.** `code_map` lists files with their declarations and line numbers,
  ranked by focus words, and a long source file opened from the top starts with its outline.
  Both let the model read 40 lines at the right offset instead of 4,000 from the top.
- **Context note.** Once the context is 40% full, each round of results ends with a line saying
  how much is used.

## Project notes

If the folder has a `SPINDLE.md` (or `AGENTS.md`), its contents go into the system prompt.
Use it for what the model should know about the project: what it is, how to build and test
it, the code layout and conventions. `/init` asks the model to look around the project and
write one for you.

## Commands

The full-screen interface (default) and the line-based one (`--no-tui`) share most commands:

| Command | |
|---|---|
| `/help` | commands and keys |
| `/new` (`/clear`) | start a new conversation |
| `/tools on\|off` | turn tools on or off |
| `/think on\|off\|show\|hide` | reasoning before answers, and whether it is shown |
| `/undo` | revert the last file the agent changed (repeat to go further back) |
| `/rewind` | undo the whole last turn: its file changes and the conversation, and put its prompt back |
| `/diff` | `git diff --stat` of the working tree |
| `/review` | ask the model to review the uncommitted diff |
| `/compact` | replace the conversation with a summary, to free context |
| `/init` | write `SPINDLE.md` / `AGENTS.md` |
| `/model` | model, memory and context details |
| `/cluster` | worker status and round-trip time |
| `/spec on\|off\|<n>` | speculative decoding (full-screen interface) |
| `/moe-prune <x>` | MoE pruning (see [moe_pruning.md](moe_pruning.md)) |
| `/system [text\|reset]` | show or replace the system prompt |
| `/max <n>` | maximum tokens per reply |
| `/bench [n]` | measure generation speed |
| `/exit` | quit (or ctrl+c on an empty prompt) |

The line-based interface also has:

| Command | |
|---|---|
| `/heal [test command]` | run the tests; if they fail, give the output to the model and let it fix the code, up to 3 attempts, then offer to undo everything |
| `/branch save\|revert\|list <name>` | save a point in the conversation and return to it later. Only the position is saved, because the KV cache before that point is unchanged |
| `/plan on\|off` | ask for a plan before any change |
| `/git <args>` | run a git command |
| `/stats` | detailed timing after each reply |
| `/draft-len <n>` | draft tokens per speculative step |

Keys in the full-screen interface:

| Key | |
|---|---|
| enter | send |
| ctrl+j, alt+enter | new line |
| tab | complete a slash command |
| ↑ / ↓ | move between lines of a multi-line prompt; from an empty prompt, browse earlier messages (↓ past the newest brings back what you had typed) |
| ctrl+a / ctrl+e, home / end | start / end of the line |
| ctrl+← / ctrl+→, alt+b / alt+f | previous / next word |
| ctrl+w, alt+backspace | delete the word before the cursor |
| ctrl+u / ctrl+k | delete to the start / end of the line |
| ctrl+z | undo the last edit to the prompt |
| esc | interrupt the reply; when idle, clear the prompt (ctrl+z brings it back) |
| mouse wheel, pgup / pgdn | scroll the conversation |
| ctrl+home / ctrl+end | jump to the top / bottom |
| ctrl+l | redraw the screen |
| ctrl+c | quit when the prompt is empty, otherwise clear it |
| shift+drag | select text with the mouse |

While you are scrolled up, new text arriving at the bottom doesn't move what you are
reading.

You can type while a reply is being generated. The message is queued and sent when the
reply ends. Pasted text keeps its line breaks.

## Choosing a model for agent work

Tool use needs a model that was trained for it. Qwen3 models handle the `<tool_call>` format
natively. Qwen3-30B-A3B is reliable at multi-step edits, and Qwen3-0.6B can do simple reads
and searches but often loses track over several steps. SmolLM2 and Granite will chat, but
they rarely produce usable tool calls. Turn tools off for them (`/tools off`) to get a
shorter system prompt and a faster start.
