# Spindle Agent Architecture: Pure-Rust Local Coding Assistant

## 1. Executive Summary

Spindle is not merely a high-performance distributed inference engine; it features a native, state-of-the-art **Agentic Coding Assistant** inspired by tools such as Claude Code, Antigravity, and Aider.

Unlike cloud-dependent assistants that require API subscriptions and expose private proprietary code to remote servers, Spindle executes **100% locally and across private LAN clusters** using pure Rust. The model (such as `Qwen3-30B-A3B`) runs with full tool-calling capabilities while the Spindle host runtime mediates file operations, shell execution, web searches, and safety guardrails.

---

## 2. The ReAct (Reason + Act) Execution Pipeline

Modern language models do not have native access to disk controllers, network adapters, or shell interpreters. Spindle provides the deterministic environment bridge via a streaming tool-interception pipeline:

```
+-------------------------------------------------------------------------+
|                              Spindle CLI                                |
|                                                                         |
|  [User Prompt]                                                          |
|       |                                                                 |
|       v                                                                 |
|  +-------------------------------------------------------------------+  |
|  |                   SmolLm2Engine (LLM Inference)                   |  |
|  |   - Token stream generated via AVX2 GEMV / Distributed Cluster   |  |
|  +-------------------------------------------------------------------+  |
|       |                                                                 |
|       | (Streams tokens)                                                |
|       v                                                                 |
|  +-------------------------------------------------------------------+  |
|  |                 Stream Interceptor (src/chat_cli.rs)              |  |
|  |   - Detects `<tool_call>` opening tag                             |  |
|  |   - Halts generation and extracts structured JSON payload         |  |
|  +-------------------------------------------------------------------+  |
|       |                                                                 |
|       | ToolCall { name, path, content, command, query }                |
|       v                                                                 |
|  +-------------------------------------------------------------------+  |
|  |               Anti-Host Sabotage Guard (src/guard.rs)             |  |
|  |   - Evaluates command safety & path boundaries                    |  |
|  |   - Enforces user confirmation [Y/n] for high-impact actions      |  |
|  +-------------------------------------------------------------------+  |
|       |                                                                 |
|       | (Approved)                                                      |
|       v                                                                 |
|  +-------------------------------------------------------------------+  |
|  |                   Tool Dispatcher (src/tools.rs)                  |  |
|  |   - `read_file`: Line-numbered source inspection                  |  |
|  |   - `write_file`: Creation & shadow undo snapshot                 |  |
|  |   - `edit_file`: Exact surgical replacement                       |  |
|  |   - `list_dir`: Directory tree inspection with file sizes         |  |
|  |   - `search_web`: Native zero-dependency DuckDuckGo search        |  |
|  |   - `run_command`: Subprocess with 30s watchdog                   |  |
|  +-------------------------------------------------------------------+  |
|       |                                                                 |
|       | Tool Result String                                              |
|       v                                                                 |
|  +-------------------------------------------------------------------+  |
|  |           Feedback Injection (`<tool_response> ... </tool_response>`) |
|  |   - Result fed back into KV-cache as context for next reasoning turn |  |
|  +-------------------------------------------------------------------+  |
|       |                                                                 |
|       v                                                                 |
|  [LLM completes final answer to user or executes subsequent tool call]   |
+-------------------------------------------------------------------------+
```

---

## 3. Tool Calling Protocol & Schema

Spindle natively implements the ChatML tool calling format fine-tuned into Qwen 2.5 and Qwen 3 models.

### 3.1 Tool Schema Injection
When agentic mode is enabled (`--tools` flag or `/tools on`), Spindle injects the tool definitions into the model's system prompt:

```text
# Tools
You have access to the following tools to assist the user:

## `read_file`
Description: Read the contents of a file with line numbers.
Parameters: {"type": "object", "properties": {"path": {"type": "string"}, "start_line": {"type": "integer"}, "limit": {"type": "integer"}}, "required": ["path"]}

## `write_file`
Description: Create or overwrite a file with given content.
Parameters: {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}

## `edit_file`
Description: Perform exact search-and-replace edit in a file.
Parameters: {"type": "object", "properties": {"path": {"type": "string"}, "target": {"type": "string"}, "replacement": {"type": "string"}}, "required": ["path", "target", "replacement"]}

## `list_dir`
Description: List files and subdirectories in a folder.
Parameters: {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}

## `run_command`
Description: Run a shell command in the project directory.
Parameters: {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}

## `search_web`
Description: Search the internet online for live information.
Parameters: {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
```

### 3.2 Tool Invocation Grammar
The model invokes a tool by outputting:
```text
<tool_call>
{"name": "read_file", "arguments": {"path": "src/main.rs", "start_line": 1, "limit": 50}}
</tool_call>
```

### 3.3 Host Response Injection
Spindle executes the tool and injects the output back into the conversation turn:
```text
<tool_response>
--- src/main.rs (showing lines 1-50 of 1585) ---
   1 | //! spindle: CLI entry point.
   2 | ...
</tool_response>
```

---

## 4. Built-in Tool Specifications

### 4.1 `read_file`
- **Purpose:** Inspects source code files with 1-based line numbers.
- **Safety:** Checks `guard::check_path` to prevent reading sensitive host credentials (e.g., `~/.ssh/id_rsa`, Windows SAM database).
- **Features:** Supports `start_line` and `limit` to prevent context exhaustion on large files.

### 4.2 `write_file` & `edit_file`
- **Purpose:** Create new files or apply surgical search-and-replace edits.
- **Automatic Parent Creation:** Automatically creates parent directories (`std::fs::create_dir_all`) if missing.
- **Shadow Undo Tracking:** Automatically records a pre-edit snapshot onto the in-memory undo stack before writing to disk, enabling instant `/undo` rollbacks.

### 4.3 `list_dir`
- **Purpose:** Inspects project directory structures.
- **Output:** Formats directories with `[DIR]` and files with human-readable sizes (B, KiB, MiB).

### 4.4 `search_symbols` (Fast Workspace Symbol Intelligence)
- **Purpose:** Scans workspace source files (`.rs`, `.py`, `.js`, `.ts`, `.go`, `.c`, `.cpp`, `.toml`) for symbol declarations (`fn`, `struct`, `enum`, `trait`, `impl`, `class`, `def`, `interface`).
- **Context Economy:** Solves the token-exhaustion bottleneck of local models. Instead of loading 500-line files, `search_symbols` returns precise file paths and line numbers in microseconds using under 50 tokens of context.
- **Filtering:** Supports optional `symbol_type` filters (`fn`, `struct`, `class`, `trait`, etc.).

### 4.5 `search_web` (Zero-Dependency Online Search)
- **Design Philosophy:** Avoids heavy third-party dependencies (`reqwest`, `tokio`, `openssl`).
- **Implementation:** Leverages Windows native `curl.exe` with built-in Schannel SSL/TLS to query DuckDuckGo HTML search.
- **Parsing:** Extracts result titles, snippets, and clean markdown links without complex DOM parsers.

### 4.6 `run_command`
- **Execution:** Spawns commands via `powershell.exe -NoProfile -NonInteractive -Command "..."`.
- **Watchdog Protection:** Enforces a strict 30-second execution deadline to eliminate runaway processes or fork-bombs.
- **Buffer Safety:** Caps output to 32 KiB to prevent blowing the KV-cache.

---

## 5. Original Innovations Unique to Spindle

Traditional coding agents (Claude Code, Cursor, Aider, Devin) operate as external API wrappers. Because they communicate over HTTP to remote proprietary endpoints, they have zero control over model weights, attention matrix layout, or the KV-cache. Spindle, by contrast, owns the inference engine and memory layout down to the hardware level:

### 5.1 In-RAM Zero-Cost KV-Cache Branching (`/branch`)
- **Mechanism:** In `SmolLm2Engine`, keys and values are stored in contiguous `k_cache` and `v_cache` vectors indexed by `(layer * max_seq + pos) * kv_dim`.
- **Zero Latency (0 ms):** Snapshotting a branch (`save_kv_branch`) simply captures the token position pointer `cur_pos` and draft engine state.
- **Zero Context Poisoning:** When exploring an uncertain code modification, if the attempt fails, `restore_kv_branch` rewinds `cur_pos` instantly. The model's context is completely cleansed of failed attempts with **zero prefill recalculation and zero network roundtrips**.

### 5.2 Autonomous Test-Driven Self-Healing (`/heal`)
- **Closed-Loop Execution:** Executes compiler or test runners (`cargo test`, `cargo check`, `pytest`, `npm test`), captures compiler diagnostics (`error[E....]`), and extracts the offending files and lines.
- **Multi-Pass Iteration:** Automatically prompts the model with the exact compiler diagnostics across up to 3 passes, letting the agent use `search_symbols` and `edit_file` to fix the bug.
- **Safety Rollback:** If the error cannot be resolved within 3 attempts, Spindle prompts to automatically revert all file edits and rewinds the KV-cache to ensure the repository remains pristine.

---

## 6. Architectural Comparison Matrix

| Capability | Spindle (Native Pure-Rust) | Claude Code / Devin | Cursor / Copilot | Aider |
|---|---|---|---|---|
| **Privacy & Sovereignty** | **100% Local / LAN Mesh** | Cloud API (Proprietary) | Cloud API (Proprietary) | Cloud API / Local Ollama |
| **KV-Cache Control** | **Direct In-RAM Rewind (0ms)** | No (Re-sent over HTTP) | No (Hidden behind API) | No (Full re-prefill) |
| **Distributed Inference** | **Wi-Fi / LAN Pipeline Mesh** | N/A (Server-only) | N/A (Server-only) | N/A |
| **Anti-Host Sabotage Guard** | **5-Tier Hard-Block & Gate** | Basic confirmation | IDE sandbox | None (Direct shell) |
| **Autonomous Self-Healing** | **Built-in `/heal` + Rollback** | Manual prompting | Manual chat | Git-based loop |
| **Context-Aware Symbol Search** | **Microsecond AST Scanner** | Grep / External indexing | Closed IDE Indexer | Ripgrep / Ctags |
| **Third-Party Dependencies** | **0 External Crates** | Node.js runtime | Electron / IDE | Python / Libuv / Pip |

---

## 7. Multi-Step Autonomous Reasoning

Spindle supports chaining up to 5 sequential tool operations within a single user request. For example:
1. User requests: *"Inspect the tests in `src/guard.rs`, run cargo test on them, and summarize."*
2. **Step 1:** Model calls `read_file("src/guard.rs")`.
3. **Step 2:** Spindle injects the file contents.
4. **Step 3:** Model analyzes the code and calls `run_command("cargo test --lib guard")`.
5. **Step 4:** Spindle executes the test and injects `Exit code: 0, 5 passed`.
6. **Step 5:** Model outputs the final synthesized response to the user.

