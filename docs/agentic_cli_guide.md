# Spindle Agentic CLI Guide: Zero-Config & Slash Commands Reference

## 1. Quickstart: The Zero-Config All-in-One CLI

Spindle eliminates the need to remember long flags or network addresses. By placing a `spindle.toml` file in your project directory, Spindle automatically discovers your model, connects to your LAN cluster, configures SIMD threads and dynamic MoE pruning, and launches directly into the interactive agent.

### 1.1 Running Spindle
Simply run:
```powershell
.\target\release\spindle.exe
```

### 1.2 `spindle.toml` Configuration File
```toml
# Spindle Cluster & Agent Configuration
model = "Qwen3-30B-A3B-Q4_K_M.gguf"
peer = "192.168.1.10:42424"
layer_range = "35..48"
wire_format = "fp8"
threads = 4
moe_prune = 0.03
tools = true
think = false
```

Any command-line argument explicitly passed will override the corresponding setting in `spindle.toml`.

---

## 2. Interactive OpenCode-Style Interface

Upon startup, Spindle presents a clean, framed header and prompt:

```text
╭─ spindle ──────────────────────────────────────────────────────────────╮
│  Engine: Spindle v0.1.0 · qwen3moe · 17697 MiB · low-mem
│  Model:  Qwen3-30B-A3B
│  4 threads · 2048-token context window · tools: ON
│  type a prompt or /help for slash commands
╰────────────────────────────────────────────────────────────────────────╯

> 
```

---

## 3. Comprehensive Slash Commands Reference

Type `/help` at any time to view the slash command catalog:

| Command | Category | Description | Example |
|---|---|---|---|
| **`/tools [on\|off]`** | Agentic | Toggle autonomous tool usage (files, web, shell) | `/tools on` |
| **`/heal [cmd]`** | Agentic / Self-Healing | Autonomous test-driven loop (detects errors, fixes code, auto-rollbacks) | `/heal cargo test` |
| **`/branch [save\|revert\|list]`** | Memory / KV-Cache | Zero-cost in-RAM KV-cache branching (rewinds context in 0 ms) | `/branch save feat1` |
| **`/review`** | Code Quality | Semantic AI review of uncommitted git diff against best practices | `/review` |
| **`/undo`** | Agentic | Reverts the last file edit made by the agent | `/undo` |
| **`/diff`** | Agentic / Git | Displays git stat and uncommitted diffs of recent edits | `/diff` |
| **`/git <args>`** | Agentic / Git | Runs native git commands in the workspace | `/git status` |
| **`/plan [on\|off]`** | Agentic | Toggles Architect Mode (demands technical plans first) | `/plan on` |
| **`/init`** | Memory | Scaffolds the `SPINDLE.md` project memory file | `/init` |
| **`/compact`** | Memory | Summarizes conversation to free context and RAM | `/compact` |
| **`/think [on\|off]`** | Reasoning | Toggles chain-of-thought reasoning mode | `/think on` |
| **`/think [show\|hide]`**| Reasoning | Shows full Thought box or collapsed summary badge | `/think hide` |
| **`/system [prompt]`** | Prompting | View or dynamically update system prompt at runtime | `/system You are a Rust pro.` |
| **`/system reset`** | Prompting | Restores the model's default system prompt | `/system reset` |
| **`/moe-prune [f32]`** | Performance | Inspect or tune dynamic expert pruning cutoff on-the-fly | `/moe-prune 0.03` |
| **`/draft-len [n]`** | Performance | Set speculative draft token batch length (1..16) | `/draft-len 2` |
| **`/cluster`** | Network | Inspect distributed mesh partition, wire format & ping | `/cluster` |
| **`/model`** | Diagnostics | View architecture, layer counts, resident RAM, KV-cache | `/model` |
| **`/bench [tokens]`** | Benchmark | Run on-demand generation benchmark (default 32) | `/bench 64` |
| **`/clear`** | State | Reset KV-cache and start a fresh conversation | `/clear` |
| **`/stats`** | Diagnostics | Toggle detailed memory and per-layer MoE metrics | `/stats` |
| **`/max <n>`** | State | Set maximum response token generation limit | `/max 1024` |
| **`/help`** | Help | Display available slash commands reference | `/help` |
| **`/exit`** | Session | Quit interactive session (`quit`, `exit`, Ctrl+C) | `/exit` |

---

## 4. Practical Agentic Workflows

### 4.1 Feature Implementation with Architect Planning
1. Enable planning mode:
   ```text
   > /plan on
   [*] Architect Planning mode: ON (model will produce plans before edits)
   ```
2. Request a new feature:
   ```text
   > Add an LRU cache eviction policy to expert_cache.rs
   ```
3. Qwen outlines the data structures, mutex locks, and test cases without touching code.
4. Once you approve, disable planning mode:
   ```text
   > /plan off
   > Implement the plan now.
   ```
5. Qwen calls `read_file`, `edit_file`, and `run_command("cargo test --lib expert_cache")`.

### 4.2 Rolling Back Unwanted Changes
If the agent makes an incorrect change to a file:
```text
> /undo
[OK] Restored 'src/expert_cache.rs' to previous state.
```

### 4.3 Autonomous Test-Driven Self-Healing (`/heal`)
When tests break or compiler errors occur, let Spindle repair the code autonomously:
```text
> /heal
[*] Validating project via 'cargo test'...
[FAIL] Build/test failure detected (exit code 101). Initiating autonomous self-healing...

[*] Self-Healing Pass 1/3:
┌ Tool Call ────────────────────────────────────────────────────────
│  Action: search_symbols
│  searching symbols: 'CachePolicy' (type: Some("enum"))
└ Result: Success

┌ Tool Call ────────────────────────────────────────────────────────
│  Action: edit_file
│  editing: src/expert_cache.rs
└ Result: Success

[*] Re-validating via 'cargo test'...
[OK] Self-healing SUCCESSFUL on pass 1! All tests now pass cleanly.
```
If errors cannot be resolved after 3 attempts, Spindle prompts to automatically roll back all file modifications and restores the exact KV-cache state.

### 4.4 Zero-Cost In-RAM KV-Cache Branching (`/branch`)
Unlike cloud API wrappers that must re-send tokens across the internet, Spindle's KV-cache lives directly in local RAM. You can snapshot and rewind branches instantly in 0 milliseconds:
```text
> /branch save clean_baseline
[OK] Saved KV-cache branch 'clean_baseline' at token pos 842.

> Try exploring a radically different SIMD algorithm for quantization
... (exploration fails or is unsatisfactory) ...

> /branch revert clean_baseline
[OK] Reverted KV-cache to branch 'clean_baseline' (pos: 842 tokens) in 0 ms.
```

### 4.5 Semantic AI Review (`/review`)
Inspect uncommitted changes before staging or committing:
```text
> /review
[*] Analyzing uncommitted changes via git diff...
... (Spindle outputs a structured, high-level code review evaluating correctness, performance, and security) ...
```

### 4.6 Reviewing Code Changes
Check exactly what files have been altered during the session:
```text
> /diff
[*] Checking uncommitted git diff...
 src/expert_cache.rs | 18 ++++++++++++++++++
 1 file changed, 18 insertions(+)
```

