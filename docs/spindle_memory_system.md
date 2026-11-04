# Spindle Memory & Context System: Persistent Knowledge & Instant Undo

## 1. Overview

Agentic coding workflows require two fundamentally different types of memory:
1. **Persistent Project Memory (`SPINDLE.md`):** Long-term guidelines, architectural decisions, and build commands that should persist across sessions.
2. **Ephemeral Session Memory & Shadow Snapshots:** Conversation KV-cache context and file modification history that enable real-time rollbacks (`/undo`) and context compaction (`/compact`).

---

## 2. Persistent Project Memory (`SPINDLE.md`)

### 2.1 The Concept
Just as Claude Code uses `CLAUDE.md`, Spindle automatically looks for `SPINDLE.md` in the current working directory upon startup. If detected, its entire contents are injected into the active system prompt under:
```text
# Project Instructions (SPINDLE.md)
<contents of SPINDLE.md>
```

### 2.2 Standard `SPINDLE.md` Structure
```markdown
# Spindle Project Instructions

## Overview
Brief description of the repository and its primary technologies.

## Build and Test Commands
- Build: `cargo build --release`
- Tests: `cargo test --release`
- Validation: `cargo check`

## Architecture & Code Guidelines
- Zero-dependency runtime principles.
- Code style, naming conventions, and file organization.
- Critical safety guardrails and system boundaries.
```

### 2.3 Automatic Initialization (`/init`)
Developers can scaffold a `SPINDLE.md` tailored to their repository by typing:
```text
> /init
[OK] Created SPINDLE.md project memory file.
```

---

## 3. Real-Time Shadow Undo Engine (`/undo`)

### 3.1 The Problem with Autonomous AI Edits
When an LLM writes code across multiple files, a minor hallucination or incorrect regex replacement can break a codebase. Traditional Git can revert changes, but it requires leaving the agent session or discarding all uncommitted work.

### 3.2 The Spindle Shadow Undo Architecture
Spindle implements an in-memory shadow snapshot stack (`UNDO_STACK: Mutex<Vec<(String, Option<String>)>>`):
1. **Pre-Edit Snapshot:** Before `tool_write_file` or `tool_edit_file` touches a file on disk, Spindle reads the file's current state.
   - If the file exists, it records `(path, Some(previous_content))`.
   - If the file is newly created, it records `(path, None)`.
2. **Atomic Rollback:** When the user types `/undo`:
   - If the file existed previously, Spindle restores the exact previous content.
   - If the file was newly created by the agent, Spindle deletes the file.
3. **Thread-Safe Keying:** `pop_undo_path` enables path-isolated rollbacks even during parallel test execution.

```text
User Request -> Agent calls edit_file("src/main.rs")
                     |
                     v
         [Record Shadow Snapshot] -> UNDO_STACK.push("src/main.rs", previous_content)
                     |
                     v
              [Write to Disk]
                     |
(If user types `/undo`)
                     v
       [Pop Snapshot & Revert File on Disk] -> "Restored src/main.rs to previous state."
```

---

## 4. Context Compaction (`/compact`)

### 4.1 The Memory Wall in Extended Sessions
Long pair-programming sessions can exhaust the model's KV-cache context window (especially on low-RAM laptops where context is clamped to ~2048 tokens).

### 4.2 The Compaction Algorithm
Typing `/compact` executes an autonomous memory compression routine:
1. Spindle instructs the LLM:
   `"Please provide a concise 3-sentence summary of the key tasks, accomplishments, and decisions made in our conversation so far."`
2. The model synthesizes its own conversation history into a dense summary.
3. Spindle resets the KV-cache (`engine.reset_kv()`), wiping raw token history.
4. Spindle feeds the summary back into the engine as a system memory context:
   `<|im_start|>system\nConversation summary:\n{summary}\n<|im_end|>\n`
5. The session continues with full situational awareness and a completely refreshed context window!

