# Spindle: Distributed MoE & Agentic Inference Engine

## Project Context
- Pure-Rust from-scratch distributed LLM inference engine optimized for low-RAM laptops and desktop clusters.
- Architecture: MoE with 128 experts (8 active Top-K), FP8/FP16 network transport, AVX2 SIMD acceleration.
- Current active cluster configuration: Worker Desktop (0..35) + Local Client Laptop (35..48).

## Build & Test Commands
- Build release: `cargo build --release`
- Run test suite: `cargo test --release`
- Code validation: `cargo check`

## Available Agent Slash Commands
- `/tools [on|off]`: Toggle autonomous tool calling (files, web, shell)
- `/heal [test_cmd]`: Autonomous test-driven compile & heal loop (auto-rollback on failure)
- `/branch [save|revert|list]`: In-RAM zero-cost KV-cache hypothesis branching
- `/review`: Semantic AI review of uncommitted git diff
- `/undo`: Revert the last file change made by the agent
- `/diff`: View git diff stat and uncommitted code changes
- `/git <args>`: Execute native git commands
- `/plan [on|off]`: Toggle Architect planning mode before edits
- `/init`: Scaffold or refresh project memory file
- `/compact`: Compact conversation context into a summary
- `/think [on|off|show|hide]`: Toggle reasoning & thought box visibility
- `/system [prompt|reset]`: Inspect or dynamically update system prompt
- `/moe-prune [threshold]`: Inspect or tune expert energy cutoff at runtime
- `/draft-len [n]`: Set speculative decoding draft token batch length (1..16)
- `/cluster`: View distributed mesh node status & network ping RTT
- `/model`: Inspect model tensors, layer range, RAM & KV-cache allocation
- `/bench [n]`: Run on-demand generation benchmark
- `/clear`: Clear KV-cache and start fresh conversation
- `/stats`: Toggle detailed memory and MoE execution metrics
- `/max <n>`: Set maximum response token limit
- `/help`: Display command reference
- `/exit`: Quit session

## Agent Tools Inventory
- `read_file`: Line-numbered source inspection with start_line and limit
- `write_file`: File creation with automatic parent folder generation and shadow undo
- `edit_file`: Exact surgical search-and-replace
- `list_dir`: Directory listing with human-readable file sizes
- `search_symbols`: Fast microsecond AST/symbol scanner (`fn`, `struct`, `class`, `trait`, etc.)
- `search_web`: Zero-dependency DuckDuckGo web search via native `curl.exe`
- `run_command`: Subprocess execution governed by AntiSabotage Guard with 30s watchdog

## Agentic Development Guidelines
- Maintain zero external third-party crate dependencies for core runtime simplicity.
- Always preserve anti-host sabotage safety guardrails in `src/guard.rs`.
- Do not make breaking changes to network wire protocol frames in `src/cluster_net.rs`.

