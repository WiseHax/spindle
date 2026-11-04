# Spindle Project Guidelines

## Project Context
- Pure-Rust distributed LLM inference engine running across Desktop Worker (0..35) and Laptop Client (35..48).

## Build & Test Commands
- Build: `cargo build --release`
- Tests: `cargo test --release`
- Check: `cargo check`

## Development Guidelines
- Maintain zero external third-party crate dependencies for core runtime simplicity.
- Always preserve anti-host sabotage safety guardrails in `src/guard.rs`.
- Do not make breaking changes to network wire protocol frames in `src/cluster_net.rs`.
