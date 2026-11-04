# Project notes for spindle

Spindle is a CPU inference engine for GGUF models in Rust. It runs on one Windows PC, or
across two (the worker runs layers 0..35, the client runs 35..48).

## Build and test

- Build: `cargo build --release`
- Test: `cargo test --release -- --test-threads 1` (some tests load real models and need the RAM)
- Check: `cargo check`

## Rules

- No new dependencies without asking. The only one is the `windows` crate.
- Never change an existing cluster frame in `src/cluster_net.rs`; add a new frame type.
- Keep the command and path checks in `src/guard.rs` when changing tools.
- New kernels need a test against the scalar reference in `src/quant.rs`.
