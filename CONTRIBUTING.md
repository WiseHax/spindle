# Contributing

Help is welcome, including small fixes and questions.

## Setup

1. Windows x86-64 with an AVX2 + FMA CPU, and Rust stable.
2. `cargo build --release`
3. Download a SmolLM2 GGUF (see the README) and check that `--chat` works.
4. `cargo test --release`

Don't commit model files. `*.gguf` and `*.bin` are ignored for that reason.

## What to work on

[docs/roadmap.md](docs/roadmap.md) has the open items. Open an issue before starting on
anything big so two people don't do the same thing.

## Before you open a PR

- New or changed kernels need a test against a plain scalar version.
- With `--temp 0` the output is deterministic. If your change is only supposed to make
  things faster, the output for a fixed prompt shouldn't change.
- For speed changes, post before/after `[chat:stats]` numbers with the model, thread count
  and CPU. Run a few times, laptop clocks are noisy.
- Ask before adding a dependency.
- If you use `unsafe`, say in a comment why it's sound.
- Run `cargo fmt` and `cargo clippy`.

Keep PRs to one topic.

## Bug reports

Include the command, the full output, the model file, your CPU, RAM and Windows version.
