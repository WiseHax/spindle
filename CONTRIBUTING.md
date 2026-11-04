# Contributing

Help is welcome, including small fixes and questions.

## Setup

1. Windows x86-64 with an AVX2 + FMA CPU, and Rust stable.
2. `cargo build --release`
3. Download a small GGUF such as Qwen3-0.6B (see the README) and check that
   `spindle <model.gguf>` starts and answers.
4. `cargo test --release -- --test-threads 1` (several tests load real models; in parallel they run out of RAM)

Don't commit model files. `*.gguf` and `*.bin` are ignored for that reason.

## What to work on

[docs/roadmap.md](docs/roadmap.md) has the open items. Open an issue before starting on
anything big so two people don't do the same thing.

## Before you open a PR

- New or changed kernels need a test against a plain scalar version.
- With `--temp 0` the output is deterministic. If your change is only supposed to make
  things faster, the output for a fixed prompt shouldn't change.
- For speed changes, post before/after numbers from the stats line after a reply (or
  `/bench`), with the model, thread count and CPU. Run a few times, because laptop clocks are
  noisy. [docs/benchmarks.md](docs/benchmarks.md) shows the method.
- Ask before adding a dependency. The only one today is the `windows` crate.
- Cluster frames are only ever added, never changed (see [docs/cluster.md](docs/cluster.md)).
- If you use `unsafe`, say in a comment why it's sound.
- Run `cargo fmt` and `cargo clippy`.

Keep PRs to one topic.

## Bug reports

Include the command, the full output (`--verbose` helps), the model file, your CPU, RAM,
Windows version and `spindle --version`.
