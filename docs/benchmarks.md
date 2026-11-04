# Benchmarks

Every number here was measured, and each section says how. Laptop clocks are noisy, so
expect about ±10% between runs. Free RAM matters a lot on an 8 GB machine. A model that
fits in RAM on one run can stream from disk on the next, if a browser was open in between.

## Machines

| | Laptop (dev machine, client) | Desktop (worker) |
|---|---|---|
| CPU | Intel i5-13420H, 4 P-cores + 4 E-cores | x86-64, 6 cores / 12 threads |
| RAM | 8 GB | 16 GB |
| OS | Windows 11 | Windows |
| Network | home Wi-Fi | home Wi-Fi |

`spindle --probe` on the laptop measures about 31 GB/s from RAM with 4 or more threads,
2.2–2.8 GB/s for large SSD reads, and 0.03–0.16 GB/s for random 4 KiB SSD reads.

## Single PC

spindle 1.0.0, laptop, `--temp 0 -n 128`, prompt "Write a short paragraph about how rivers
shape valleys.", 4 threads. Each model was loaded fully into RAM, except where noted.

| Model | File | Speculation off | Speculation on |
|---|---|---|---|
| SmolLM2-135M-Instruct | 101 MiB, Q4_K_M | 170.7 tok/s | 191.0 tok/s |
| SmolLM2-360M-Instruct | 368 MiB, Q8_0 | 56.6 tok/s | 48.4 tok/s |
| Qwen3-0.6B | 609 MiB, Q8_0 | 39.1 tok/s | 33.6 tok/s |
| Granite 3.1 1B-A400M | 1.3 GiB, Q8_0 | 42.2 tok/s | 13.2 tok/s* |

\* In that run free RAM was too low to hold the model, so it streamed from disk. It is not
comparable with the column next to it.

This prompt asks for new prose, so few drafts match: 0 of 11 for SmolLM2-360M, for example.
The "on" column shows what a rejected draft costs. When the reply copies text from the
context, speculation pays off instead:

| Qwen3-0.6B, copy task | tok/s |
|---|---|
| speculation off | 14.7 |
| speculation on | 32.5 (same reply) |

To reproduce, run from a folder without a `spindle.toml`:

```powershell
.\spindle.exe qwen3-0.6b-q8_0.gguf --prompt "Write a short paragraph about how rivers shape valleys." --temp 0 -n 128
.\spindle.exe qwen3-0.6b-q8_0.gguf --prompt "Write a short paragraph about how rivers shape valleys." --temp 0 -n 128 --no-spec
```

The last line of output is the speed. `/bench` in a chat does the same with a fixed prompt.

## Two PCs: Qwen3-30B-A3B

`Qwen3-30B-A3B-Q4_K_M.gguf` (17.3 GiB, 48 layers, 128 experts, 8 routed per token):

- desktop: layers 0..35, 5 threads
- laptop: layers 35..48 and the LM head, 4 threads
- `wire_format = fp8`, `moe_prune = 4`, speculation on, `ctx = 8192`

Three replies in one session, spindle 1.0.0:

| Reply | tok/s | Worker per token | Laptop per token |
|---|---|---|---|
| 28 tokens, 3.4 s | 8.2 | 73 ms | 31 ms |
| 260 tokens, 31.6 s | 8.2 | 71 ms | 30 ms |
| 320 tokens, 44.2 s | 7.2 | 76 ms | 40 ms |

"Worker per token" is the time from sending a request until its activations come back, so it
includes the network. "Laptop per token" is the local layers and the LM head. The two add up
to about 104 ms, or about 9.6 tok/s. The rest goes to sampling, detokenizing and drawing the
interface. Neither machine paged to disk.

Earlier versions on the same hardware, for context. These come from the development log and
were measured with different settings:

| Version | Change | tok/s |
|---|---|---|
| one laptop, streaming from SSD | model doesn't fit | 0.1–0.2 |
| first two-PC run | single-threaded worker | 1.1 |
| | multi-threaded worker | 3.1 |
| | 35/13 layer split | 3.5 |
| | fp8 activations | 5.1–6.0 |
| | 2-row Q6_K kernel, no allocations in the MoE router | 6.3–7.2 |
| 1.0.0 | multi-vector kernels, `moe_prune = 4`, speculation | 7.2–8.2 |

## Kernels

`cargo test --release bench_multi_vs_serial -- --ignored --nocapture` measures the cost of a batch of 5
vectors against a single one through the multi-vector kernels. These are the kernels behind
prompt processing and speculative verification. Single thread, laptop:

| Weights | 5 vectors vs 1 |
|---|---|
| Q4_K | 1.78x |
| Q6_K | 1.36x |
| Q6_K LM head | 1.37x |
| Q8_0 | 1.61x |

Before these kernels, a batch of 5 cost 5x.

## Not measured yet

- The output quality of MoE pruning (perplexity, or task scores).
- Prompt processing speed, separately from generation.
- The same models in llama.cpp on the same machines. `tools/golden.ps1` checks that the
  output agrees with llama.cpp, not the speed.
