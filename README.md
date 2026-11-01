# spindle

LLM inference written from scratch in Rust, aimed at cheap laptops.

I built this on an IdeaPad with an i5-13420H and 8 GB of RAM, where most local AI tools
either crawl or take the whole machine down with them. spindle doesn't use llama.cpp, BLAS
or any ML framework. The GGUF parser, tokenizer, transformer, AVX2 kernels and thread pool
are all in this repo.

It's early. It runs SmolLM2 135M and 360M today, and has code paths for Qwen2/Qwen3 and
mixture-of-experts models that pass unit tests but haven't been run on a real Qwen or MoE
file yet. The goal is to run models bigger than RAM
by streaming weights from the NVMe drive, mostly MoE models. That part isn't done yet; see
[docs/roadmap.md](docs/roadmap.md).

## What works

- GGUF v3 parsing (metadata, tensors, vocab, BPE merges)
- Llama-style models: RMSNorm, RoPE, grouped-query attention, SwiGLU
- Qwen2/Qwen3 attention details: QKV bias, per-head Q/K norm, NEOX RoPE, `head_dim`
  different from `dim / heads` (unit-tested, not yet run on a real Qwen file)
- MoE layers: softmax router, top-k experts, optional shared expert, with per-turn stats on
  how many MiB of experts each token reads (unit-tested, not yet run on a real MoE file)
- Byte-level BPE, so non-ASCII text (ñ, emoji, CJK) round-trips instead of being dropped
- Model sizes read from GGUF metadata, nothing hardcoded
- AVX2/FMA kernels for `F32`, `F16`, `BF16`, `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`,
  `Q4_K`, `Q5_K`, `Q6_K`, each tested against a scalar reference
- Matvecs split across the P-cores with a persistent thread pool
- Weights memory-mapped, not copied
- Memory safety net for models bigger than RAM: the KV-cache is sized to free memory at
  startup, a governor caps spindle's working set so other apps keep their RAM, and weights
  that are about to be used are prefetched in large reads from a background thread
- Multi-turn chat with a KV-cache that slides when it fills up
- Temperature, top-k, top-p, min-p sampling with a seed
- `--probe` to measure your RAM and SSD bandwidth

Not yet: `Q2_K`, `Q3_K` and the IQ formats, explicit SSD streaming and expert caching,
Linux/macOS, ARM.

## Speed

i5-13420H, 4 P-cores, 8 GB RAM, Windows 11.

| Model | File | 1 thread | 4 threads |
|---|---|---|---|
| SmolLM2-135M-Instruct | Q4_K_M, 105 MB | 60-70 tok/s | 140-195 tok/s |
| SmolLM2-360M-Instruct | Q8_0, 386 MB | ~19 tok/s | 30-55 tok/s |

Every token has to read all the weights once, so RAM bandwidth sets the ceiling. On this
laptop `--probe` measures about 31 GB/s, which puts 360M's ceiling around 80 tok/s.

## Build

You need Windows 10/11 on x86-64, a CPU with AVX2, FMA and F16C (Haswell / Zen or newer), and
[Rust](https://rustup.rs).

```powershell
git clone https://github.com/WiseHax/spindle.git
cd spindle
cargo build --release
```

Get a model:

```powershell
Invoke-WebRequest "https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct-GGUF/resolve/main/smollm2-360m-instruct-q8_0.gguf" -OutFile smollm2-360m-q8_0.gguf
```

## Usage

```powershell
.\target\release\spindle.exe smollm2-360m-q8_0.gguf --chat
```

In the chat, type a message and press Enter. `/clear` forgets the conversation, `exit`
quits.

| Option | Default | |
|---|---|---|
| `--prompt "..."` | | send a first message right away |
| `--temp` | 0.6 | 0 means greedy |
| `--top-k` | 40 | |
| `--top-p` | 0.9 | |
| `--min-p` | 0.05 | |
| `--repeat-penalty` | 1.1 | |
| `--seed` | random | printed at startup, reuse it to get the same replies |
| `--probe` | | measure RAM/SSD speed instead of chatting |
| `--reserve` | 1024 | MiB of RAM to always leave free for other apps |
| `--ram-budget` | | hard cap on spindle's resident memory in MiB |
| `$env:CHAT_THREADS` | 4 | |
| `$env:CHAT_CTX` | 1024 | KV-cache length in tokens |

For honest SSD numbers, run `--probe` against a file of 1 GB or more.

`$env:SPINDLE_FORCE_STREAM = "1"` makes spindle treat any model as too big for RAM. Useful for
testing the low-memory path with a small model.

## Layout

| | |
|---|---|
| `src/smollm_chat.rs` | the engine: forward pass (dense and MoE), tokenizer, sampler, chat |
| `src/quant.rs` | quant formats, AVX2 kernels, scalar reference, kernel tests |
| `src/gguf_loader.rs` | GGUF parser |
| `src/worker_pool.rs` | thread pool used for matvecs |
| `src/mmap_file.rs` | memory-mapped model files, background prefetcher |
| `src/memory_guard.rs` | working-set governor |
| `src/probe.rs` | bandwidth probe |
| `src/main.rs` | CLI and CPU topology (P-core / E-core detection) |
| `src/q4k_gemv.rs`, `transformer.rs`, `moe_engine.rs`, `rio_mesh.rs` | older experiments behind the synthetic benchmark, not used by chat |

More detail in [docs/architecture.md](docs/architecture.md).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT. Model weights aren't included and have their own licenses (SmolLM2 is Apache-2.0).
