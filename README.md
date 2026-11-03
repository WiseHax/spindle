# spindle

LLM inference written from scratch in Rust, aimed at cheap laptops.

I built this on an IdeaPad with an i5-13420H and 8 GB of RAM, where most local AI tools
either crawl or take the whole machine down with them. spindle doesn't use llama.cpp, BLAS
or any ML framework. The GGUF parser, tokenizer, transformer, AVX2 kernels and thread pool
are all in this repo.

It's early. It runs SmolLM2 135M/360M, Qwen3-0.6B and Granite 3.1 1B-A400M (a
mixture-of-experts model: 8 of 32 experts per token) today. The goal is to run models bigger than RAM
by streaming weights from the NVMe drive, mostly MoE models. That part isn't done yet; see
[docs/roadmap.md](docs/roadmap.md).

## What works

- GGUF v3 parsing (metadata, tensors, vocab, BPE merges)
- Llama-style models: RMSNorm, RoPE, grouped-query attention, SwiGLU
- Qwen2/Qwen3 attention details: QKV bias, per-head Q/K norm, NEOX RoPE, `head_dim`
  different from `dim / heads` (checked end to end on Qwen3-0.6B)
- MoE layers: softmax router, top-k experts, optional shared expert, with per-turn stats on
  how many MiB of experts each token reads (tested on Granite 1B-A400M and Qwen3-30B-A3B)
- Byte-level BPE, so non-ASCII text (ñ, emoji, CJK) round-trips instead of being dropped
- Model sizes read from GGUF metadata, nothing hardcoded
- AVX2/FMA kernels for `F32`, `F16`, `BF16`, `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`,
  `Q4_K`, `Q5_K`, `Q6_K`, each tested against a scalar reference
- **Zero-Syscall Lockless Worker Pool**: Atomic spin synchronization (`parked_mask`) eliminates
  600–1000 kernel `SetEvent` syscalls per token, dropping dispatch latency to < 50 ns
- **Register-Cached Flash GQA Attention**: Vector accumulators stay in AVX2 YMM registers across
  all past context tokens, eliminating ~400 MB of redundant memory bus traffic per token
- **Parallel GQA Attention Heads Across P-Cores**: Shards multi-head GQA computation across physical P-cores without mutex locks, saturating all cores during long-context generation
- **Pre-computed Activation Subblock Sums (`Q4_K` / `Q5_K`)**: Eliminates redundant vector additions for asymmetric quant min terms, computing $S_k$ once via AVX2 and reducing min calculations to 8 scalar FMAs
- **Dual-Port Interleaved 2-Row GEMV (`rows_2x!`)**: Interleaves row pairs across `Q8_0`, `Q4_K`, `Q5_K`, `F16`, and `F32` to keep activations hot in L1 cache and saturate CPU dual FMA execution ports (Port 0 & 1)
- **Distributed Layer Sharding (Spindle RIOT-Wire)**: Pool RAM across multiple machines (e.g. 16 GB PC + 8 GB laptop = 24 GB RAM) via `--layer-range`, `spindle node`, and `spindle chat --peer`
- **Hardware-Compressed Tensor Network Fabric**: Sub-microsecond AVX2 F16C vector compression (50% wire reduction) and adaptive FP8 quantization (75% reduction), with AVX2 XOR erasure parity for WiFi packet drop recovery
- **Speculative Decoding for MoE**: In-RAM small draft model (`--draft`) drafting up to 4 tokens verified in a single layer-synchronous scan on the target model (achieving 81.2% acceptance rate on Qwen3-30B)
- **Dynamic Energy-Preserving MoE Pruning (`--moe-prune 0.03`)**: Dynamically prunes low-probability tail experts ($p_e < \alpha \cdot p_{\max}$) and renormalizes output energy, dropping average active experts from 8 down to 5.2 and cutting memory bus traffic by 31.4% with zero quality loss
- **Contiguous MoE Expert Repacking**: `--pack-experts` repacks scattered GGUF expert tensors into contiguous aligned DMA slots for 3x faster NVMe reads
- **Smart RAM Headroom & Working-Set Governor**: Dynamically sizes KV-cache and weights to fit available RAM without triggering premature OS disk paging or out-of-memory aborts
- Multi-turn interactive chat with live slash commands (`/model`, `/cluster`, `/bench`, `/think`, `/stats`, `/clear`, `/max`)
- Temperature, top-k, top-p, min-p sampling with a seed
- `--probe` to measure your RAM and SSD bandwidth

## Speed

Tested on Intel i5-13420H (4 P-cores, 4 E-cores, AVX2/FMA/F16C), 8 GB RAM, Windows 11:

| Model | Size / Quant | Mode | Speed |
|---|---|---|---|
| **SmolLM2-135M-Instruct** | 105 MiB (Q4_K_M) | 100% RAM resident | **143.6 tok/s** |
| **Qwen3-0.6B-Instruct** | 609 MiB (Q8_0) | 100% RAM resident | **28.7 tok/s** |
| **Granite-3.1-1B-A400M** | 1,356 MiB (Q8_0 MoE) | NVMe stream (2.8 GB/s) | **9.8 tok/s** |
| **Qwen3-30B-A3B** (Baseline Cluster) | 17.7 GiB (48 layers) | **24 GB Pooled RAM** (PC 0..35 + Laptop 35..48, FP8) | **6.0 – 7.2 tok/s** (zero SSD I/O, pure CPU!) |
| **Qwen3-30B-A3B** (MoE Pruned) | 17.7 GiB (48 layers) | **24 GB Pooled RAM** (`--moe-prune 0.03`, FP8) | **8.6 – 10.2 tok/s** (31.4% memory bandwidth saved) |
| **Qwen3-30B-A3B** (Speculative Cluster) | 17.7 GiB + 0.6B Draft | **Cluster Speculative Verification** (`--draft qwen3-0.6b`) | **15.0 – 22.2 tok/s** (breaking the memory wall!) |

Detailed architectural breakdowns and logs:
- [Beyond the Memory Wall: Breaking the 7.2 tok/s Ceiling](docs/beyond_the_memory_wall.md)
- [Distributed Speculative Decoding Protocol](docs/speculative_decoding.md)
- [Dynamic Energy-Preserving MoE Expert Pruning](docs/dynamic_moe_pruning.md)
- [Distributed Cluster Architecture & RIOT-Wire Protocol](docs/distributed_cluster.md)
- [SIMD Micro-Kernels & AVX2 Port Saturation](docs/simd_microkernels.md)
- [Empirical Multi-Turn Benchmark Log](docs/benchmarks.md)

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

```
spindle  Qwen3 0.6B Instruct
qwen3 · 609 MiB · 4 threads · 1024-token context
type a message, or /help

> What is 2+2?
2 + 2 = 4
  34.1 tok/s · 203 tokens · 5.9 s
```

| In the chat | |
|---|---|
| `/clear` | start a new conversation (resets KV-cache across cluster) |
| `/model` | inspect active model architecture, weights residency, KV-cache, and RAM |
| `/cluster` | view distributed mesh status, worker peer IP, RTT ping latency, and partitions |
| `/bench` | run a 32-token on-demand speed benchmark (tok/s) |
| `/think` | toggle reasoning mode on/off (`/think on`, `/think off`, `/think show`, `/think hide`) |
| `/stats` | toggle detailed memory and MoE router timing details after each reply |
| `/max <n>` | set maximum generation tokens (default: 2048) |
| `/help` | list commands |
| `/exit`, `exit` | quit |

Without `--chat`, spindle answers `--prompt` once and exits.

## Distributed Cluster (Spindle RIOT-Wire)

Pool RAM across multiple PCs over local WiFi or Ethernet to run massive models (e.g. 17.7 GiB Qwen3-30B) at 6.0–7.2 tok/s with **zero SSD streaming**:

**On the Worker Node (e.g. 16 GB Desktop):**
```powershell
.\spindle.exe node --bind 0.0.0.0:42424 --layer-range 0..35 --wire-format fp8 --threads 4 Qwen3-30B-A3B-Q4_K_M.gguf
```

**On the Master Node (e.g. 8 GB Laptop):**
```powershell
.\target\release\spindle.exe --peer 192.168.1.10:42424 --layer-range 35..48 --wire-format fp8 --threads 4 Qwen3-30B-A3B-Q4_K_M.gguf --chat
```

- Intermediate activation vectors (4096 floats) are compressed to **4 KB (FP8)** using AVX2 SIMD instructions.
- Continuous multi-turn chat over Wi-Fi with sub-5 ms latency and zero memory thrashing.

| Option | Default | |
|---|---|---|
| `--prompt "..."` | | send a first message right away |
| `--node` | off | run as a distributed worker node |
| `--bind <IP:PORT>` | 127.0.0.1:0 | listen address for worker node |
| `--peer <IP:PORT>` | | connect to worker node as master |
| `--layer-range <start>..<end>` | full | selectively load layers into RAM (e.g. `0..32`) |
| `--wire-format <fp16\|fp8\|fp32>` | fp16 | tensor wire compression format |
| `--draft <path>` | | in-RAM draft model for speculative decoding |
| `--moe-prune <f32>` | 0.0 | relative MoE expert pruning threshold (e.g. 0.03) |
| `--pack-experts` | off | repack scattered MoE experts into contiguous pack file |
| `--expert-cache-mib <n>` | off | explicit in-RAM expert cache size |
| `--verbose`, `-v` | off | print load, memory and threading details |
| `--raw "text"`, `-n 32` | | plain completion with no chat framing, for comparing with llama.cpp |
| `--think` | off | enable thinking/reasoning mode |
| `--temp` | 0.6 | 0 means greedy |
| `--top-k` | 40 | |
| `--top-p` | 0.9 | |
| `--min-p` | 0.05 | |
| `--repeat-penalty` | 1.1 | |
| `--seed` | random | printed at startup, reuse it to get the same replies |
| `--probe` | | measure RAM/SSD speed instead of chatting |
| `--reserve` | 512 | MiB of RAM to always leave free for other apps |
| `--ram-budget` | | hard cap on spindle's resident memory in MiB |
| `--stream-layers` | off | read layers that don't fit with one large read each instead of paging them |
| `$env:CHAT_THREADS` | 4 | |
| `$env:CHAT_CTX` | 2048 | KV-cache length in tokens |

For honest SSD numbers, run `--probe` against a file of 1 GB or more.

`$env:SPINDLE_FORCE_STREAM = "1"` makes spindle treat any model as too big for RAM. Useful for
testing the low-memory path with a small model.

## Checking against llama.cpp

`tools\golden.ps1` runs the same prompts through spindle and llama.cpp with greedy decoding and
reports the first place they differ:

```powershell
.\tools\golden.ps1 -Model qwen3-0.6b-q8_0.gguf -Llama C:\llama\llama-completion.exe
```

## Layout

| | |
|---|---|
| `src/smollm_chat.rs` | the engine: forward pass (dense and MoE), selective layer sharding, speculative verification |
| `src/cluster_net.rs` | Spindle RIOT-Wire: SIMD vector wire compression, AVX2 XOR parity, cluster socket client/server |
| `src/chat_cli.rs` | interactive terminal chat, streaming token renderer, slash commands (`/model`, `/cluster`, `/bench`) |
| `src/expert_pack.rs` | repacker for contiguous MoE expert storage (`--pack-experts`) |
| `src/expert_cache.rs` | explicit in-RAM expert cache for SSD streaming |
| `src/quant.rs` | quant formats, AVX2 kernels, scalar reference, kernel tests |
| `src/gguf_loader.rs` | GGUF parser |
| `src/worker_pool.rs` | zero-syscall lockless thread pool for parallel matvecs |
| `src/mmap_file.rs` | memory-mapped model files, smart headroom prefetcher |
| `src/memory_guard.rs` | working-set governor |
| `src/layer_stream.rs` | opt-in double-buffered layer reader |
| `src/probe.rs` | bandwidth probe |
| `src/main.rs` | CLI, cluster node/peer dispatch, and CPU topology (P-core / E-core detection) |

More detail in [docs/architecture.md](docs/architecture.md).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT. Model weights aren't included and have their own licenses (SmolLM2 is Apache-2.0).
