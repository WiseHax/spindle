# Related work

Notes on what other people have done in this area. Mostly from abstracts and discussion
threads (Nov 2026), and several are preprints, so check the sources before quoting numbers.

| Work | Idea | Notes for us |
|---|---|---|
| LLM in a Flash (Apple, ACL 2024) [1] | Weights in flash, load only rows predicted to be non-zero, cache recent ones, bundle rows and columns into bigger reads | Bundling is useful. The sparsity trick is less useful for SwiGLU models. Gets ~2x DRAM size, not 10x |
| llama.cpp mmap | Map the file and let the OS page | Once the model is bigger than RAM, page faults thrash and speed falls to ~0.1-0.2 tok/s [3] |
| Expert-aware SSD streaming, llama.cpp #27149 [2] | Read only the 8 of 128 experts each Qwen3-30B-A3B layer uses | Reports ~4.7 tok/s in ~2 GB RAM. Prototype, no prediction or memory pressure handling |
| FlashMoE (2026) [4] | SSD expert offload with a learned cache replacement policy | Cache hit rate matters more than raw I/O |
| Edge0 (2026) [5] | Trained predictor guesses the next layer's experts, 35B MoE in ~2.9 GiB active memory | Prefetching is how you hide SSD latency, but it needs a trained predictor per model |
| SSD-LLaMA (2026) [6] | SSD, RAM and VRAM managed as one hierarchy, trillion-param MoE at 1+ tok/s | Assumes a GPU |
| Hybrid CPU parallelism (NeurIPS 2024) [7] | Split work by each core's measured speed on Intel P+E CPUs | Only splits compute, doesn't give E-cores different jobs |
| SSD offload energy study (2025) [8] | SSD offload costs ~3-5x more energy per token | Laptops run on battery, worth measuring |

What I take from this:

- MoE plus SSD is the right target.
- Predicting the next experts matters more than raw read speed.
- Cache policy decides most of the performance.
- Almost all of it targets Linux, macOS or a GPU. Cheap Windows laptops are barely covered.

## Where spindle tries something different

These are things I haven't seen done this way. Each one needs a test that could prove it
wrong.

- **Memory governor.** Others assume a fixed RAM budget. On a real laptop, someone opens Chrome
  halfway through a reply. The cache should shrink and grow with available memory.
- **E-cores as the I/O side.** P-cores do the math. E-cores predict, prefetch and manage the
  cache instead of doing slow math.
- **Draft-guided prefetch.** Use a small draft model's lookahead to predict experts several
  tokens early, without training a predictor.
- **Streaming layout.** Repack experts so each one is one contiguous, aligned read.
- **Layer split across two consumer PCs.** Tensor parallelism needs a fast interconnect.
  Splitting contiguous layers needs only one hidden-state vector per token on the wire
  (about 2 KB at 8 bits for Qwen3-30B-A3B), which a home network handles. The cost is that the
  machines take turns.
- **Prompt-lookup speculation over the network.** Draft tokens come from the conversation
  itself, so verifying a batch costs one round trip and no draft model.
- **Energy.** Report Joules per token, and offer a battery mode.

## Measurements from the dev laptop

i5-13420H, 8 GB RAM, Micron MTFDKCD512QGN 512 GB NVMe, `--probe` on a 1 GiB file:

| Test | Result |
|---|---|
| RAM, 1 thread | 20.7 GB/s |
| RAM, 4 / 12 threads | 30.0 / 31.2 GB/s |
| SSD sequential 4 MiB, QD1 / QD8 | 2.24 / 2.74 GB/s |
| SSD random 1 MiB, QD1 / QD8 | 1.61 / 2.84 GB/s |
| SSD random 4 KiB, QD1 / QD8 | 0.03 / 0.16 GB/s |

The same test on a 369 MiB file gave higher sequential numbers (3.5 GB/s) but no gain from
QD8. A file that small probably sits in the drive's own cache, so use bigger files.

What this means:

1. Small models aren't bandwidth-bound yet. 360M runs at ~53 tok/s against a ~80 tok/s ceiling.
2. 4 KiB random reads are 20-100x slower than 1 MiB reads, so the repacked layout and large
   explicit reads are needed, not page faults.
3. Keep about 8 reads in flight.
4. A ~30B / 3B-active MoE at 4-bit is ~1.7 GB per token, ~1.6 tok/s with no cache hits.
   Caching and prediction are required to hit 2 tok/s.

## Sources

1. [LLM in a Flash](https://arxiv.org/abs/2312.11514v1)
2. [Expert-Aware SSD Streaming for MoE Models, llama.cpp #27149](https://github.com/ggml-org/llama.cpp/discussions/27149)
3. [llama-cpp-expert-sniper notes](https://huggingface.co/waltgrace/llama-cpp-expert-sniper/commit/c94ca8dc68a7e6d37dbee0cc5e1adb3af5d2a0e8)
4. [FlashMoE](https://arxiv.org/html/2601.17063)
5. [The Other Half of the Memory Wall (Edge0)](https://www.alphaxiv.org/abs/2609.18063)
6. [SSD-LLaMA](https://arxiv.org/html/2609.18110v1)
7. [A dynamic parallel method for performance optimization on hybrid CPUs](https://huggingface.co/papers/2411.19542)
8. [SSD Offloading for LLM MoE Weights Considered Harmful in Energy Efficiency](https://www.alphaxiv.org/abs/2508.06978)
