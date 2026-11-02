# Roadmap

The goal: run models much bigger than RAM on an 8 GB laptop at a usable speed, without
crashing it.

## Why MoE

Each token reads every active weight once, so `tok/s <= bandwidth / active bytes per token`.
Measured on the dev laptop:

| | Bandwidth |
|---|---|
| RAM | ~31 GB/s |
| NVMe, large reads | ~2.2-2.8 GB/s |
| NVMe, random 4 KiB | ~0.03-0.16 GB/s |

A dense 70B model at 4-bit is around 40 GB per token, so ~0.1 tok/s from the SSD no matter
what. A mixture-of-experts model only touches a few experts per token. A ~30B MoE with ~3B
active parameters reads ~1.7 GB per token, and experts repeat between tokens, so a RAM cache
helps a lot. With no cache hits that's ~1.6 tok/s from this SSD. To get past 2 tok/s we
need roughly a 25% cache hit rate.

## Done

- GGUF parser, BPE tokenizer, Llama forward pass
- AVX2 kernels for `F32`, `F16`, `BF16`, `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q4_K`,
  `Q5_K`, `Q6_K`, each tested against a scalar dequantizer
- P-core thread pool (135M went from ~10 to ~190 tok/s)
- Multi-turn chat, sliding KV-cache, config from metadata
- Memory-mapped weights (private RAM for 135M: 373 MiB down to 63 MiB)
- Bandwidth probe
- Temperature / top-k / top-p / min-p sampling with a seed
- Memory safety net: KV-cache sized to free RAM, working-set governor (`--reserve`,
  `--ram-budget`) that shrinks under pressure and recovers, background prefetch of the next
  layer and chosen experts
- f16 KV-cache, weight pinning against cyclic-scan thrash, 512 MiB default reserve, opt-in
  double-buffered layer streaming (Qwen3-0.6B oversized run: 3.7 -> 24-29 tok/s)
- MoE FFN (softmax router, top-k, shared expert), Qwen2/Qwen3 attention, NEOX RoPE,
  byte-level BPE, stop tokens from metadata, per-layer weights resolved at load time
- Qwen3 reasoning handled properly: no stop inside `<think>`, old reasoning dropped from
  history
- Chat UI: quiet startup, slash commands, `--verbose`

## Plan

Rough timing assumes evenings and weekends. Each phase ends with a number we can measure,
and nothing moves to the next phase until that number is real.

### Phase 1: first real MoE (weeks 1-3)

The MoE code has only run in unit tests. Before building anything on top of it, it has to
produce correct text from a real file.

- Start with a MoE that fits in RAM so speed doesn't hide bugs. Candidates: IBM Granite MoE
  (around 1B total, ~400M active; needs the `granitemoe` architecture and its embedding,
  residual and logit multipliers) or OLMoE-1B-7B (~1B active of 7B; `olmoe` architecture,
  close to what we have).
- Golden tests: run the same prompt through llama.cpp with greedy decoding and compare the
  first 32 tokens. Do this for every supported architecture.
- CI on Windows: build, test, clippy, fmt.
- Publish v0.1 on GitHub once SmolLM2, Qwen3 and one MoE pass golden tests.

Done when: a MoE model matches llama.cpp token for token on a fixed prompt.

Status: Granite 3.1 1B-A400M (Q8_0) chats coherently, the first real MoE through the
engine (18 tok/s with `--reserve 256` and ~500 MB free). Against llama.cpp b12032 with
`tools/golden.ps1`, prompt tokens match exactly for every model, SmolLM2-360M matches on all
4 prompts and Qwen3-0.6B on 3. Every other divergence happens at a near-tie: llama.cpp's
token is spindle's second choice with a logit gap of 0.01-0.25 (llama.cpp rounds activations
to 8 bits, spindle doesn't). No divergence pointed at a logic bug. Left: push and tag v0.1.

### Phase 2: experts from disk (months 1-2)

The first model that really doesn't fit. OLMoE-1B-7B at Q4 (~4 GB) is the practice target,
since it's bigger than the RAM we can use but small enough to iterate on.

- Repacked layout: a one-time pass that writes each expert's gate/up/down next to each
  other, 4 KiB aligned, so one read gets a whole expert. The original GGUF stays untouched.
- Explicit expert cache in private memory with a fixed budget, replacing mmap paging for
  experts. Track hit rate per layer; start with LRU, then try frequency plus recency.
- Reader thread with ~8 reads in flight (the probe showed QD8 nearly doubles 1 MiB random
  reads), running on an E-core.
- Dense parts (attention, norms, router, embeddings) stay pinned.

Done when: OLMoE-1B-7B runs at 10+ tok/s with a 3 GB budget, and the cache hit rate is
printed with every reply.

Status: first version in, not yet faster than mmap.

- `--pack-experts` writes `<model>.spindle.pack`: every expert's gate/up/down back to back in
  one 4 KiB-aligned slot (Granite 1B-A400M: 768 experts, 1224 MiB).
- `--expert-cache-mib N` turns on an in-RAM cache of N MiB filled by unbuffered reads from
  the pack, with per-layer fair share and frequency/recency eviction. Output is identical to
  the mmap path (greedy, 3 prompts).
- Measured on Granite 1B-A400M with ~800 MB free and `--reserve 256`: 5.7-10.2 tok/s with the
  cache (43-83% hit rate depending on how big a budget free RAM allowed) against 8.7-33 tok/s
  reading experts through the mmap. Unbuffered misses always go to the SSD, while mapped pages
  often come back from the standby cache. So the cache is opt-in for now.
- Not done: the cache needs to beat mmap before it's the default. Ideas: reads off the compute
  thread with several in flight, buffered reads so misses can hit the OS cache, and a model
  that really doesn't fit (OLMoE-1B-7B) where mmap has no standby cache to lean on.

### Phase 3: knowing which experts come next (months 2-4)

Waiting for the router before starting a read is where the time goes. Read early instead.

- Lookahead routing: run layer `l+1`'s router on layer `l`'s hidden state to guess its experts
  a layer early. Measure how often the guess is right before relying on it.
- Draft-guided prefetch: a small model drafts a few tokens ahead; the big model's routers on
  the draft's path predict experts several tokens early.
- Speculative decoding: verify k draft tokens in one pass, so each loaded expert serves
  several tokens. The drafter runs on the E-cores.

Done when: Qwen3-30B-A3B runs at 2+ tok/s on the dev laptop (the original target), then
4+ tok/s.

### Phase 4: faster math (months 4-6)

Once the right weights are in RAM, compute is the next limit, especially for long prompts.

- Batched prompt processing: the whole prompt as a matrix-matrix multiply instead of one
  token at a time.
- Int8 activations with `maddubs`, then AVX-VNNI, which 12th/13th gen Intel has.
- Attention split across threads; no allocations per token.
- Benchmark against llama.cpp on the same files, prefill and decode measured separately.

Done when: decode within 20% of llama.cpp on SmolLM2 and Qwen3 dense, prefill at least 3x
faster than today.

### Phase 5: usable by other people (months 6-9)

- More architectures: Llama 3, Gemma, Mistral, Phi. Also LFM2-style hybrid models: Liquid's
  LFM2.5-8B-A1B is a laptop-sized MoE (~1.5B active of ~8B) that fits this project well, but
  it mixes convolution blocks with attention, so it needs new layer types.
- Chat templates read from the GGUF instead of hardcoded ChatML.
- Remaining quant formats: `Q2_K`, `Q3_K`, the IQ family.
- OpenAI-compatible local server, so existing chat apps can use spindle.
- Releases as a single `.exe`; a one-line install.

Done when: someone who isn't us can download a release, point it at a GGUF from Hugging
Face and chat, without reading the source.

### Phase 6: other machines (months 9-12)

- Linux: `mmap`, `io_uring`, `sched_setaffinity`.
- ARM NEON kernels. Android phones are the other big group of people with limited RAM.
- Energy: Joules per token in the stats, and a battery mode that trades speed for runtime.

Done when: the same model runs on Linux and on an ARM device, with numbers in the README.

### Year 2: research

- Activation sparsity for dense models: skip FFN rows predicted to be near zero (the "LLM in a
  Flash" idea), now with SwiGLU-aware predictors.
- Quantization designed for streaming: formats that pack each expert for one aligned read.
- Write up what worked and what didn't, with the measurements.

## Things that are easy to forget

- Measure before and after every change, on the same prompt and seed; laptop clocks are noisy.
- Keep the golden tests green. A faster engine that says different things is a bug.
- Free RAM on this laptop swings between ~400 MB and ~1.5 GB depending on what else is open.
  Test the low end, not the good days.
