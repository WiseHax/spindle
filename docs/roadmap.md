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
- AVX2 kernels for `Q8_0`, `Q5_0`, `Q4_K`, `Q6_K`
- P-core thread pool (135M went from ~10 to ~190 tok/s)
- Multi-turn chat, sliding KV-cache, config from metadata
- Memory-mapped weights (private RAM for 135M: 373 MiB down to 63 MiB)
- Bandwidth probe
- Temperature / top-k / top-p / min-p sampling with a seed
- `Q4_0`, `Q4_1`, `Q5_1`, `Q5_K`, `F16`, `BF16` kernels, all kernels tested against a scalar
  dequantizer
- Memory safety net: KV-cache sized to free RAM, working-set governor (`--reserve`,
  `--ram-budget`) that shrinks under pressure and recovers, background prefetch of the next
  layer and chosen experts
- MoE FFN (softmax router, top-k, shared expert), Qwen2/Qwen3 attention, NEOX RoPE,
  byte-level BPE, stop tokens from metadata, per-layer weights resolved at load time

## Next

1. **Run a real MoE end to end.** The MoE layer, Qwen attention and NEOX RoPE are written and
   unit-tested, but no Qwen or MoE GGUF has gone through them yet. Validate on a small dense
   Qwen first, then a Qwen MoE, and compare output against llama.cpp.
2. **Streaming file layout.** A one-time repack that puts each expert's tensors next to each
   other, 4 KiB aligned, so one large read gets a whole expert. The original GGUF stays as is.
3. **Expert cache.** The governor now caps memory and recovers under pressure (see "Done"),
   but which experts stay resident is left to Windows. An explicit cache that keeps the most
   used experts would raise the hit rate.
4. **Prefetch on the E-cores.** P-cores do the math. E-cores predict which experts come next and
   issue the reads (IOCP, ~8 in flight) and manage the cache.
5. **Draft-guided prefetch.** A small draft model runs a few tokens ahead. Running the big
   model's router on the draft's path predicts experts several tokens early, and verifying k
   draft tokens at once loads the union of their experts in one go.
6. **Energy numbers.** Joules per token in the stats, and a battery mode.

Target: a ~30B MoE at 2+ tok/s on the dev laptop.

## Also needed

- CI on Windows (build, test, clippy, fmt)
- Golden-output test for a tiny model
- Use the model's own chat template instead of hardcoded ChatML
- Proper GPT-2 pre-tokenizer rules (contractions, digits, unicode)
- Quantized KV-cache (f16, then q8)
- No allocations per token
- Int8 activations with `maddubs` / AVX-VNNI dot products
- Batched prompt processing
- Linux and macOS, ARM NEON, AVX-512
- More architectures: Qwen2/3, Llama 3, Mistral, Phi, Gemma
- OpenAI-compatible local server
