# Roadmap

The goal hasn't changed: run models much bigger than one cheap PC's RAM at a usable speed,
without freezing the machine.

## Where 1.1 stands

- Qwen3-30B-A3B runs across a 16 GB desktop and an 8 GB laptop at 8–9 tok/s, at `moe_prune
  = 0.1` (3% perplexity cost). Prompt processing is 25 tok/s.
- Small models run on one laptop at 40–170 tok/s.
- The coding agent works with Qwen3 models, with confirmations and undo.
- Output agrees with llama.cpp up to near-ties (`tools/golden.ps1`).

Done so far, in order:

- GGUF parser, byte-level BPE, Qwen2 pre-tokenizer, Llama/Qwen2/Qwen3/Granite forward passes
- AVX2 kernels for 11 formats, each tested against a scalar reference
- P-core thread pool, memory-mapped weights, working-set governor, KV cache sized to free RAM
- MoE layers: Qwen3, Granite, Qwen2-MoE shared experts
- Expert repacking and an explicit expert cache (opt-in; not faster than mmap yet)
- Two-PC layer split with fp16/fp8 activations, KV-shift sync and context negotiation
- Batched forward pass on multi-vector kernels
- Prompt-lookup and draft-model speculative decoding, both exact
- MoE expert pruning, with its quality cost measured (`--ppl`)
- Agent tools, guard, undo, full-screen interface, startup progress display
- 1.1: per-phase timing (`/timing`, `--bench`), system prompt cache on disk, hot experts kept
  in RAM, integer Q4_K/Q6_K kernels, fused MoE experts, measured draft lengths, pipelined
  prompt processing across the two PCs

## Next

Ordered by how much each item is expected to help, measured against what it costs to build.

### 1. Make verification cheap on MoE

A verify pass costs about 110% of a step for every extra token, so speculation rarely pays on
Qwen3-30B-A3B. The batched MoE path still runs each expert as three separate thread-pool jobs,
and every extra token can route to experts the others don't use.

- Fuse the batched MoE path the way the single-token path is fused (two passes per layer).
- Measure the extra-token cost again (`--bench` prints it).

Done when: an extra verify token costs under 40% of a step.

### 2. More accepted tokens per pass

Only once verification is cheap: lookahead decoding (the model drafts n-grams for itself in
the same pass) and tree verification (several candidate continuations in one batch).

Done when: prose averages ≥ 1.25 tokens per pass with the output unchanged.

### 3. The laptop's share in RAM

The laptop needs 4.9 GiB for its layers and usually has 1.5–3.6 GiB free. Its layers take 26
ms with 3.6 GiB free and about 40 ms when it pages.

- Smaller resident footprint for the LM head and attention (both are read every token).
- Suggest a `layer_range` from measured speed and free RAM on both machines.

### 4. Usable by other people

- Chat templates read from the GGUF instead of the built-in ChatML/Granite formats.
- Remaining quant formats: `Q2_K`, `Q3_K`, the IQ family.
- An OpenAI-compatible local server, so existing chat apps can use spindle.
- Llama 3, Gemma, Mistral, Phi.
- Release builds on GitHub, so nobody needs Rust installed.

### 5. Other platforms

- Linux: `mmap`, `madvise`, `sched_setaffinity`.
- ARM NEON kernels.

## Open questions from earlier phases

- **Expert cache vs. mmap.** The explicit cache (`--expert-cache-mib`) reaches 43–83% hit
  rates on Granite 1B-A400M, but it is still slower than letting Windows page experts in.
  Its misses always go to the SSD, while mapped pages often come back from the standby list.
  It needs reads off the compute thread and a model that really doesn't fit before it can
  win.
- **Predicting experts ahead of time.** Running layer `l+1`'s router on layer `l`'s hidden
  state might guess the next experts early enough to prefetch them. How often the guess is
  right hasn't been measured.
- **E-cores.** They do nothing useful today. They could prefetch, draft, or run the
  interface.

## Rules

- Measure before and after every change, on the same prompt and seed.
- A faster engine that says different things is a bug. Keep `tools/golden.ps1` green.
- Test with little free RAM, not on good days. On the dev laptop, free RAM swings between
  about 400 MB and 1.5 GB.
