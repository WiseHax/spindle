# Roadmap

The goal hasn't changed: run models much bigger than one cheap PC's RAM at a usable speed,
without freezing the machine.

## Where 1.0 stands

- Qwen3-30B-A3B runs across a 16 GB desktop and an 8 GB laptop at 7–8 tok/s, with no disk
  paging on either one.
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
- MoE expert pruning
- Agent tools, guard, undo, full-screen interface, startup progress display

## Next

Ordered by how much each item is expected to help, measured against what it costs to build.

### 1. Find the missing 18 ms per token

On the two-PC setup, a token costs about 73 ms on the worker plus 31 ms on the laptop, which
would allow 9.6 tok/s. The measured speed is 8.2 tok/s, so about 18 ms per token goes
somewhere else. Candidates are sampling over a 151k vocabulary, detokenizing, drawing the
interface, and network jitter.

- Add a per-phase breakdown to `/stats`: network wait, worker compute (reported by the
  worker), local layers, LM head, sampling, interface.
- Fix whatever dominates.

Done when: the parts add up to the measured time per token, and the gap is under 5 ms.

### 2. Measure what pruning costs

`moe_prune = 4` is in daily use without a quality number.

- Perplexity on a fixed text with pruning off, at 6, at 4, and at 0.05.
- A small set of fixed coding prompts, comparing the replies.

Done when: docs/moe_pruning.md has a table and a recommended default.

### 3. Keep both PCs busy

Right now the machines take turns: while the worker computes, the laptop waits, and the
other way round. Options:

- Overlap the laptop's LM head and sampling with the worker's next step, when a draft makes
  the next token predictable.
- Split by measured speed instead of by hand: time a few tokens at startup and suggest a
  `layer_range`.

Done when: both machines are busy more than 70% of the time during generation.

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
