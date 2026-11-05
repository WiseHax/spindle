# Changelog

## Unreleased

### Speed

- Integer matrix kernels for Q4_K and Q6_K weights. The activation vector is quantized to
  8 bits once per matrix (one scale per 256 values), and the products run in integers
  (AVX2 `vpmaddubsw` / `vpmaddwd`) with one float multiply per block, the same scheme as
  llama.cpp's CPU backend. Per matrix this is 1.8–2.4x faster on one thread. One Qwen3-30B-A3B
  MoE layer's expert work (8 experts, 24.5 MB) went from 2.3 ms to 0.94 ms on the laptop's 4
  threads (10.7 → 26 GB/s). Perplexity on `tools/ppl_sample.txt` (Qwen3-30B-A3B, 256 tokens):
  9.224 float, 9.207 integer. `--float-kernels` (or `float_kernels = true`,
  `SPINDLE_FLOAT_KERNELS=1`) switches back; the startup display says which is in use.
- The chosen experts of a MoE layer run as one fused job: every gate/up row of every expert in
  one parallel pass, then every output row in a second pass that also sums the experts. That
  is 2 thread-pool handoffs per layer instead of 3 per expert (24 for 8 experts), and the input
  is quantized once for all experts. Dense FFNs get the same two-pass form.
- Batched verify and one-by-one decoding give bit-identical results on all-K-quant models: the
  integer kernels compute each vector the same way at any batch size, and F32 router matrices
  now run per vector in a batch too.

### Fixed

- `spindle.toml` values followed by a comment (`moe_prune = 4  # top 4 experts`) were
  silently ignored, because the comment was read as part of the value. The `spindle.toml`
  shipped with 1.0 had such comments, so `moe_prune`, `ctx`, `tools` and `speculate` from it
  did not apply. Comments after a value now work (a `#` inside quotes is kept), and a value
  that doesn't parse or an unknown setting prints a warning with its line number.

### Startup

- The system prompt's KV entries are saved to disk after the first start (on the client and
  on the worker, each for its own layers, next to the model as `<model>.kv-<layers>.bin`).
  Later starts load them instead of recomputing: on Qwen3-0.6B over a local worker, the
  902-token tools prompt went from 5.9 s to 61 ms. The cache key covers the spindle version,
  the model file, the layer split, MoE pruning, the wire format and the prompt itself, so
  any change computes it again. A test checks that the reply is identical either way. Set
  `SPINDLE_NO_PROMPT_CACHE=1` to turn it off.

### Memory

- When a PC's share of the model doesn't fit in RAM, spindle now keeps the experts that
  earlier sessions used most locked in RAM, after the weights every token reads (attention,
  routers, LM head). Rare experts still read from disk. Usage counts are kept next to the
  model (`<model>.experts-<layers>.bin`, 24 KB for Qwen3-30B-A3B) and updated as you chat.
  The first start after this change has no history yet and says so. On Qwen3-30B-A3B the
  benchmark prompts needed 93 of 128 experts per layer (11.9 GiB) for 99% of picks.
- The worker now gets the same pinning and working-set governor as the client. Before, it
  had neither.
- A "Pinning" step on the loading screen shows the progress, what was kept in RAM, and the
  share of past expert picks it covers.

- `--resident-experts` (or `resident_experts = true`), opt-in: when a PC's share doesn't
  fit, MoE layers pick only among the experts locked in RAM, so nothing is read from disk.
  This changes the output. On Granite 1B-A400M with pinned experts covering 54% of past picks
  it ran 2.2x faster but perplexity rose from 7.7 to 29.8, so it is off by default. Measure
  it on your model and split with `--ppl` before using it. With every expert resident it is
  exact (tested). On a cluster the worker restricts its own layers to its own pinned set
  (`FRAME_SET_EXPERT_MODE`, a new frame).
- The set of experts kept in RAM is saved (`<model>.pins-<layers>.bin`) and reused on the
  next start while it still fits and fills at least 85% of the budget. Resident-only routing
  then stays the same from start to start, so its prompt cache keeps working. Before, the
  set shifted slightly with free RAM, and every start recomputed the system prompt.

### Measuring (more)

- `--ppl <file>` prints the model's perplexity on a text file (no chat framing, up to 1,024
  tokens by default, `-n` to change). Use it to compare settings that change the output:
  `moe_prune`, `--resident-experts`, the wire format. `tools/ppl_sample.txt` is a ready
  sample with prose and code.
- `--bench` on MoE models reports expert usage: how many experts per layer cover 50%, 90%,
  95% and 99% of the picks, and how much RAM those experts take. On two PCs it includes the
  worker's layers (`FRAME_EXPERT_STATS`, a new frame).

### Measuring

- `/timing` (both interfaces) shows where the last reply's time went, per forward pass:
  worker compute, network, local layers, LM head, sampling, drafts, text and interface, and
  whatever is left unaccounted, plus the prompt processing speed.
- The worker reports its own compute time (`FRAME_TIMING`, a new frame), so the client can
  split worker time from network time. Workers older than this don't answer; the client
  asks once, then stops.
- `spindle --bench [runs]` runs three fixed prompts (prose, a copy-heavy code edit, a list)
  in one session, greedy, each from the same starting point, and prints the median speed,
  accepted drafts and the time breakdown. `tools\bench.ps1` wraps it and saves the result with
  the commit, CPU and free RAM to `bench-logs\`.
- When this PC's share of the weights doesn't fit in RAM, the startup display says how much
  is needed and how much is free, instead of only "read from disk".
- The RAM fit check uses the exact size of this PC's layers and LM head, not an estimate from
  the file size.

## 1.0.0 (2026-11-05)

First release. Spindle runs Qwen3-30B-A3B across two PCs on a home network (7–8 tok/s on a
16 GB desktop plus an 8 GB laptop over Wi-Fi).

### Inference

- Multi-vector matrix kernels for Q4_K, Q6_K, Q8_0 and F32. Each quant block is decoded once
  and applied to up to 8 tokens, so a batch costs much less than one pass per token. On one
  thread, 5 tokens cost 1.4–1.8x a single token, where separate passes would cost 5x.
- The batched forward pass (prefill and speculative verify) runs attention projections, the
  MoE router, dense FFNs and every routed expert through these kernels.
- Prompt-lookup speculative decoding, on by default. Drafts come from n-gram matches in the
  recent context. Verification samples every position with the normal sampler, so the output
  is the same as without speculation.
- The tokenizer now matches Qwen2's pre-tokenizer exactly. Before this, prompts reached Qwen
  models split differently from training.

### Cluster

- The client tells the worker how to shift its KV cache (`FRAME_KV_SHIFT`), so both halves
  of the model keep the same positions after the context window slides. Before, the two
  drifted apart after the first shift.
- The worker reports its context capacity (`FRAME_KV_INFO`), and the client uses the smaller
  of the two. The worker rejects positions outside its cache instead of writing past it.
- Clearer worker messages for connects, disconnects and mismatched models.

### Startup

- A loading display that shows each step: reading the model, loading the weights into RAM
  (with a progress bar, speed and time left), connecting to the worker, and processing the
  system prompt (with a progress bar). Before, startup printed a few lines and then paused
  silently while the system prompt was processed.
- The full-screen interface opens on a loading screen while the system prompt is
  processed: the startup steps already done, the current one, and a progress bar driven by
  the tokens actually processed, with speed and time left. Starting a new conversation
  (`/new`, `/tools`, `/system`) shows the same screen.
- Weights that fit in RAM are read in at startup, so the first reply doesn't wait on the disk.
- Fixed the read-ahead ranges for resident weights: most layers were being skipped.
- If the worker can't be reached, the error says how to start it and which port to open.
- `--help` lists every option, grouped. `--version` was added.

### Agent and interface

- Tools use Qwen's native `<tools>` / `<tool_call>` format, with a lenient JSON parser.
  Every tool call in a reply now runs, not just the first one.
- New `grep`, `glob` and `fetch_url` tools. `edit_file` needs a unique match unless
  `replace_all` is set.
- Deleting files, guarded commands and writes outside the project ask for confirmation in
  both interfaces.
- The full-screen interface was rebuilt with Spindle's own look: a navy and blue theme, the
  pixel logo with a blue gradient, and a rounded prompt box that shows the mode, model and
  worker in its border. User messages are marked `›`, and tool calls appear as rows (`▸`
  read, `✎` edit, `⌕` search, `❯` shell) or as diff and output blocks. It also has a
  permission dialog, a session sidebar, command completion and multi-line input.
- Rewrote the interface text, help and messages to be plain and specific.
- Fixed: scrolling with the mouse wheel recalled earlier messages into the prompt (the
  terminal turned wheel steps into ↑/↓). The wheel now scrolls the conversation only.
  Separately, ↑ only browses history from an empty prompt, and editing a recalled message
  keeps it.
- Fixed: emoji and CJK text pushed the rest of the row one column to the right, which
  misaligned the sidebar.
- Fixed: a burst of keys or wheel steps was drawn one frame per key, so the screen lagged
  behind. All pending input is now handled before each frame, and frames are drawn as one
  synchronized update.
- New keys: ctrl+a/e, ctrl+←/→, ctrl+w, ctrl+u/k, ctrl+z (undo), ctrl+home/end, ctrl+l,
  alt+enter. ↑/↓ move between lines of a multi-line prompt. Page up/down scrolls a full
  screen. While scrolled up, streaming text no longer moves the view.

### Fixes

- Thread pinning works on CPUs with more than 4 performance cores. Before, `--threads 5` or
  higher crashed.
- Model auto-discovery picks the largest `.gguf` that isn't named "draft".
