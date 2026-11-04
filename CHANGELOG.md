# Changelog

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
- The full-screen interface was rebuilt. It has a prompt box, inline tool rows, diff and
  shell blocks, a permission dialog, a session sidebar, command completion and multi-line
  input.
- Rewrote the interface text, help and messages to be plain and specific.

### Fixes

- Thread pinning works on CPUs with more than 4 performance cores. Before, `--threads 5` or
  higher crashed.
- Model auto-discovery picks the largest `.gguf` that isn't named "draft".
