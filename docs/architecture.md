# How it works

Roughly what happens between loading a file and printing a token. Paths are under `src/`.

## Loading

A GGUF file is a header, a key/value metadata table, a tensor table, then the tensor data.
`gguf_loader.rs` parses everything except the data: model sizes, vocab, BPE merges, and each
tensor's name, shape, type and offset.

`ModelConfig::from_gguf` pulls the hyperparameters out of keys like
`llama.embedding_length`, `llama.block_count` and `llama.attention.head_count_kv`, so the same
code runs any Llama-shaped model.

The file is memory-mapped read-only (`mmap_file.rs`). Kernels read weights directly from the
mapping. If the file fits in less than half of the free RAM, it's prefetched in the
background with `PrefetchVirtualMemory`; otherwise pages load as they're touched. Because the
pages are file-backed and never written, Windows can drop them under memory pressure and read
them back later. Chat gets slower instead of the laptop running out of memory.

## Tokenizing

Vocab strings use GPT-2's byte alphabet: every byte maps to a printable char (space is
`Ġ`, newline `Ċ`). Encoding maps input bytes into that alphabet, decoding maps them back and
holds partial UTF-8 characters until the rest arrives.

`encode_bpe_segment` splits text GPT-2 style (a leading space sticks to the next word and
becomes `Ġ`, newlines become `Ċ`), then applies BPE merges in rank order. Turns are wrapped in
ChatML:

```
<|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n
```

## Forward pass

`forward_hidden` runs one token through every layer:

```
x -> RMSNorm -> Q,K,V -> RoPE -> store K,V -> attention -> output proj -> add to x
  -> RMSNorm -> gate, up -> SiLU(gate) * up -> down proj -> add to x
```

Qwen models add a bias to Q, K and V, and Qwen3 RMS-normalizes each Q and K head before
RoPE. Both are picked up automatically if the tensors exist. All tensor lookups happen once
at load (`resolve_layers`), so the per-token path never formats a tensor name.

Then a final RMSNorm and the LM head give one logit per vocab entry. The LM head is
`output.weight` if the model has one, otherwise the embedding matrix.

RoPE comes in two layouts. Llama GGUFs rotate adjacent pairs `(2i, 2i+1)`; Qwen rotates
`(i, i + head_dim/2)` ("NEOX"). The architecture name decides which.

## Mixture of experts

In an MoE layer the FFN is replaced by `n_expert` smaller FFNs stored as 3D tensors
(`ffn_gate_exps` is `[dim, n_ff, n_expert]`). A `Mat` view with `.expert(e)` points at one
expert's slab, so the same matvec kernels work on it. Per token:

1. the router (`ffn_gate_inp`) gives one logit per expert, softmaxed
2. the top `expert_used_count` experts are picked; their weights are renormalized for Qwen3
   and Mixtral, not for Qwen2-MoE
3. each chosen expert runs as a normal SwiGLU block and the outputs are summed by weight
4. Qwen2-MoE also adds a shared expert scaled by `sigmoid(shared_gate . x)`

Only the chosen experts' pages are touched, which is what makes streaming from SSD
possible later. The chat prints how many MiB of experts each token read.

Attention is grouped-query (SmolLM2-135M has 9 query heads sharing 3 K/V heads) and uses an
online softmax, so the score vector is never stored.

## Kernels

Weights stay quantized. A row is a sequence of blocks:

| Type | Weights/block | Bytes | Contents |
|---|---|---|---|
| `Q4_0` | 32 | 18 | f16 scale, 32 x 4 bits |
| `Q4_1` | 32 | 20 | f16 scale and min, 32 x 4 bits |
| `Q5_0` | 32 | 22 | f16 scale, 32 high bits, 32 x 4 bits |
| `Q5_1` | 32 | 24 | f16 scale and min, 32 high bits, 32 x 4 bits |
| `Q8_0` | 32 | 34 | f16 scale, 32 int8 |
| `Q4_K` | 256 | 144 | f16 d and dmin, 8 six-bit scales and mins, 256 x 4 bits |
| `Q5_K` | 256 | 176 | like `Q4_K` plus 256 high bits |
| `Q6_K` | 256 | 210 | 256 x (4+2) bits, 16 int8 scales, f16 d |

`F32`, `F16` (via F16C) and `BF16` are also supported. All of this lives in `quant.rs`.

Each `dot_*_row_avx2` unpacks a block into 8-wide float vectors and accumulates with FMA.
Scales are applied once per block or sub-block. For `Q4_K` each 32-weight sub-block
contributes `d*sc*sum(q*x) - dmin*mn*sum(x)` (same for `Q5_K`, and `Q4_1`/`Q5_1` use
`d*sum(q*x) + m*sum(x)`).

`dequant_row` is a plain scalar version of every format written from the ggml definitions.
The test in `quant.rs` runs each SIMD kernel on random blocks through the real matvec
dispatch and compares against it. It also serves embedding lookups.

## Threads

A token is around 200 matvecs. Spawning threads for each one would cost more than the
math, so `worker_pool.rs` keeps one helper per P-core alive:

1. The main thread stores the job and bumps a generation counter.
2. Helpers see it (they spin briefly between jobs, then park).
3. Every thread, the main one included, computes its own slice of output rows.
4. The main thread waits until all helpers report done.

Idle helpers park, so a chat waiting for input uses no CPU. Matvecs under 64K weights run on
one thread because syncing costs more than they do.

## KV-cache

The cache holds K and V (after RoPE) for every layer and position, and it survives between
chat turns, so each turn only processes the new message.

When it's full, `ensure_kv_room` drops the oldest tokens after the system prompt and moves the
rest down. Those K vectors were rotated for their old positions, so each one is rotated again
by `-n_discard`. RoPE rotations add up, so this gives the same result as recomputing.

## Sampling

`sample_next_token` computes logits with the same parallel matvec, then:

1. penalizes tokens already used in this reply
2. keeps the top k (partial select, then sorts only those)
3. softmax with temperature
4. min-p: drops anything below `min_p * p(best)`
5. top-p: keeps the smallest set that reaches `top_p`
6. draws with a seeded SplitMix64

`--temp 0` skips all of that and takes the argmax.

## Not crashing the laptop

Running a model bigger than free RAM fails in three ways, and each has a guard:

1. **A big allocation fails and the process aborts.** The KV-cache is the only large private
   allocation. At load it is sized to at most a quarter of free RAM and allocated with
   `try_reserve`, so a tight machine gets a shorter context and a message instead of an abort.
2. **The desktop freezes.** Mapped weights don't fail allocations, but as they page in, our
   working set grows and Windows starts trimming other programs. `memory_guard.rs` sets a hard
   working-set cap with `SetProcessWorkingSetSizeEx` and moves it every 250 ms so that
   `--reserve` MiB (default 1024) stay free. Over the cap, Windows recycles our own model
   pages, which go to the standby list and usually come back as soft faults. The cap is only
   switched on when the model doesn't fit in RAM, or when `--ram-budget` is given.
3. **4 KiB page faults are slow.** When the model doesn't fit, a background thread issues
   `PrefetchVirtualMemory` for the next layer's attention weights and, as soon as the router
   has picked them, for the chosen experts. That turns page faults into large reads that
   overlap with compute. It runs off the compute thread because the call walks every page,
   which costs as much as the matvec when pages are already resident.

Tested on the dev laptop with SmolLM2-360M forced into streaming mode:

- `--ram-budget 300` with a 368 MiB model: working set held at 299 MiB, same output, 7.7 tok/s.
- Another process grabbing 600 MB mid-chat with `--reserve 1200`: the cap dropped to the floor
  (272 MiB), replies kept going at 13-30 tok/s, the cap grew back once the memory was released,
  and all four replies finished.
- `CHAT_CTX=100000`: the KV-cache was shrunk from 8192 to 4365 tokens to fit.

### Making the oversized case faster

Measured with Qwen3-0.6B (610 MiB) on the dev laptop, greedy, same prompt:

| Change | Result |
|---|---|
| Default reserve 1024 -> 512 MiB | the run that started at 3.7 tok/s now gets 24-29 tok/s when ~700 MiB is free. "Available" already includes the standby cache, so the larger reserve was squeezing a model that RAM could hold |
| f16 KV-cache | private memory for SmolLM2-360M 98 -> 58 MiB; under a 450 MiB cap, 31 -> 111 MiB of weights pinned, 6.3 -> 7.8 tok/s |
| Pinning (`VirtualLock`) a fixed part of the model | +10-20% under a cap, noisy. The working set acts like LRU, and LRU on a cyclic scan bigger than the cache misses on nearly every access; pinned bytes can't be evicted, so they always hit |
| `--stream-layers` | page faults per reply 4M -> 41k, speed the same (~7-8 tok/s). That is the drive's speed: ~400 MiB read per token at ~2.5 GB/s. Splitting each read 4 ways didn't help either |

Under a tight cap the soft faults themselves, not the SSD, were most of the time at first:
21M faults in 35 s on the original run. Once the cap stops squeezing needlessly and more
weights are pinned, what is left is the drive. Beyond that the only levers are reading fewer
bytes per token: smaller quants, more RAM for pinning, and MoE models that read a fraction of
their weights.

`--stream-layers` stays opt-in because it plans residency once at startup and can't grow
into memory that frees up later, while mmap paging can.

## The speed limit

Generating a token reads every weight once and does little math per byte, so:

```
tok/s <= memory bandwidth / model bytes
```

On the dev laptop `--probe` reports about 31 GB/s from RAM. SmolLM2-360M Q8_0 is 0.39 GB, so
the ceiling is around 80 tok/s and we get about 53. There's still room on the compute side.

The SSD is about 10x slower than RAM for large reads and far worse for small random ones:
4 KiB random reads came out at 0.03-0.16 GB/s. That's what mmap page faults look like, so
anything streamed from the SSD has to be read in big contiguous chunks with several reads in
flight.
