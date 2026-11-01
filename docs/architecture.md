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

Then a final RMSNorm and the LM head give one logit per vocab entry. The LM head is
`output.weight` if the model has one, otherwise the embedding matrix.

Attention is grouped-query (SmolLM2-135M has 9 query heads sharing 3 K/V heads) and uses an
online softmax, so the score vector is never stored.

## Kernels

Weights stay quantized. A row is a sequence of blocks:

| Type | Weights/block | Bytes | Contents |
|---|---|---|---|
| `Q8_0` | 32 | 34 | f16 scale, 32 int8 |
| `Q5_0` | 32 | 22 | f16 scale, 32 high bits, 32 x 4 bits |
| `Q4_K` | 256 | 144 | f16 d and dmin, 8 six-bit scales and mins, 256 x 4 bits |
| `Q6_K` | 256 | 210 | 256 x (4+2) bits, 16 int8 scales, f16 d |

Each `dot_*_row_avx2` unpacks a block into 8-wide float vectors and accumulates with FMA.
Scales are applied once per block or sub-block. For `Q4_K` each 32-weight sub-block
contributes `d*sc*sum(q*x) - dmin*mn*sum(x)`.

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
