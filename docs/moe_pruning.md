# MoE expert pruning

In a mixture-of-experts layer, a router scores every expert for the current token, and the
top `k` of them run. For Qwen3-30B-A3B that is 8 of 128 per layer. The experts are most of
the weights a token reads, so running fewer of them reads less memory and finishes sooner.

The router's weights are usually uneven: the first few experts carry most of the probability,
and the last ones add little. `moe_prune` drops some of those last experts.

## Settings

`--moe-prune <x>`, `moe_prune = x` in `spindle.toml`, or `/moe-prune x` during a chat:

| Value | Meaning |
|---|---|
| `0` | off: all `k` routed experts run (the default) |
| `x ≥ 1` | run the top `x` experts, for example `4` |
| `0 < x < 1` | run the fewest top experts whose probability adds up to at least `1 − x` of the top-`k` total, for example `0.1` keeps 90% |

In both modes at least half of the routed experts always run (at least 2). For Qwen3-30B-A3B
that means 4 of 8. The kept experts' weights are rescaled so they sum to the same total as
before, which keeps the size of the layer's output the same.

On two PCs, the client sends the setting to the worker (`SET_PRUNE`), so both halves of the
model prune the same way. A change made during a chat applies to both.

## What it costs

Pruning changes the model's computation, so it changes the output. Unlike speculative
decoding, it is not exact.

Measured on Qwen3-30B-A3B Q4_K_M split across two PCs (16 GB desktop running layers 0..35, 8 GB
laptop running 35..48, fp8 activations). Quality is `spindle --ppl tools/ppl_sample.txt` (883
tokens of prose and code; lower is better). Speed is `spindle --bench`: the time per forward
pass is the fair comparison, because tok/s also depends on how many drafts a reply accepts.

| Setting | Perplexity | vs all 8 | ms per pass | prose tok/s |
|---|---|---|---|---|
| all 8 experts (`0`) | 6.357 | — | 180.6 | 7.0 |
| `0.1` | 6.557 | +3.1% | 157.0 | 7.2 |
| top 6 (`6`) | 6.654 | +4.7% | 156.9 | 7.8 |
| top 4 (`4`) | 8.169 | +28.5% | 130.1 | 9.1 |
| top 4 + `--resident-experts` | 8.520 | +34.0% | 128.2 | 9.1 |

- **`0.1` is the recommended setting**, and the one in the shipped `spindle.toml`. It costs 3%
  in perplexity and makes each pass 13% faster than running all 8 experts. It is as fast as
  top 6 and loses less.
- **Top 4 is a noticeable loss** (+28.5%) for a further 17% of speed. Use it only when speed is
  all that matters.
- Running half the experts costs much more than the router weights suggest, so measure any
  new setting with `--ppl` before relying on it.

The 0.1 and 6 rows come from single benchmark runs, so their tok/s differ by noise. Their time
per pass is the same.

Also checked: a unit test (`test_dynamic_moe_pruning`) runs a small random layer with and
without pruning and checks that the output stays finite.

## Where it helps

Expert weights are read straight from the memory-mapped file. A pruned expert is never
touched, and when the model streams from disk, it is never prefetched either. With 4 of 8
experts, a Qwen3-30B-A3B token reads roughly a quarter fewer bytes. The experts are about
1.8 of the 3.3 billion weights a token uses. How much of that becomes speed depends on what
else the step waits for, such as attention, the LM head and the network. On the two-PC setup
most of the saving lands on the desktop worker: its time per pass went from 117 ms with all
8 experts to 103 ms at `0.1` and 81 ms at top 4.
