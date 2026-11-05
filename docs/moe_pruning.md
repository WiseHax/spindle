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

Measured with `spindle --ppl tools/ppl_sample.txt` (883 tokens of prose and code) on
Qwen3-30B-A3B Q4_K_M, split across two PCs with fp8 activations. Lower perplexity is better:

| Setting | Perplexity | vs all 8 experts |
|---|---|---|
| all 8 experts (`moe_prune = 0`) | 6.357 | — |
| top 4 (`moe_prune = 4`) | 8.169 | +28.5% |
| top 4 with `--resident-experts` | 8.520 | +34.0% |

Running half the experts costs noticeably more quality than the router weights suggest. For
work where answers matter (code, reasoning, tool use), keep all experts unless speed is the
only goal. A fraction such as `0.1` drops experts only when the router concentrates its weight,
which should cost less. Measure it with `--ppl` the same way before relying on it.

Also checked: a unit test (`test_dynamic_moe_pruning`) runs a small random layer with and
without pruning and checks that the output stays finite.

## Where it helps

Expert weights are read straight from the memory-mapped file. A pruned expert is never
touched, and when the model streams from disk, it is never prefetched either. With 4 of 8
experts, a Qwen3-30B-A3B token reads roughly a quarter fewer bytes. The experts are about
1.8 of the 3.3 billion weights a token uses. How much of that becomes speed depends on what
else the step waits for, such as attention, the LM head and the network. On the two-PC setup,
top 4 on both PCs ran at 9.1 tok/s on prose. The same benchmark with all 8 experts on both
PCs hasn't been measured yet.
