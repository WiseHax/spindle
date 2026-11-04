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

Pruning changes the model's computation, so it changes the output. The kept experts are the
ones the router weighted most, so the change is usually small, but it is not zero. Unlike
speculative decoding, pruning is not exact.

Spindle 1.0 has no perplexity or benchmark-score measurements for pruning yet. What has been
checked:

- A unit test (`test_dynamic_moe_pruning`) runs a small random layer with and without
  pruning and checks that the output stays finite. It does not measure quality.
- Qwen3-30B-A3B with `moe_prune = 4` holds normal multi-turn conversations and tool use in
  daily use on the two-PC setup.

If answers look off, turn it off (`/moe-prune 0`) and compare. A fraction such as `0.05` drops
experts only when the router really concentrates its weight, so it is the more cautious
setting.

## Where it helps

Expert weights are read straight from the memory-mapped file. A pruned expert is never
touched, and when the model streams from disk, it is never prefetched either. With 4 of 8
experts, a Qwen3-30B-A3B token reads roughly a quarter fewer bytes. The experts are about
1.8 of the 3.3 billion weights a token uses. How much of that becomes speed depends on what else the step waits for, such as attention, the LM head and the
network. On the two-PC setup it is one of the settings behind the 7–8 tok/s figure in the
README. A careful measurement with and without pruning is on the roadmap.
