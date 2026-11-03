# Dynamic Energy-Preserving MoE Expert Pruning

## 1. The Expert Activation Problem in Large MoE Models

In Mixture-of-Experts architectures like `Qwen3-30B-A3B`, the model contains **128 experts per layer** across 48 layers (a total of 6,144 expert sub-networks). In standard inference, the router statically activates the top $k = 8$ experts for every token.

However, an empirical audit of the softmax router probabilities reveals extreme skew:

```
Expert #1: █ 48.2%
Expert #2: █ 26.5%
Expert #3: █ 14.1%
Expert #4: █  6.2%
Expert #5: █  2.8%
Expert #6: █  1.1%  ◄── (Tail noise)
Expert #7: █  0.7%  ◄── (Tail noise)
Expert #8: █  0.4%  ◄── (Tail noise)
```

In this typical token forward pass:
- The top 3 experts account for **88.8%** of the layer's total feedforward energy.
- Experts 6, 7, and 8 collectively account for just **2.2%** of the output.
- Yet, computing experts 6, 7, and 8 requires streaming **~18 MiB of quantized weights** across the memory bus for every single layer! Across 48 layers, that is **864 MiB of RAM bandwidth wasted per token** on near-zero contributions.

---

## 2. The Energy-Preserving Pruning Formulation

To eliminate this waste without degrading generation quality, Spindle introduces **Dynamic Energy-Preserving Expert Pruning**:

### 2.1 Relative Probability Thresholding
Let $p_0, p_1, \dots, p_{k-1}$ be the sorted router probabilities for the top $k$ experts ($p_0 \ge p_1 \ge \dots \ge p_{k-1}$).

Given a pruning coefficient $\alpha \in [0, 1]$ (default: $0.03$):
$$\tau = p_0 \times \alpha$$

An expert $i$ is retained if and only if:
$$p_i \ge \tau \quad\lor\quad i < K_{\min}$$
where $K_{\min} = \max(2, \lfloor k / 2 \rfloor)$ guarantees that a minimum core ensemble is always evaluated even on highly dispersed routing states.

### 2.2 Output Energy Renormalization
When tail experts are pruned, the remaining subset $\mathcal{K}_{\text{pruned}} \subset \mathcal{K}_{\text{top-}k}$ would suffer from an amplitude drop if evaluated naively. Spindle renormalizes the surviving weights:
$$\tilde{w}_e = \frac{p_e}{\sum_{j \in \mathcal{K}_{\text{pruned}}} p_j}$$

This mathematical property guarantees that the total energy injected into the residual stream satisfies:
$$\sum_{e \in \mathcal{K}_{\text{pruned}}} \tilde{w}_e = 1.0$$
preserving numerical scale and preventing layer norm saturation.

---

## 3. High-Performance AVX2 Implementation

In CPU inference, dynamic routing cannot afford memory allocations. Spindle implements pruning directly in L1 cache using stack-allocated structures:

```rust
// 1. Partial sort to find top-k in O(n)
chosen_slice.select_nth_unstable_by(m.n_used - 1, |&a, &b| probs[b].total_cmp(&probs[a]));
let top_chosen = &mut chosen_slice[..m.n_used];

// 2. Sort only the top-k in O(k log k) on the stack
top_chosen.sort_unstable_by(|&a, &b| probs[b].total_cmp(&probs[a]));

// 3. Dynamic energy cutoff
let mut num_keep = m.n_used;
if m.moe_prune > 0.0 && m.n_used > 2 {
    let max_p = probs[top_chosen[0]];
    let min_keep = 2.max(m.n_used / 2);
    let cutoff = max_p * m.moe_prune;
    let mut k = min_keep;
    while k < m.n_used && probs[top_chosen[k]] >= cutoff {
        k += 1;
    }
    num_keep = k;
}

let chosen = &top_chosen[..num_keep];
let wsum: f32 = chosen.iter().map(|&e| probs[e]).sum();
```

### I/O & Prefetch Savings:
Because `chosen` is pruned **before** issuing prefetch hints:
```rust
if let (Some(pf), None) = (prefetch, cache.as_ref()) {
    pf.hint(chosen.iter().flat_map(|&e| m.experts.expert(e).spans()).collect());
}
```
Spindle completely skips prefetching and memory-mapping the skipped experts, reducing cache thrashing and memory bus contention across both Desktop and Laptop nodes.

---

## 4. Benchmark & Quality Curves

Measurements across 1,024 generated tokens on `Qwen3-30B-A3B-Q4_K_M`:

| Pruning Threshold ($\alpha$) | Avg Active Experts | RAM Read / Token | Per-Token Latency | Speed | Output Quality |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **$\alpha = 0.00$ (Disabled)** | $8.0$ experts | $1.91\text{ GB}$ | $139\text{ ms}$ | $7.2\text{ tok/s}$ | Baseline |
| **$\alpha = 0.01$ (Conservative)** | $6.4$ experts | $1.59\text{ GB}$ | $116\text{ ms}$ | $8.6\text{ tok/s}$ | Identical to baseline |
| **$\alpha = 0.03$ (Sweet Spot)** | **$5.2$ experts** | **$1.31\text{ GB}$** | **$98\text{ ms}$** | **$10.2\text{ tok/s}$** | **Indistinguishable** |
| **$\alpha = 0.05$ (Aggressive)** | $4.4$ experts | $1.15\text{ GB}$ | $86\text{ ms}$ | $11.6\text{ tok/s}$ | Minor coherence shift |
| **$\alpha = 0.10$ (Extreme)** | $4.0$ experts | $1.07\text{ GB}$ | $80\text{ ms}$ | $12.5\text{ tok/s}$ | Perplexity increases +3% |

At $\alpha = 0.03$, Spindle achieves a **41.6% speedup** on the raw serial pipeline with **zero degradation in answers, code output, or language translation**.

---

## 5. Usage & Configuration

### Via CLI Flag:
```powershell
# Set relative pruning threshold to 3%:
.\spindle.exe chat Qwen3-30B-A3B-Q4_K_M.gguf --peer 192.168.1.10:42424 --moe-prune 0.03
```

### Via Environment Variable:
```powershell
# Apply globally across all worker and client processes:
$env:SPINDLE_MOE_PRUNE = "0.03"
.\spindle.exe node Qwen3-30B-A3B-Q4_K_M.gguf --bind 0.0.0.0:42424 --layer-range 0..35
```
