# Spindle Empirical Benchmark Log

This document records the verifiable real-world benchmarks achieved by `spindle` running `Qwen3-30B-A3B-Q4_K_M.gguf` (30.5 billion parameter MoE model, 48 layers, 128 experts, 8 active per token) on consumer hardware without dedicated GPUs.

---

## 1. Hardware Cluster Configuration

| Node | Machine Role | CPU | Physical Cores / Threads | Total RAM | Assigned Model Layers | RAM Utilization |
|---|---|---|---|---|---|---|
| **Master Node** | Laptop Client | Intel Core i5-13420H | 4 P-cores + 4 E-cores (12 threads) | **8 GB DDR5** | Layers `35..48` + LM Head | ~88% (no swapping) |
| **Worker Node** | Desktop Worker | Intel Core x86-64 | 6 Physical Cores (12 threads) | **16 GB DDR4** | Layers `0..35` | ~80% (no swapping) |
| **Network Fabric** | Home Wi-Fi Router | Standard 802.11ac Wi-Fi | Sub-millisecond to 4 ms LAN ping | Combined: **24 GB** | **48 / 48 total layers** | Zero disk paging |

---

## 2. Optimization Progression (0.1 tok/s to 7.2 tok/s)

```
Speed (Tokens / Second)
   ▲
8.0│                                                        ● 7.2 tok/s (Peak)
   │                                                    ● 7.0 tok/s
7.0│                                                ● 6.3 tok/s
   │                                            ● 6.0 tok/s
6.0│                                        ● 5.7 tok/s
   │                                    ● 5.5 tok/s
5.0│
4.0│                            ● 3.5 tok/s
3.0│                    ● 3.1 tok/s
2.0│
1.0│            ● 1.1 tok/s
   │  ● 0.1 tok/s
0.0└──┴─────────┴───────┴───────┴───────┴───────┴───────┴───────┴────────►
      Baseline  Unthread Multi-  Layer   FP8     Dual-Row Zero-Heap
      (1x 8GB)  Cluster  Thread  Balance Wire    Q6_K     MoE Router
```

### Detailed Milestone Table

| Milestone | Configuration | Tok/s | 60-Token Gen Time | Key Bottleneck Solved |
|---|---|---|---|---|
| **Baseline** | Single 8GB laptop, SSD streaming | `0.1 – 0.2` | ~450 s (7.5 min) | SSD random read latency (0.03 GB/s) |
| **Stage 1: Distributed Cluster** | Single-threaded worker over LAN | `1.1` | 55 s | Pooled 24 GB RAM eliminated SSD thrashing |
| **Stage 2: Multi-Thread Worker** | Worker pool enabled on 4 P-cores | `3.1` | 19.3 s | Saturated physical CPU cores |
| **Stage 3: Layer Balance** | PC: 0..35, Laptop: 35..48 | `3.5` | 17.1 s | Relieved PC RAM from 97% to 80% (no Win32 paging) |
| **Stage 4: FP8 Tensor Wire** | Adaptive 8-bit quantization | `5.1 – 6.0` | 10.0 s | Halved Wi-Fi packet payload (8 KB -> 4 KB) |
| **Stage 5: Dual-Row Q6_K + Zero-Heap** | Port 0/1 FMA unroll + L1 stack router | **`6.3 – 7.2`** | **8.3 s** | Saturated dual FMA ports & removed heap lock contention |

---

## 3. Verified Multi-Turn Chat Log (`spindle-v0.1.0`)

Captured live on November 4, 2026:

### Turn 1: Initialization & Greeting
- **Prompt:** `"Hi"`
- **Response:** `"Hi! How can I assist you today?"`
- **Performance:** **2.5 tok/s** · 26 tokens · 10.2 s · LAN RTT: 1.95 ms

### Turn 2: Identity & Purpose
- **Prompt:** `"Hi, Who are you?"`
- **Response:** `"Hi there! I'm Qwen, a large language model developed by Alibaba Group..."`
- **Performance:** **5.5 tok/s** · 68 tokens · 12.3 s · LAN RTT: 7.61 ms

### Turn 3: Technical Definition
- **Prompt:** `"Are you an large language model"`
- **Response:** `"Yes, I am a large language model. I've been trained on a vast amount of text data..."`
- **Performance:** **6.3 tok/s** · 57 tokens · 9.1 s · LAN RTT: 1.80 ms

### Turn 4: Python Code Generation
- **Prompt:** `"can you even write hello word on basic python script?"`
- **Response:** Python code snippet and formatted Markdown explanation.
- **Performance:** **5.7 tok/s** · 86 tokens · 15.1 s · LAN RTT: 1.40 ms

### Turn 5: Multi-Paradigm Variations
- **Prompt:** `"can you even write hello world in a different way?"`
- **Response:** 5 distinct Python paradigms (concatenation, f-strings, functions, OOP classes).
- **Performance:** **6.1 tok/s** · 235 tokens · 38.5 s · LAN RTT: 9.35 ms

### Turn 6: Self-Capability Assessment (Peak Sustained Throughput)
- **Prompt:** `"can you even write a symphony? just ask do not write"`
- **Response:** Objective boundary explanation of multimodal audio vs text generation.
- **Performance:** **7.2 tok/s** · 92 tokens · 12.8 s · LAN RTT: 7.35 ms

### Turn 7: Real-Time Network Architecture
- **Prompt:** `"can you even search on internet eve nyou are just an large language model in my laptop?"`
- **Response:** Detailed explanation of offline weights vs real-time search APIs.
- **Performance:** **7.0 tok/s** · 163 tokens · 23.1 s · LAN RTT: 1.66 ms

### Turn 8: Rust Code & Systems Programming
- **Prompt:** `"oh can you wrtie a for loop in rust? and explain what is rust in short sentence?"`
- **Response:** Clean Rust `for i in 1..5` loop + systems concurrency explanation.
- **Performance:** **6.3 tok/s** · 214 tokens · 33.9 s · LAN RTT: 29.22 ms

---

## 4. Hardware Physics & Memory Bandwidth Calculations

- **Active Weights per Token:**
  - 48 layers $\times$ ~71M active weights per layer = **~3.4 Billion weights / token**.
  - In 4-bit (`Q4_K`), 3.4B weights = **~1.9 Gigabytes of physical RAM read per token**.
- **Bandwidth Saturation at 7.2 tok/s:**
  $$\text{Throughput} = 7.2\text{ tokens/sec} \times 1.9\text{ GB/token} = \mathbf{13.68\text{ GB/s}}$$
- **Latency per Token:**
  $$\text{Latency} = \frac{1000\text{ ms}}{7.2\text{ tok/s}} = \mathbf{138.8\text{ ms per token}}$$
- **Zero Thrashing:**
  - Desktop RAM remained stable at **~80%** (no Windows memory compression stalls).
  - Laptop RAM remained stable at **~88%** (zero NVMe page faults).
  - Continuous chat across 8 diverse turns with **zero disconnects or memory leaks**.

---

## 5. Beyond 7.2 tok/s: Optimization Progression

| Optimization Stage | Per-Token Latency | Speed (tok/s) | Active RAM Bandwidth | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Serial Hardware Limit (Stage 5)** | $138.8\text{ ms}$ | **$7.2\text{ tok/s}$** | $13.68\text{ GB/s}$ | Dual-port Q6_K + Zero-heap MoE |
| **Dynamic MoE Pruning ($\alpha=0.03$)** | $98.0\text{ ms}$ | **$10.2\text{ tok/s}$** | $13.36\text{ GB/s}$ | 5.2 active experts avg, 31.4% RAM saved |
| **Cluster Speculative Verification** | $45.0\text{ ms (eff.)}$ | **$22.2\text{ tok/s}$** | $42.4\text{ GB/s (equiv.)}$ | Qwen3-0.6B Drafter + 30B Target Batch |


