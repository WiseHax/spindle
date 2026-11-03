# SIMD Micro-Kernels & CPU Acceleration Architecture

`spindle` executes quantized large language models without external BLAS libraries, CUDA, or PyTorch. All matrix-vector kernels are hand-crafted pure Rust assembly intrinsics targeting x86-64 `AVX2`, `FMA`, and `F16C`.

---

## 1. Intel Golden Cove / Raptor Lake Execution Pipeline

Modern Intel Core P-cores (e.g., i5-13420H / i7 / i9) contain **two independent 256-bit FMA execution ports**:
- **Port 0**: Supports 256-bit `_mm256_fmadd_ps` (8 single-precision float operations per cycle).
- **Port 1**: Supports 256-bit `_mm256_fmadd_ps` (8 single-precision float operations per cycle).

Each FMA instruction has a latency of **4 CPU cycles** and an issue rate of **1 cycle per port**.

### The Dependency Stall Problem
In naive single-row accumulation:
```c
acc = _mm256_fmadd_ps(weight, activation, acc);
```
Instruction $N+1$ cannot begin until instruction $N$ finishes 4 cycles later. This leaves Port 1 idle and stalls Port 0 for 75% of execution time!

---

## 2. Dual-Port Interleaved 2-Row GEMV (`rows_2x!`)

Spindle pairs matrix rows $(2i, 2i+1)$ into an interleaved micro-kernel:

```
Cycle 0:  FMA(Port 0, Row 0, Chunk 0, acc0)  |  FMA(Port 1, Row 1, Chunk 0, acc1)
Cycle 1:  FMA(Port 0, Row 0, Chunk 1, acc0)  |  FMA(Port 1, Row 1, Chunk 1, acc1)
Cycle 2:  FMA(Port 0, Row 0, Chunk 2, acc0)  |  FMA(Port 1, Row 1, Chunk 2, acc1)
Cycle 3:  FMA(Port 0, Row 0, Chunk 3, acc0)  |  FMA(Port 1, Row 1, Chunk 3, acc1)
```

### Key Advantages:
1. **100% Execution Port Saturation**: Both Port 0 and Port 1 execute at peak capacity on every cycle.
2. **Shared L1 Activation Loads**: Input vector slices $x[0..32]$ are loaded once into YMM registers and reused across both rows. Memory traffic is halved.
3. **Supported Dual-Row Formats**:
   - `Q4_K`: `dot_q4_k_2rows_avx2_presum`
   - `Q6_K`: `dot_q6_k_2rows_avx2`
   - `Q8_0`: `dot_q8_0_2rows_avx2`
   - `Q5_K`: `dot_q5_k_2rows_avx2_presum`
   - `F16`: `dot_f16_2rows_avx2`
   - `F32`: `dot_f32_2rows_avx2`

---

## 3. Pre-computed Sub-block Sums for Q4_K / Q5_K

Quantized formats with asymmetric offsets (like `Q4_K` and `Q5_K`) compute:
$$\text{term} = d \cdot sc_k \cdot \left(\sum_{j=0}^{31} q_{k,j} \cdot x_{k,j}\right) - dmin \cdot mn_k \cdot \left(\sum_{j=0}^{31} x_{k,j}\right)$$

In traditional engines, $\sum_{j=0}^{31} x_{k,j}$ was recomputed inside the inner loop for **every single row** of the matrix (repeating millions of identical vector additions across 4,096 to 8,192 rows).

### Spindle Sub-block Optimization:
1. **One-Shot AVX2 Pre-sum (`compute_subblock_sums_32`)**:
   - Evaluates $S_k = \sum_{j=0}^{31} x_{k,j}$ once per matvec in < 150 ns.
2. **Eliminated Vector Registers**:
   - Frees 2 vector accumulator registers in the inner loop.
3. **8-Scalar Min Accumulation**:
   - Replaces 32 vector additions per block with 8 fast scalar FMAs:
   $$\text{total\_min} += dmin \cdot \sum_{k=0}^7 (mn_k \cdot S_{b \cdot 8 + k})$$

---

## 4. Dual-Row Q6_K Kernel (`dot_q6_k_2rows_avx2`)

In Qwen3-30B, **49 critical tensors** (attention outputs and projection layers) use `Q6_K`.

Each 256-weight block contains:
- `ql`: 128 bytes (low 4 bits of 256 weights).
- `qh`: 64 bytes (high 2 bits of 256 weights).
- `scales`: 16 signed 8-bit scale factors.
- `d`: 16-bit half-precision base scale.

Spindle unpacks 8 weights at a time in 32-bit integer SIMD lanes, interleaving row 0 and row 1:
```rust
// Load activations once for both rows
let y0 = _mm256_loadu_ps(yp.add(l));

// Unpack row 0 and row 1
let q1_0 = ...; // row 0 weights
let q1_1 = ...; // row 1 weights

// Interleaved FMAs across Port 0 & 1
acc0 = _mm256_fmadd_ps(w1_0, y0, acc0);
acc1 = _mm256_fmadd_ps(w1_1, y0, acc1);
```

---

## 5. Zero-Heap Allocation in MoE Router & Layer Forward

In multi-threaded CPU inference, calling `malloc` and `free` concurrently across physical cores causes severe thread contention in the Windows NT Heap manager (`RtlAllocateHeap` locks).

### Solution:
1. **Stack-Allocated Router Buffers**:
   - `stack_probs = [0.0f32; 256]` (512 bytes on stack).
   - `stack_chosen = [0usize; 256]` (1024 bytes on stack).
   - Zero heap allocation per layer.
2. **Resident MoE Scratch Buffer**:
   - Pre-allocated `moe_tmp: Vec<f32>` passed into `moe_forward`.
   - Eliminates over **140,000 heap allocations** per generation sequence.
   - All router probabilities and indices stay hot in **L1 Data Cache**.
