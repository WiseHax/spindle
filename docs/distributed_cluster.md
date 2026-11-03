# Distributed Cluster Inference (Spindle RIOT-Wire)

`spindle` implements **Spindle RIOT-Wire**, a pure Rust, zero-syscall, zero-dependency distributed pipeline parallelism protocol designed to pool RAM across multiple consumer PCs over standard Wi-Fi or Ethernet.

---

## 1. The Core Problem: The Memory & VRAM Wall

State-of-the-art open-source LLMs (such as `Qwen3-30B-A3B` with 30.5 billion parameters) require **17.7 GiB of weights** in 4-bit quantization (`Q4_K_M`), plus KV cache.

- On a single 8 GB laptop:
  - 17.7 GiB model > 8 GB physical RAM.
  - The OS must swap pages from an NVMe SSD or stream layers on demand.
  - Result: **0.1 to 0.2 tok/s** (10 to 15 seconds per token).
- On expensive hardware:
  - Requires a \$2,000+ NVIDIA RTX 3090/4090 (24 GB VRAM) or an Apple Mac Studio with unified memory.

**Spindle RIOT-Wire solves this without GPUs:**
By clustering an **8 GB Laptop** with a **16 GB Desktop PC**, the combined physical RAM becomes **24 GB**, allowing the entire 30-Billion model to stay 100% resident in RAM across the local network.

---

## 2. Distributed Architecture Topology

```
                  ┌──────────────────────────────────────────────┐
                  │          Wi-Fi / Ethernet LAN Router         │
                  └───────────────▲───────────────┬──────────────┘
                                  │               │
                 StepRequest (8B) │               │ StepResponse (FP8: 4KB)
                                  │               │
       ┌──────────────────────────┴────┐     ┌────┴──────────────────────────┐
       │   Master Node (Laptop Client) │     │  Worker Node (Desktop Worker) │
       │   IP: 192.168.1.40            │     │  IP: 192.168.1.10:42424       │
       │   RAM: 8 GB Physical          │     │  RAM: 16 GB Physical          │
       ├───────────────────────────────┤     ├───────────────────────────────┤
       │ - CLI Chat / REPL             │     │ - Headless worker daemon      │
       │ - Embeddings (Stage 0)        │     │ - Layers 0..35 resident in RAM│
       │ - Layers 35..48 resident      │     │ - Multi-threaded AVX2 GEMV    │
       │ - LM Head & Sampler           │     │ - Sub-block presum dot kernel │
       │ - Output display (streaming)  │     │ - Zero-copy socket buffer     │
       └───────────────────────────────┘     └───────────────────────────────┘
```

---

## 3. Wire Protocol Specification (`src/cluster_net.rs`)

### Binary Frame Layout

All communication uses a compact 16-byte aligned binary wire frame (`WireHeader`):

```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                  Magic: 0x5350494E ('SPIN')                   |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|   Frame Type  |  Wire Format  |             Flags             |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                          Payload Length                       |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                      FNV1a-32 Checksum                        |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                       Payload Data [...]                      |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

### Frame Types

| ID | Name | Description |
|---|---|---|
| `0x01` | `FRAME_HANDSHAKE` | Client advertises model name, dimension, total layers, and local layer partition |
| `0x02` | `FRAME_HANDSHAKE_ACK` | Worker confirms model match, dimensions, and remote layer partition |
| `0x03` | `FRAME_STEP_REQ` | Client sends token ID and position: `[token_id: u32, pos: u32]` (8 bytes) |
| `0x04` | `FRAME_STEP_RESP` | Worker returns intermediate hidden activation vector across its layers |
| `0x05` | `FRAME_PREFILL_REQ` | Multi-token batched prompt sequence for layer-synchronous prompt prefilling |
| `0x06` | `FRAME_PREFILL_RESP` | Slices of intermediate hidden states for all prefilled prompt tokens |
| `0x07` | `FRAME_PING` | High-precision LAN latency measurement probe |
| `0x08` | `FRAME_PONG` | Ping reply with nanosecond timestamp |
| `0x09` | `FRAME_RESET` | Conversation reset: clears KV caches across the cluster in lockstep |
| `0xFF` | `FRAME_SHUTDOWN` | Clean disconnect notification |

---

## 4. Tensor Wire Compression Formats

Intermediate transformer activations are high-dimensional float vectors ($D = 4096$ in Qwen3-30B).

1. **`FP32` (Uncompressed)**:
   - Size: $4096 \times 4\text{ bytes} = \mathbf{16,384\text{ bytes}}$ per token.
   - Raw floating-point fidelity, suitable for 10GbE / InfiniBand.
2. **`FP16` (IEEE-754 Half Precision)**:
   - Size: $4096 \times 2\text{ bytes} = \mathbf{8,192\text{ bytes}}$ per token.
   - Vectorized conversion via `_mm256_cvtps_ph` and `_mm256_cvtph_ps` in < 250 ns.
3. **`FP8` (Adaptive Symmetric Quantization)**:
   - Size: $4\text{ bytes (f32 scale)} + 4096\text{ bytes (int8)} = \mathbf{4,100\text{ bytes}}$ per token (**75% network reduction**).
   - Dynamic scaling factor $s = \frac{\max(|x|)}{127.0}$.
   - Negligible perplexity delta ($< 0.02\%$), but cuts Wi-Fi transmission time in half.

---

## 5. Deployment Commands

### On the Worker Node (e.g., 16 GB Desktop PC):
```powershell
.\spindle.exe node --bind 0.0.0.0:42424 --layer-range 0..35 --wire-format fp8 --threads 4 Qwen3-30B-A3B-Q4_K_M.gguf
```

### On the Master Node (e.g., 8 GB Laptop):
```powershell
.\target\release\spindle.exe --peer 192.168.1.10:42424 --layer-range 35..48 --wire-format fp8 --threads 4 Qwen3-30B-A3B-Q4_K_M.gguf --chat
```

---

## 6. Resilience & Production Features

1. **Persistent Connections (No Idle Timeouts)**:
   - Sockets remain connected indefinitely across long human chat pauses (`read_timeout = None`).
2. **Zero-Allocation Socket Buffers**:
   - `recv_frame_into` reuses pre-allocated heap buffers (`scratch_payload`) across all tokens. Zero `malloc`/`free` calls during generation.
3. **Graceful Worker Reconnect & Re-binding**:
   - Worker automatically resets cleanly on client exit without hanging or leaking port 42424.
