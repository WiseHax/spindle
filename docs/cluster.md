# Running a model across two PCs

A 4-bit Qwen3-30B-A3B is 17.3 GiB. That doesn't fit on an 8 GB laptop, and on a 16 GB
desktop it leaves no room for anything else. Together, though, the two machines have enough
RAM. Spindle splits the model by layers: the worker PC holds the first layers and the client
PC holds the rest plus the output head. Neither machine reads weights from disk while
generating.

```
 client (laptop, 8 GB)                         worker (desktop, 16 GB)
 layers 35..48, LM head, sampler,              layers 0..35
 chat interface and tools

   token ids, position  ───────────────────▶   embeds, runs layers 0..35
                                               (one batched pass for several tokens)
   runs layers 35..48,  ◀───────────────────   one hidden-state vector per token
   samples the next token                      (2,052 bytes each at fp8)
```

This is pipeline parallelism with a single request in flight. For each token, the two
machines take turns: the worker computes, then the client computes. The time per token is the
worker's time plus the client's time plus one network round trip. Splitting the model makes it
fit in RAM. It doesn't make it faster than a single machine that could hold all of it.

## Setting it up

1. Put the same GGUF file on both PCs. The worker checks that the hidden size matches and
   rejects a different model.
2. On the worker, allow `spindle.exe` through Windows Firewall on private networks. Then
   start it:

   ```powershell
   .\spindle.exe node Qwen3-30B-A3B-Q4_K_M.gguf --bind 0.0.0.0:42424 --layer-range 0..35 --wire-format fp8 --threads 5 --moe-prune 4 --ctx 8192
   ```

   It loads its layers, prints `Ready`, and waits for a client. When a client disconnects,
   it waits for the next one. You don't need to restart it.
3. On the client, set `peer`, `layer_range`, `wire_format` and `ctx` in `spindle.toml` (see
   the README), or pass the same values as flags, and run `spindle`.

The startup display on the client shows the worker's address, its layer range and the
measured round-trip time. If the worker can't be reached, the display shows the command to
start it and the port to open.

Both machines must run the same spindle version. Newer clients use frames that older workers
don't understand.

## Choosing the split

Give each machine as many layers as its RAM can hold with some room left over:

- Qwen3-30B-A3B has 48 layers of about 0.36 GiB each at Q4_K_M. Layers 0..35 are about
  12.7 GiB on the 16 GB desktop. Layers 35..48 plus the output head are about 4.8 GiB on the
  8 GB laptop.
- Both machines also need room for the KV cache. At 8,192 tokens, it is about 0.6 GiB for
  35 layers and 0.2 GiB for 13 layers.
- Watch the RAM numbers in Task Manager during the first reply. If either machine starts
  paging, move a layer or two to the other one, or lower `ctx`.

The machine that does more of the work should be the faster one. In the setup above, a token
takes about 73 ms on the desktop (35 layers, 6 cores) and about 31 ms on the laptop
(13 layers and the LM head, 4 cores). The two are close per layer, so moving layers changes
where the time goes more than how much time there is.

## Wire formats

What travels for each token is one hidden-state vector of `dim` floats. For Qwen3-30B-A3B,
`dim` is 2,048.

| `--wire-format` | Encoding | Bytes per token (dim 2,048) |
|---|---|---|
| `fp32` | raw little-endian f32 | 8,192 |
| `fp16` | IEEE half precision, converted with F16C | 4,096 |
| `fp8` | one f32 scale (`max abs / 127`), then one int8 per value | 2,052 |

`fp8` is a scaled 8-bit integer format, not one of the FP8 float formats. Rounding the
residual stream at the layer boundary changes the logits slightly, so greedy output can differ
from a single-machine run at near-ties. On a LAN the bytes are rarely the bottleneck: a 2 KB
payload is a fraction of a millisecond even over Wi-Fi. Use `fp16` if you want the output to
stay closer to a single-machine run.

## Protocol

Everything goes over one TCP connection, with `TCP_NODELAY` set. Each message is a 16-byte
header followed by a payload:

| Bytes | Field |
|---|---|
| 0..4 | magic `0x5350494E` ("SPIN"), little-endian |
| 4 | frame type |
| 5 | wire format (0 = fp32, 1 = fp16, 2 = fp8) |
| 6..8 | flags |
| 8..12 | payload length |
| 12..16 | FNV-1a checksum of the header fields |

| Type | Name | Direction | Payload |
|---|---|---|---|
| `0x01` | `HANDSHAKE` | client → worker | model name, `dim`, layer count, client layer range |
| `0x02` | `HANDSHAKE_ACK` | worker → client | the same fields for the worker |
| `0x03` | `PREFILL_REQ` | client → worker | start position and a batch of token ids |
| `0x04` | `PREFILL_RESP` | worker → client | one hidden state per token |
| `0x05` | `STEP_REQ` | client → worker | one token id and its position |
| `0x06` | `STEP_RESP` | worker → client | one hidden state |
| `0x07` | `RESET` | client → worker | clear the KV cache (new conversation) |
| `0x08` | `PING` | client → worker | round-trip measurement |
| `0x09` | `PONG` | worker → client | |
| `0x0A` | `SET_PRUNE` | client → worker | MoE pruning setting, so both halves prune the same way |
| `0x0B` | `KV_SHIFT` | client → worker | `n_keep`, `n_discard`, client position; see below |
| `0x0C` | `KV_INFO` | both | the worker's context capacity in tokens |
| `0xFF` | `SHUTDOWN` | client → worker | clean disconnect |

Prompt processing and speculative verification both use `PREFILL_REQ`, so verifying a batch
of draft tokens costs one round trip, not one per token.

Frame types are only ever added. Existing types never change meaning, so a mismatched version
fails with a clear message instead of misreading data.

### Keeping the two KV caches in step

Each machine stores the keys and values for its own layers, indexed by token position. The
two caches must always describe the same positions.

- **Context window full.** The client decides how many old tokens to drop (it keeps the system
  prompt) and sends `KV_SHIFT` with the exact numbers. The worker applies the same shift and
  replies. Before 1.0, the worker shifted on its own with different numbers, and the two halves
  disagreed after the first shift.
- **Context length.** At connect time the client asks for the worker's capacity with `KV_INFO`
  and uses the smaller of the two. The worker also refuses any position outside its cache and
  reports it as a version mismatch, instead of writing past the end.
- **Rejected draft tokens.** Positions after the rejected draft are overwritten by the next
  pass. Attention only reads positions up to the current one, so no message is needed.

## Troubleshooting

| What you see | Likely cause |
|---|---|
| `could not reach the worker` with `actively refused` | worker not running, wrong port, or a different `--bind` |
| `could not reach the worker`, timing out | firewall on the worker, or the PCs are on different networks |
| worker prints `Rejected: ... hidden size` | the PCs loaded different GGUF files |
| worker prints `client asked for KV positions ...` | the PCs run different spindle versions |
| client says the worker did not report its context size | the worker runs a version older than 1.0 |
| first reply very slow, later ones normal | the worker was still reading weights; wait for its `Ready` |
