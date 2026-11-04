# Speculative decoding

Generating one token reads every active weight once. On a CPU, most of that time goes into
moving weights from RAM, not into the math. If the engine can guess the next few tokens, it
can check all of them in a single pass over the weights. When the guesses are right, it gets
several tokens for roughly the price of one.

Spindle does this without a draft model by default, and it never changes the output.

## Where the drafts come from

Spindle keeps the token ids of the recent conversation: the system prompt, messages, replies
and tool output. Before each step it takes the last 4 tokens it produced and searches for an
earlier place where the same 4 tokens appear. If it finds one, the tokens that followed there
become the draft. If there is no 4-token match, it tries 3, then 2. A 2-token match is weak
evidence, so it proposes at most 2 tokens from it.

This is prompt-lookup decoding. It works well when the reply repeats text that is already in
the context:

- editing a file the model just read
- quoting code, logs or earlier answers
- reformatting or translating structured text
- tool calls that repeat paths and names

On fresh prose, matches are rare and most steps have no draft at all. Those steps cost the
same as normal decoding.

The draft length adapts to how well drafts are doing:

- It starts at `draft_len` (4 by default, `--draft-len`).
- After a fully accepted draft, the next one may be 2 tokens longer, up to 7. Seven drafts
  plus the pending token fill one 8-token kernel pass.
- After any rejection, it goes back to `draft_len`.
- When a weak 2-token draft is rejected outright, weak drafts are skipped for a while: 1 step,
  then 2, 4 and so on, up to 16. Text with no repetition then stops paying for verify batches.
  Matches of 3 or more tokens are always tried.

`--draft <small.gguf>` uses a small model with the same tokenizer to write the drafts instead,
for example Qwen3-0.6B for Qwen3-30B-A3B. This costs the draft model's RAM and its own decode
time on every step. Prompt lookup costs neither, so it is the default.

## Verification

Each step runs one batch through the model: the pending token plus the draft tokens.

```
batch:      [ t   d1  d2  d3  d4 ]     t = last accepted token, d = drafts
positions:   p   p+1 p+2 p+3 p+4
```

The batch goes through the same batched forward pass that processes prompts. On two PCs that
is one `PREFILL_REQ` round trip. The pass returns a hidden state for every position, and from
each one spindle samples a token with the normal sampler and the same random number
generator:

1. Sample from position `p`. If the token equals `d1`, accept it and go on to position `p+1`.
2. At the first position where the sampled token differs from the draft, keep the sampled
   token, which is the model's own choice, and stop.
3. If every draft matched, the last position yields one extra token for free.

Every token that reaches the output was sampled by the target model, from the same
distribution and in the same order as without speculation. With `--temp 0`, the reply is
identical token for token. With sampling on, it follows the same distribution. Turning
speculation off (`--no-spec`, `/spec off`) never changes the quality, only the speed.

Rejected draft positions need no cleanup. Their KV entries are past the accepted position,
attention reads only up to the current position, and the next pass overwrites them. On two
PCs the worker needs no message for this either.

## What a batch costs

A batch is only cheap if processing 5 tokens costs much less than 5 single-token passes.
The multi-vector kernels in `src/quant_multi.rs` decode each quantized block once and apply it
to every token in the batch. Single-thread cost of a 5-token batch against one token:

| Weights | 5 tokens vs 1 |
|---|---|
| Q4_K | 1.78x |
| Q6_K | 1.36x |
| Q6_K LM head | 1.37x |
| Q8_0 | 1.61x |

Separate passes would cost 5x. MoE layers group the tokens by the experts they chose, so an
expert used by several tokens in the batch is read once.

## Results

| Case | Without | With |
|---|---|---|
| Qwen3-0.6B, a copy task (reply repeats text from the prompt) | 14.7 tok/s | 32.5 tok/s, same reply |
| SmolLM2-360M, a fresh paragraph of prose | 56.6 tok/s | 48.4 tok/s (0 of 11 drafts accepted) |
| Granite 1B-A400M, a fresh paragraph of prose | | 5 of 11 drafts accepted |

The prose rows show the cost of a step whose draft is rejected: the batch is a little more
expensive than a single token. Over a whole agent session, with file reads, edits and tool
output, the copy-like steps dominate. On prose-only chats where every millisecond counts, use
`--no-spec`.

The stats line after each reply shows how many drafts were accepted, for example
`speculation: 41/52 drafts accepted (79%), 18 verify steps`.

## Settings

| | |
|---|---|
| `speculate = true` / `--no-spec` | on by default |
| `draft_len = 4` / `--draft-len n` | starting draft length (1..16) |
| `/spec on`, `/spec off`, `/spec <n>` | change it during a chat (full-screen interface) |
| `/draft-len <n>` | change it during a chat (line-based interface) |
| `--draft <model.gguf>` | use a draft model instead of prompt lookup |
