//! The agent loop shared by the TUI and the line-mode CLI.
//!
//! One user turn = generate → parse every `<tool_call>` → run the tools → feed all results
//! back as one `<tool_response>` turn (Qwen's native format) → repeat until the model answers
//! without calling a tool (or the step limit is hit). The UI follows along through
//! [`AgentEvent`]s; risky actions go through the caller's `confirm` callback.

use std::{
    collections::HashMap,
    sync::atomic::{AtomicBool, Ordering},
    time::Instant,
};

use crate::{
    smollm_chat::SmolLm2Engine,
    tools::{self, ToolCall, ToolCtx, ToolOutcome},
};

#[derive(Clone, Copy, Debug)]
pub struct AgentOpts {
    /// Token budget for each model reply.
    pub max_tokens: usize,
    pub allow_think: bool,
    /// Tool rounds per user turn before the model is told to answer.
    pub max_steps: usize,
}

impl Default for AgentOpts {
    fn default() -> Self {
        Self { max_tokens: 1024, allow_think: false, max_steps: 24 }
    }
}

pub enum AgentEvent {
    /// Raw streamed text of the reply being generated (may include `<tool_call>` markup).
    Token(String),
    /// A reply finished; `visible` is its text with tool-call blocks removed.
    ReplyDone { visible: String },
    /// A tool is about to run.
    ToolStart { icon: &'static str, title: String },
    /// A tool finished.
    ToolEnd(ToolOutcome),
}

#[derive(Default, Clone, Copy, Debug)]
pub struct TurnStats {
    pub tokens: usize,
    pub gen_secs: f64,
    pub wall_secs: f64,
    pub tool_calls: usize,
    pub spec_accepted: usize,
    pub spec_drafted: usize,
    pub round_trips: u64,
    pub worker_ms: f64,
    pub local_ms: f64,
    pub aborted: bool,
    /// Where the generation time went, averaged per forward pass.
    pub breakdown: Breakdown,
}

/// Average time per forward pass (one token, or one verify batch), in milliseconds.
#[derive(Default, Clone, Copy, Debug)]
pub struct Breakdown {
    pub passes: u64,
    pub wall_ms: f64,
    /// Waiting for the worker (its compute plus the network).
    pub worker_ms: f64,
    /// The worker's own compute, when it reports it.
    pub worker_compute_ms: Option<f64>,
    pub local_ms: f64,
    pub head_ms: f64,
    pub sample_ms: f64,
    pub draft_ms: f64,
    pub emit_ms: f64,
    /// Prompt processing speed for this turn (tokens per second).
    pub prefill_tok_s: Option<f64>,
}

impl Breakdown {
    pub fn from_timing(t: &crate::smollm_chat::StepTiming, gen_secs: f64) -> Self {
        let n = t.passes.max(1) as f64;
        let ms = |us: u64| us as f64 / 1e3 / n;
        Breakdown {
            passes: t.passes,
            wall_ms: gen_secs * 1e3 / n,
            worker_ms: ms(t.worker_us),
            worker_compute_ms: (t.worker_requests > 0).then(|| t.worker_compute_us as f64 / 1e3 / t.worker_requests as f64),
            local_ms: ms(t.local_us),
            head_ms: ms(t.head_us),
            sample_ms: ms(t.sample_us),
            draft_ms: ms(t.draft_us),
            emit_ms: ms(t.emit_us),
            prefill_tok_s: (t.prefill_us > 0 && t.prefill_tokens > 0).then(|| t.prefill_tokens as f64 / (t.prefill_us as f64 / 1e6)),
        }
    }

    /// Time per pass not covered by any measured part.
    pub fn other_ms(&self) -> f64 {
        (self.wall_ms - self.worker_ms - self.local_ms - self.head_ms - self.sample_ms - self.draft_ms - self.emit_ms).max(0.0)
    }

    /// Multi-line table for `/timing` and `--bench`.
    pub fn table(&self, tokens: usize) -> String {
        if self.passes == 0 {
            return "No generation yet.".into();
        }
        let per_tok = if tokens > 0 { self.passes as f64 / tokens as f64 } else { 1.0 };
        let mut rows: Vec<(String, f64)> = Vec::new();
        if self.worker_ms > 0.0 {
            match self.worker_compute_ms {
                Some(c) => {
                    rows.push(("worker compute".into(), c));
                    rows.push(("network".into(), (self.worker_ms - c).max(0.0)));
                }
                None => rows.push(("worker + network".into(), self.worker_ms)),
            }
        }
        rows.push((if self.worker_ms > 0.0 { "local layers".into() } else { "layers".into() }, self.local_ms));
        rows.push(("LM head".into(), self.head_ms));
        rows.push(("sampling".into(), self.sample_ms));
        rows.push(("drafts".into(), self.draft_ms));
        rows.push(("text + interface".into(), self.emit_ms));
        rows.push(("unaccounted".into(), self.other_ms()));
        let mut out = format!(
            "Per forward pass ({} passes, {:.2} tokens each): {:.1} ms\n",
            self.passes,
            1.0 / per_tok.max(1e-9),
            self.wall_ms
        );
        for (name, v) in rows {
            let pct = if self.wall_ms > 0.0 { v / self.wall_ms * 100.0 } else { 0.0 };
            out.push_str(&format!("  {name:<18} {v:>7.1} ms  {pct:>3.0}%\n"));
        }
        if let Some(p) = self.prefill_tok_s {
            out.push_str(&format!("Prompt processing: {p:.0} tok/s\n"));
        }
        if self.worker_ms > 0.0 && self.worker_compute_ms.is_none() {
            out.push_str("(the worker doesn't report its compute time; update it to split worker and network)\n");
        }
        out.trim_end().to_string()
    }
}

impl TurnStats {
    pub fn tok_per_sec(&self) -> f64 {
        if self.gen_secs > 0.0 {
            self.tokens as f64 / self.gen_secs
        } else {
            0.0
        }
    }
}

/// Icon and one-line title for a call before it runs (the UI shows it with a spinner).
pub fn call_title(call: &ToolCall) -> (&'static str, String) {
    let path = call.str_arg(&["path", "file_path"]).unwrap_or("");
    match call.name.as_str() {
        "read_file" | "read" => ("▸", format!("Read {path}")),
        "write_file" | "write" => ("✎", format!("Write {path}")),
        "edit_file" | "edit" => ("✎", format!("Edit {path}")),
        "delete_file" => ("✎", format!("Delete {path}")),
        "list_dir" | "ls" => ("▸", format!("List {}", if path.is_empty() { "." } else { path })),
        "glob" => ("⌕", format!("Glob \"{}\"", call.str_arg(&["pattern"]).unwrap_or(""))),
        "grep" | "search_symbols" => ("⌕", format!("Grep \"{}\"", call.str_arg(&["pattern", "query"]).unwrap_or(""))),
        "run_command" | "bash" | "shell" => ("❯", call.str_arg(&["command", "cmd"]).unwrap_or("").to_string()),
        "web_search" | "search_web" => ("◎", format!("Search \"{}\"", call.str_arg(&["query"]).unwrap_or(""))),
        "fetch_url" => ("⇣", format!("Fetch {}", call.str_arg(&["url"]).unwrap_or(""))),
        other => ("▪", other.to_string()),
    }
}

/// Characters of tool output the model can take right now: about a third of the free
/// context (≈3.5 chars/token), split across this round's calls.
fn output_budget(engine: &SmolLm2Engine, n_calls: usize) -> usize {
    let free_tokens = engine.max_seq.saturating_sub(engine.cur_pos);
    (free_tokens * 7 / 6 / n_calls.max(1)).clamp(1200, 24_000)
}

/// Runs one user turn to completion. `confirm` answers permission prompts; `abort` stops
/// generation and skips any remaining tools.
pub fn run_turn(
    engine: &mut SmolLm2Engine,
    prompt: &str,
    opts: AgentOpts,
    abort: Option<&AtomicBool>,
    on_event: &mut dyn FnMut(AgentEvent),
    confirm: &mut dyn FnMut(&str) -> bool,
) -> TurnStats {
    let t0 = Instant::now();
    let mut stats = TurnStats::default();
    let timing0 = engine.timing;
    let mut input = prompt.to_string();
    let mut seen: HashMap<String, usize> = HashMap::new();
    let aborted = || abort.is_some_and(|a| a.load(Ordering::Relaxed));

    for step in 0..=opts.max_steps {
        if aborted() {
            stats.aborted = true;
            break;
        }
        let (text, secs, toks) = unsafe {
            engine.generate_reply(&input, opts.max_tokens, opts.allow_think, &mut |piece| on_event(AgentEvent::Token(piece.to_string())))
        };
        stats.tokens += toks;
        stats.gen_secs += secs;
        stats.spec_accepted += engine.spec_stats.0;
        stats.spec_drafted += engine.spec_stats.1;

        let (calls, visible) = if engine.enable_tools { tools::parse_tool_calls(&text) } else { (Vec::new(), text.trim().to_string()) };
        on_event(AgentEvent::ReplyDone { visible });
        if calls.is_empty() || aborted() {
            stats.aborted |= aborted();
            break;
        }

        let budget = output_budget(engine, calls.len());
        let mut responses = Vec::with_capacity(calls.len());
        for call in calls.iter().take(8) {
            if aborted() {
                stats.aborted = true;
                break;
            }
            let (icon, title) = call_title(call);
            let n = seen.entry(call.signature()).or_insert(0);
            *n += 1;
            if *n > 2 {
                // A model stuck repeating itself: don't run it again, nudge it instead.
                let out = ToolOutcome {
                    ok: false,
                    icon,
                    title,
                    output: "Error: you already made this exact call twice and have its result above. Use that result, try a different approach, or answer the user.".into(),
                    preview: None,
                };
                on_event(AgentEvent::ToolEnd(out.clone()));
                responses.push(out.output);
                continue;
            }
            on_event(AgentEvent::ToolStart { icon, title });
            let mut ctx = ToolCtx { confirm: &mut *confirm, max_chars: budget };
            let out = tools::execute(call, &mut ctx);
            stats.tool_calls += 1;
            responses.push(out.output.clone());
            on_event(AgentEvent::ToolEnd(out));
        }
        if stats.aborted {
            break;
        }
        input = responses.iter().map(|r| format!("<tool_response>\n{}\n</tool_response>", r.trim_end())).collect::<Vec<_>>().join("\n");
        if step + 1 == opts.max_steps {
            input.push_str("\n\nTool-call limit for this turn reached. Do not call more tools; summarize what you did and what is left.");
        }
    }

    let t = engine.timing;
    stats.breakdown = Breakdown::from_timing(&t.since(&timing0), stats.gen_secs);
    stats.round_trips = t.steps - timing0.steps;
    if stats.round_trips > 0 {
        stats.worker_ms = (t.worker_us - timing0.worker_us) as f64 / 1e3 / stats.round_trips as f64;
        stats.local_ms = (t.local_us - timing0.local_us) as f64 / 1e3 / stats.round_trips as f64;
    }
    stats.wall_secs = t0.elapsed().as_secs_f64();
    stats
}
