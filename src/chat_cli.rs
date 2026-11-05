//! The interactive chat: header, prompt, streamed replies and slash commands.

use std::io::{BufRead, Write};

use crate::{
    memory_guard::{self, Governor},
    smollm_chat::SmolLm2Engine,
    ui::{accent, bold, bold_accent, dim, green},
};

const HELP: &str = "\
Commands
  /help                     this list
  /clear                    start a new conversation
  /tools [on|off]           let the model read/edit files and run commands
  /think [on|off|show|hide] reasoning before answers, and whether it is shown
  /undo                     revert the last file the model changed
  /diff                     show uncommitted changes (git diff)
  /review                   ask the model to review the uncommitted diff
  /git <args>               run a git command
  /heal [test command]      run the tests and let the model fix failures (3 tries)
  /branch [save|revert|list] <name>
                            save or restore a point in the conversation
  /plan [on|off]            ask for a plan before any edits
  /init                     write a SPINDLE.md with project notes
  /compact                  summarize the conversation to free context
  /system [text|reset]      show or replace the system prompt
  /moe-prune [n]            MoE experts per token (n >= 1) or tail cutoff (n < 1)
  /draft-len [n]            draft tokens per speculative step (1..16)
  /cluster                  worker status and round-trip time
  /model                    model, memory and context details
  /bench [n]                measure generation speed (default 32 tokens)
  /stats                    toggle detailed timing after each reply
  /timing                   where the last reply's time went, per forward pass
  /max <n>                  maximum tokens per reply
  /exit                     quit";

const TOOLS_ON: &str = "Tools on: the model can read and edit files, search and run commands.";
const TOOLS_OFF: &str = "Tools off: plain chat.";
const PLAN_ON: &str = "Plan mode on: the model writes a plan before making changes.";
const PLAN_OFF: &str = "Plan mode off.";
const THINK_ON: &str = "Reasoning on: the model thinks before answering.";
const THINK_OFF: &str = "Reasoning off: direct answers.";

/// Prints model output as it streams. Reasoning between `<think>` and `</think>` is rendered
/// in a framed block, or collapsed to a one-line note.
struct Render {
    show_think: bool,
    in_think: bool,
    skip_newlines: bool,
    wrote: bool,
    think_start: Option<std::time::Instant>,
    line_has_prefix: bool,
    tool_filter: crate::tools::ToolMarkupFilter,
}

impl Render {
    fn new(show_think: bool) -> Self {
        Self {
            show_think,
            in_think: false,
            skip_newlines: true,
            wrote: false,
            think_start: None,
            line_has_prefix: false,
            tool_filter: Default::default(),
        }
    }

    fn push(&mut self, piece: &str) {
        let mut out = std::io::stdout();
        // Tool calls are shown as tool rows by the agent loop, not as raw JSON.
        let filtered = self.tool_filter.push(piece);
        let piece = filtered.as_str();
        if piece.is_empty() {
            return;
        }
        if piece.contains("<think>") {
            self.in_think = true;
            self.think_start = Some(std::time::Instant::now());
            self.skip_newlines = true;
            self.line_has_prefix = false;
            if self.show_think {
                let _ = writeln!(out, "{}", dim("┌ Thinking"));
            } else {
                let _ = write!(out, "{}", dim("Thinking…"));
            }
            let _ = out.flush();
            return;
        }
        if piece.contains("</think>") {
            self.in_think = false;
            self.skip_newlines = true;
            let elapsed = self.think_start.map_or(0.0, |t| t.elapsed().as_secs_f64());
            if self.show_think {
                if self.line_has_prefix {
                    let _ = writeln!(out);
                }
                let _ = writeln!(out, "{}", dim(&format!("└ Thought for {elapsed:.1}s\n")));
            } else {
                if crate::ui::color() {
                    let _ = write!(out, "\r\x1b[2K");
                }
                let _ = writeln!(out, "{}", dim(&format!("Thought for {elapsed:.1}s (/think show to see it)\n")));
            }
            let _ = out.flush();
            return;
        }

        if self.in_think {
            if self.show_think {
                let text = if self.skip_newlines { piece.trim_start_matches(['\n', '\r']) } else { piece };
                if !text.is_empty() {
                    self.skip_newlines = false;
                    for ch in text.chars() {
                        if !self.line_has_prefix {
                            let _ = write!(out, "{}", dim("│ "));
                            self.line_has_prefix = true;
                        }
                        if ch == '\n' {
                            let _ = writeln!(out);
                            self.line_has_prefix = false;
                        } else {
                            let _ = write!(out, "{}", dim(&ch.to_string()));
                        }
                    }
                }
            }
        } else {
            let text = if self.skip_newlines { piece.trim_start_matches(['\n', '\r']) } else { piece };
            if text.is_empty() {
                return;
            }
            self.skip_newlines = false;
            self.wrote = true;
            let _ = write!(out, "{text}");
        }
        let _ = out.flush();
    }
}

/// What a `moe_prune` value does, in words (integer = expert count, fraction = mass cut).
pub fn describe_prune(p: f32) -> String {
    if p <= 0.0 {
        "MoE pruning off: every routed expert runs".to_string()
    } else if p >= 1.0 {
        format!("MoE pruning: top {} experts per token (never below half of the routed experts)", p as usize)
    } else {
        format!("MoE pruning: keep experts covering {:.0}% of router probability (never below half)", (1.0 - p) * 100.0)
    }
}

pub struct ChatOptions {
    pub first_prompt: Option<String>,
    pub interactive: bool,
    pub threads: usize,
    pub max_tokens: usize,
    pub allow_think: bool,
    pub tui: bool,
    /// Startup phases already shown, `(label, result)`, repeated on the TUI loading screen.
    pub startup: Vec<(String, String)>,
}

/// One line for the line-based chat (the TUI draws its own header). The model, memory
/// and cluster details were already shown by the startup display.
pub fn print_header(_engine: &SmolLm2Engine, opts: &ChatOptions) {
    if opts.interactive && !opts.tui {
        println!("{}", dim("Type a message, or /help for commands."));
    }
}

fn stats_line(secs: f64, tokens: usize, engine: &mut SmolLm2Engine, governor: Option<&Governor>, detailed: bool) -> String {
    let mut parts = vec![format!("{:.1} tok/s", tokens as f64 / secs), format!("{tokens} tokens"), format!("{secs:.1} s")];
    if let Some(client) = engine.cluster_client.as_mut() {
        if let Ok(rtt) = client.ping() {
            parts.push(format!("worker {:.1} ms away", rtt.as_secs_f64() * 1000.0));
        } else {
            parts.push("worker".to_string());
        }
    } else if !engine.is_streaming() {
        parts.push("in RAM".to_string());
    } else {
        parts.push("streaming from disk".to_string());
    }
    if let Some((w_ms, l_ms)) = engine.take_cluster_timing() {
        if detailed {
            parts.push(format!("per token: worker {w_ms:.1} ms, local {l_ms:.1} ms"));
        }
    }
    if detailed {
        let (ws, private, _) = memory_guard::process_counters();
        parts.push(format!("{} MiB RAM ({} MiB private)", ws >> 20, private >> 20));
        if let Some(g) = governor {
            let ord = std::sync::atomic::Ordering::Relaxed;
            parts.push(format!("cap {} MiB", g.status.cap.load(ord) >> 20));
            parts.push(format!("{} MiB free", g.status.available.load(ord) >> 20));
        }
    }
    if let Some((mib, touched, total, timing)) = engine.take_moe_stats() {
        if detailed {
            parts.push(format!("{mib:.0} MiB experts/token, {touched}/{total} experts used"));
            let total_us = timing.router_us + timing.cache_lookup_us + timing.disk_read_us + timing.expert_matvec_us;
            if total_us > 0 {
                let r_ms = timing.router_us as f64 / 1000.0;
                let l_ms = timing.cache_lookup_us as f64 / 1000.0;
                let d_ms = timing.disk_read_us as f64 / 1000.0;
                let m_ms = timing.expert_matvec_us as f64 / 1000.0;
                let t_ms = total_us as f64 / 1000.0;
                parts.push(format!(
                    "moe: router {:.1}ms ({:.0}%), lookup {:.1}ms ({:.0}%), disk {:.1}ms ({:.0}%), matvec {:.1}ms ({:.0}%)",
                    r_ms, r_ms / t_ms * 100.0,
                    l_ms, l_ms / t_ms * 100.0,
                    d_ms, d_ms / t_ms * 100.0,
                    m_ms, m_ms / t_ms * 100.0,
                ));
            }
        }
    }
    if let Some(c) = engine.expert_cache.as_ref() {
        if detailed {
            parts.push(format!(
                "cache: {:.1}% hit ({}/{}), {:.1} MiB from disk",
                c.hit_rate(),
                c.hits,
                c.hits + c.misses,
                c.bytes_read as f64 / (1024.0 * 1024.0)
            ));
        }
    }
    let (acc, tot, b) = engine.spec_stats;
    if tot > 0 {
        parts.push(format!(
            "speculation: {}/{} drafts accepted ({:.0}%), {} verify step{}",
            acc,
            tot,
            (acc as f64 / tot as f64) * 100.0,
            b,
            if b == 1 { "" } else { "s" }
        ));
    }
    parts.join(" · ")
}

/// Fixed prompts for `--bench`: fresh prose, a copy-heavy code edit, and a list. Together they
/// show both the plain decode speed and what speculation adds when text repeats.
const BENCH_PROMPTS: &[(&str, &str)] = &[
    ("prose", "Write a short paragraph about how rivers shape valleys."),
    (
        "code edit",
        "Rename the variable `count` to `total` in this function and output only the full updated code:\n\n\
```rust\n\
fn summarize(values: &[i64]) -> (i64, usize, f64) {\n\
    let mut count = 0;\n\
    let mut sum = 0;\n\
    for v in values {\n\
        if *v >= 0 {\n\
            sum += v;\n\
            count += 1;\n\
        }\n\
    }\n\
    let mean = if count > 0 { sum as f64 / count as f64 } else { 0.0 };\n\
    println!(\"{count} non-negative values, sum {sum}, mean {mean:.2}\");\n\
    (sum, count, mean)\n\
}\n\
```",
    ),
    ("list", "List the eight planets of the solar system, one line each with one fact."),
];

/// `--bench`: runs the fixed prompts `runs` times each in this session, greedy, each from the
/// same starting point (the cached system prompt), and prints speed and a time breakdown.
pub fn run_bench(engine: &mut SmolLm2Engine, runs: usize, max_tokens: usize) {
    use crate::agent::Breakdown;
    unsafe { engine.prefill_system() };
    let saved_sampler = engine.sampler;
    let mut greedy = saved_sampler;
    greedy.temperature = 0.0;
    engine.set_sampler(greedy);
    engine.save_kv_branch("__bench");
    let corpus = engine.lookup_corpus.clone();
    let where_ = match engine.cluster_client.as_ref() {
        Some(c) => format!(
            "worker layers {}..{} + local layers {}..{}",
            c.worker_range.0, c.worker_range.1, engine.layer_range.start, engine.layer_range.end
        ),
        None => "single PC".to_string(),
    };
    println!("{}", bold(&format!("Benchmark: {} · {where_}", engine.idx.model_name)));
    println!(
        "{}",
        dim(&format!(
            "{} run(s) per prompt · greedy · up to {max_tokens} tokens · speculation {}",
            runs,
            if engine.speculate { "on" } else { "off" }
        ))
    );
    println!();
    let picks0 = all_expert_picks(engine);
    let mut all = crate::smollm_chat::StepTiming::default();
    let mut all_secs = 0.0;
    let mut all_tokens = 0usize;
    for (name, prompt) in BENCH_PROMPTS {
        let mut speeds = Vec::new();
        let mut accepted = (0usize, 0usize);
        for _ in 0..runs.max(1) {
            let _ = engine.restore_kv_branch("__bench");
            engine.lookup_corpus = corpus.clone();
            let t0 = engine.timing;
            let (_, secs, toks) = unsafe { engine.generate_reply(prompt, max_tokens, false, &mut |_| {}) };
            let d = engine.timing.since(&t0);
            all = add_timing(&all, &d);
            all_secs += secs;
            all_tokens += toks;
            accepted.0 += engine.spec_stats.0;
            accepted.1 += engine.spec_stats.1;
            speeds.push(toks as f64 / secs.max(1e-6));
        }
        speeds.sort_by(|a, b| a.total_cmp(b));
        let median = speeds[speeds.len() / 2];
        let spec = if accepted.1 > 0 { format!(" · drafts {}/{} accepted", accepted.0, accepted.1) } else { String::new() };
        println!(
            "  {:<10} {:>6.1} tok/s   (min {:.1}, max {:.1}){spec}",
            name,
            median,
            speeds[0],
            speeds[speeds.len() - 1]
        );
    }
    let _ = engine.restore_kv_branch("__bench");
    engine.lookup_corpus = corpus;
    engine.set_sampler(saved_sampler);
    let _ = engine.save_expert_stats();
    println!();
    println!("{}", Breakdown::from_timing(&all, all_secs).table(all_tokens));
    if engine.is_moe() {
        let picks1 = all_expert_picks(engine);
        let delta: Vec<_> = picks1
            .iter()
            .filter_map(|b| picks0.iter().find(|a| a.layer == b.layer).map(|a| b.since(a)))
            .collect();
        println!();
        println!("{}", expert_report(&delta));
    }
}

/// Expert pick counts for every MoE layer: this PC's, plus the worker's when it reports them.
fn all_expert_picks(engine: &mut SmolLm2Engine) -> Vec<crate::smollm_chat::ExpertPicks> {
    let mut v = engine.expert_picks();
    if let Some(c) = engine.cluster_client.as_mut() {
        if let Some(w) = c.worker_expert_picks() {
            v.extend(w);
        }
    }
    v.sort_by_key(|p| p.layer);
    v
}

/// How many experts (and how much RAM) cover most of the picks, from benchmark counts.
fn expert_report(layers: &[crate::smollm_chat::ExpertPicks]) -> String {
    let shares = [0.5, 0.9, 0.95, 0.99, 1.0];
    let cov = crate::smollm_chat::expert_coverage(layers, &shares);
    if cov.total_picks == 0 {
        return "Expert usage: no expert picks recorded.".into();
    }
    let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
    let mut out = format!(
        "Expert usage: {} picks over {} MoE layers ({} experts each)\n  share of picks   experts per layer   RAM for those experts\n",
        cov.total_picks, cov.layers, cov.n_expert
    );
    for (share, k, bytes) in &cov.rows {
        let label = if *share >= 1.0 { "all picks".to_string() } else { format!("{:.0}%", share * 100.0) };
        out.push_str(&format!("  {label:<16} {k:>8.1}            {:>6.2} GiB\n", gib(*bytes)));
    }
    out.push_str(&format!("  every expert     {:>8}            {:>6.2} GiB\n", cov.n_expert, gib(cov.all_bytes)));
    out.push_str("Counted on the benchmark prompts only; a wider mix of work uses more experts.");
    out
}

fn add_timing(a: &crate::smollm_chat::StepTiming, b: &crate::smollm_chat::StepTiming) -> crate::smollm_chat::StepTiming {
    crate::smollm_chat::StepTiming {
        worker_us: a.worker_us + b.worker_us,
        local_us: a.local_us + b.local_us,
        steps: a.steps + b.steps,
        passes: a.passes + b.passes,
        head_us: a.head_us + b.head_us,
        sample_us: a.sample_us + b.sample_us,
        draft_us: a.draft_us + b.draft_us,
        emit_us: a.emit_us + b.emit_us,
        prefill_us: a.prefill_us + b.prefill_us,
        prefill_tokens: a.prefill_tokens + b.prefill_tokens,
        worker_compute_us: a.worker_compute_us + b.worker_compute_us,
        worker_requests: a.worker_requests + b.worker_requests,
    }
}

/// Runs `opts.first_prompt` (if any), then the interactive loop.
pub fn run(engine: &mut SmolLm2Engine, governor: Option<&Governor>, opts: ChatOptions) {
    if opts.tui && opts.interactive {
        crate::tui::run_tui(engine, governor, opts);
        let _ = engine.save_expert_stats();
        return;
    }

    let mut allow_think = opts.allow_think;
    let mut show_think = opts.allow_think;
    let mut detailed = crate::ui::verbose();
    let mut plan_mode = false;
    let mut max_tokens = if opts.max_tokens > 0 {
        opts.max_tokens.min(engine.max_seq)
    } else {
        2048.min(engine.max_seq)
    };

    let last_timing: std::cell::RefCell<Option<(crate::agent::Breakdown, usize)>> = std::cell::RefCell::new(None);
    let reply = |engine: &mut SmolLm2Engine, text: &str, allow_think: bool, show_think: bool, detailed: bool, max: usize| {
        let mut r = Render::new(show_think);
        let opts = crate::agent::AgentOpts { max_tokens: max, allow_think, ..Default::default() };
        let mut confirm = |what: &str| -> bool {
            print!("\n{}  {} {what} [y/N]: ", dim("│"), crate::ui::bold_magenta("Permission:"));
            let _ = std::io::stdout().flush();
            let mut resp = String::new();
            let _ = std::io::stdin().read_line(&mut resp);
            matches!(resp.trim().to_lowercase().as_str(), "y" | "yes")
        };
        let stats = crate::agent::run_turn(
            engine,
            text,
            opts,
            None,
            &mut |ev| match ev {
                crate::agent::AgentEvent::Token(p) => r.push(&p),
                crate::agent::AgentEvent::ReplyDone { .. } => {
                    if r.wrote {
                        println!();
                    }
                    r = Render::new(show_think);
                }
                crate::agent::AgentEvent::ToolStart { icon, title } => {
                    print!("  {} {}", dim(icon), dim(&title));
                    let _ = std::io::stdout().flush();
                }
                crate::agent::AgentEvent::ToolEnd(o) => {
                    if o.ok {
                        println!("  {}", green("ok"));
                    } else {
                        let first = o.output.lines().next().unwrap_or("").trim_start_matches("Error: ");
                        println!("  {}", crate::ui::bold_magenta(first));
                    }
                    if let Some(p) = &o.preview {
                        for l in p.lines().take(12) {
                            println!("    {}", dim(l));
                        }
                    }
                }
            },
            &mut confirm,
        );
        *last_timing.borrow_mut() = Some((stats.breakdown, stats.tokens));
        let mut line = stats_line(stats.gen_secs.max(1e-3), stats.tokens, engine, governor, detailed);
        if stats.tool_calls > 0 {
            line.push_str(&format!(" · {} tool call{}", stats.tool_calls, if stats.tool_calls == 1 { "" } else { "s" }));
        }
        println!("{}", dim(&format!("  {line}")));
        if detailed && stats.breakdown.passes > 0 {
            for l in stats.breakdown.table(stats.tokens).lines() {
                println!("{}", dim(&format!("  {l}")));
            }
        }
    };

    if engine.n_keep == 0 {
        unsafe { engine.prefill_system(); }
    }

    if let Some(p) = &opts.first_prompt {
        if opts.interactive {
            println!("\n{} {p}", bold_accent(">"));
        }
        reply(engine, p, allow_think, show_think, detailed, max_tokens);
    }
    if !opts.interactive {
        let _ = engine.save_expert_stats();
        return;
    }

    let stdin = std::io::stdin();
    loop {
        print!("\n{} ", bold_accent(">"));
        let _ = std::io::stdout().flush();
        let mut line = String::new();
        match stdin.lock().read_line(&mut line) {
            Ok(0) | Err(_) => break, // EOF / closed stdin
            Ok(_) => {}
        }
        // PowerShell pipes may prepend a UTF-8 BOM to the first line.
        let input = line.trim_start_matches('\u{feff}').trim();
        // Only the command word is case-insensitive; arguments (system prompts, git args,
        // branch names, test commands) keep the case the user typed.
        let cmd_norm = match input.split_once(char::is_whitespace) {
            Some((head, rest)) if head.starts_with('/') => format!("{} {}", head.to_ascii_lowercase(), rest.trim()),
            _ if input.starts_with('/') => input.to_ascii_lowercase(),
            _ => input.to_string(),
        };
        match cmd_norm.as_str() {
            "" => continue,
            "exit" | "quit" | "/exit" | "/quit" | "/q" => break,
            "/help" | "/?" => println!("{}", dim(HELP)),
            "/clear" | "/reset" => {
                engine.reset_kv();
                println!("{}", dim("New conversation."));
            }
            "/model" | "/info" => {
                let mib = engine.raw_bytes.len() >> 20;
                let c = engine.cfg;
                println!("{}  {}", bold("Model:"), accent(&engine.idx.model_name));
                println!(
                    "  architecture  {} · hidden size {} · layers {} of {} here · heads {}/{}",
                    engine.idx.architecture, c.dim,
                    engine.layer_range.len(), c.n_layers,
                    c.n_heads, c.n_kv_heads
                );
                let (ws, private, _) = memory_guard::process_counters();
                let free_ram = crate::mmap_file::available_ram_bytes() >> 20;
                println!(
                    "  memory        {mib} MiB file · {} MiB resident · {} MiB private · {free_ram} MiB free",
                    ws >> 20, private >> 20
                );
                println!(
                    "  context       {}/{} tokens ({:.0} MiB KV cache)",
                    engine.cur_pos, engine.max_seq,
                    ((engine.k_cache.len() + engine.v_cache.len()) * 2) as f64 / (1024.0 * 1024.0)
                );
            }
            "/cluster" | "/nodes" => {
                if let Some(client) = engine.cluster_client.as_mut() {
                    let rtt = client.ping().unwrap_or(std::time::Duration::from_millis(1));
                    println!("{}  {}", bold("Worker:"), green("connected"));
                    println!(
                        "  layers        worker {}..{} · this PC {}..{} · {} total",
                        client.worker_range.0, client.worker_range.1,
                        engine.layer_range.start, engine.layer_range.end,
                        engine.cfg.n_layers
                    );
                    println!(
                        "  network       {:?} activations · {:.2} ms round trip",
                        client.wire_format,
                        rtt.as_secs_f64() * 1e3
                    );
                } else {
                    println!("{}  {}", bold("Worker:"), dim("none (running every layer on this PC)"));
                    println!(
                        "  set peer and layer_range in spindle.toml (or --peer/--layer-range) to use one"
                    );
                }
            }
            cmd if cmd == "/bench" || cmd.starts_with("/bench ") => {
                let n_tokens = if cmd.starts_with("/bench ") {
                    cmd[7..].trim().parse::<usize>().unwrap_or(32).clamp(1, 256)
                } else {
                    32
                };
                println!("{}", dim(&format!("Generating {n_tokens} tokens…")));
                let t_bench0 = std::time::Instant::now();
                let prompt = "Explain in one sentence why systems programming in Rust is reliable and performant.";
                let (_, secs, tokens) = unsafe { engine.generate_reply(prompt, n_tokens, false, &mut |_| {}) };
                let t_total = t_bench0.elapsed().as_secs_f64();
                let speed = tokens as f64 / secs.max(0.001);
                println!(
                    "  {:.1} tok/s · {} tokens in {:.2} s ({:.2} s including the prompt)",
                    speed, tokens, secs, t_total
                );
            }
            "/system" => {
                println!("{}\n{}", bold("System prompt:"), engine.system_prompt);
                println!("{}", dim("/system <text> replaces it, /system reset restores the default"));
            }
            cmd if cmd.starts_with("/system ") => {
                let rest = cmd[8..].trim();
                if rest == "reset" || rest == "default" {
                    let name = engine.idx.model_name.to_lowercase();
                    let def = if name.contains("smollm") {
                        crate::smollm_chat::SMOLLM_SYSTEM_PROMPT
                    } else if name.contains("granite") {
                        crate::smollm_chat::GRANITE_SYSTEM_PROMPT
                    } else if name.contains("qwen") {
                        crate::smollm_chat::QWEN_SYSTEM_PROMPT
                    } else {
                        crate::smollm_chat::DEFAULT_SYSTEM_PROMPT
                    };
                    engine.set_system_prompt(def);
                    println!("{}", dim("System prompt reset to the default."));
                } else if rest == "clear" || rest == "none" {
                    engine.set_system_prompt("");
                    println!("{}", dim("System prompt cleared."));
                } else {
                    engine.set_system_prompt(rest);
                    println!("{}", dim("System prompt updated."));
                }
            }
            "/moe-prune" => {
                let p = engine.moe_prune();
                println!("{}", describe_prune(p));
                println!("{}", dim("/moe-prune 4 keeps the top 4 experts, /moe-prune 0.1 keeps 90% of the router probability, 0 turns it off"));
            }
            cmd if cmd.starts_with("/moe-prune ") => {
                if let Ok(p) = cmd[11..].trim().parse::<f32>() {
                    engine.set_moe_prune(p);
                    println!("{}", dim(&describe_prune(p)));
                } else {
                    println!("{}", dim("usage: /moe-prune <n experts, or a fraction such as 0.1>"));
                }
            }
            "/draft-len" => {
                println!("Up to {} draft tokens per speculative step.", engine.draft_len);
                println!("{}", dim("/draft-len <1..16> changes it"));
            }
            cmd if cmd.starts_with("/draft-len ") => {
                if let Ok(n) = cmd[11..].trim().parse::<usize>() {
                    engine.set_draft_len(n);
                    println!("{}", dim(&format!("Up to {} draft tokens per step.", engine.draft_len)));
                } else {
                    println!("{}", dim("usage: /draft-len <1..16>"));
                }
            }
            "/tools" | "/tools toggle" => {
                let now = !engine.enable_tools;
                engine.set_tools_enabled(now);
                println!("{}", dim(if now { TOOLS_ON } else { TOOLS_OFF }));
            }
            "/tools on" => {
                engine.set_tools_enabled(true);
                println!("{}", dim(TOOLS_ON));
            }
            "/tools off" => {
                engine.set_tools_enabled(false);
                println!("{}", dim(TOOLS_OFF));
            }
            cmd if cmd == "/heal" || cmd.starts_with("/heal ") => {
                let test_cmd = if cmd.starts_with("/heal ") {
                    cmd[6..].trim()
                } else {
                    if std::path::Path::new("Cargo.toml").exists() {
                        "cargo test"
                    } else if std::path::Path::new("package.json").exists() {
                        "npm test"
                    } else if std::path::Path::new("pytest.ini").exists() || std::path::Path::new("pyproject.toml").exists() {
                        "pytest"
                    } else {
                        "cargo test"
                    }
                };

                println!("{}", dim(&format!("Running {test_cmd}…")));
                let (initial_out, initial_code) = crate::guard::run_sandboxed(test_cmd, std::time::Duration::from_secs(60))
                    .unwrap_or_else(|e| (format!("Execution error: {e}"), 1));

                if initial_code == 0 {
                    println!("{}", green("Tests pass. Nothing to fix."));
                } else {
                    println!("{}", crate::ui::warn(&format!("{test_cmd} failed (exit code {initial_code}). Asking the model to fix it.")));
                    engine.save_kv_branch("pre_heal");

                    let mut current_diag = initial_out;
                    let mut healed = false;

                    for attempt in 1..=3 {
                        println!("\n{}", bold(&format!("Attempt {attempt} of 3")));
                        let diag_lines: Vec<&str> = current_diag.lines().collect();
                        let snippet = if diag_lines.len() > 30 {
                            diag_lines[diag_lines.len() - 30..].join("\n")
                        } else {
                            diag_lines.join("\n")
                        };

                        let heal_prompt = format!(
                            "`{test_cmd}` fails with this output (attempt {attempt} of 3):\n\n\
                            ```\n{snippet}\n```\n\n\
                            Find the cause with grep and read_file, then fix it with edit_file. Change only what the fix needs."
                        );

                        reply(engine, &heal_prompt, allow_think, show_think, detailed, max_tokens);

                        println!("\n{}", dim(&format!("Running {test_cmd} again…")));
                        let (re_out, re_code) = crate::guard::run_sandboxed(test_cmd, std::time::Duration::from_secs(60))
                            .unwrap_or_else(|e| (format!("Execution error: {e}"), 1));

                        if re_code == 0 {
                            println!("\n{}", green(&format!("Tests pass after attempt {attempt}.")));
                            println!("{}", dim("/diff shows the changes, /undo reverts them one file at a time."));
                            healed = true;
                            break;
                        } else {
                            current_diag = re_out;
                        }
                    }

                    if !healed {
                        println!("\n{}", crate::ui::warn("Tests still fail after 3 attempts."));
                        print!("Undo the model's edits and return to where the conversation was? [Y/n]: ");
                        let _ = std::io::stdout().flush();
                        let mut ans = String::new();
                        let _ = std::io::stdin().read_line(&mut ans);
                        let t = ans.trim().to_lowercase();
                        if t.is_empty() || t == "y" || t == "yes" {
                            while let Some(_) = crate::tools::pop_undo() {}
                            let _ = engine.restore_kv_branch("pre_heal");
                            println!("{}", dim("Edits undone, conversation restored."));
                        }
                    }
                }
            }
            cmd if cmd == "/branch" || cmd.starts_with("/branch ") => {
                let rest = if cmd.starts_with("/branch ") { cmd[8..].trim() } else { "list" };
                let parts: Vec<&str> = rest.split_whitespace().collect();
                match parts.as_slice() {
                    ["list"] | [] => {
                        let branches = engine.list_kv_branches();
                        println!("{}", bold(&format!("Saved points: {}", branches.len())));
                        for (name, pos) in branches {
                            println!("  {}  at {} tokens", accent(&name), pos);
                        }
                        println!("{}", dim("/branch save <name> saves the current point, /branch revert <name> returns to one"));
                    }
                    ["save", name] => {
                        let pos = engine.save_kv_branch(name);
                        println!("{}", dim(&format!("Saved '{name}' at {pos} tokens.")));
                    }
                    ["revert", name] | ["restore", name] => {
                        match engine.restore_kv_branch(name) {
                            Ok(pos) => println!("{}", dim(&format!("Back to '{name}' ({pos} tokens)."))),
                            Err(e) => println!("{}", crate::ui::warn(&e.to_string())),
                        }
                    }
                    _ => println!("{}", dim("usage: /branch [list | save <name> | revert <name>]")),
                }
            }
            "/review" => {
                let output = std::process::Command::new("git")
                    .args(["diff", "HEAD"])
                    .output();
                match output {
                    Ok(out) => {
                        let diff_str = String::from_utf8_lossy(&out.stdout);
                        let trimmed = diff_str.trim();
                        if trimmed.is_empty() {
                            println!("{}", dim("No uncommitted changes to review."));
                        } else {
                            let max_diff = crate::tools::truncate_middle(trimmed, 8000);
                            let review_prompt = format!(
                                "Review this uncommitted diff. List concrete bugs, regressions and risky changes first, then smaller suggestions.\n\n\
                                ```diff\n{}\n```",
                                max_diff
                            );
                            reply(engine, &review_prompt, allow_think, show_think, detailed, max_tokens);
                        }
                    }
                    Err(e) => println!("{}", crate::ui::warn(&format!("could not run git: {e}"))),
                }
            }
            "/undo" => {
                match crate::tools::pop_undo() {
                    Some(Ok(msg)) => println!("{}", dim(&msg)),
                    Some(Err(e)) => println!("{}", crate::ui::warn(&e.to_string())),
                    None => println!("{}", dim("Nothing to undo.")),
                }
            }
            "/diff" => {
                let _ = std::process::Command::new("git")
                    .args(["diff", "--stat"])
                    .status();
                println!();
                let _ = std::process::Command::new("git")
                    .args(["diff"])
                    .status();
            }
            cmd if cmd == "/git" || cmd.starts_with("/git ") => {
                let args_str = if cmd.starts_with("/git ") { cmd[5..].trim() } else { "status" };
                let args: Vec<&str> = args_str.split_whitespace().collect();
                let _ = std::process::Command::new("git")
                    .args(&args)
                    .status();
            }
            "/plan" | "/plan toggle" => {
                plan_mode = !plan_mode;
                println!("{}", dim(if plan_mode { PLAN_ON } else { PLAN_OFF }));
            }
            "/plan on" => {
                plan_mode = true;
                println!("{}", dim(PLAN_ON));
            }
            "/plan off" => {
                plan_mode = false;
                println!("{}", dim(PLAN_OFF));
            }
            "/init" => {
                let init_content = "# Project notes\n\n\
Spindle adds this file to the system prompt. Describe the project for the model:\n\n\
## What this is\n\n\
## Build and test\n\n\
## Code layout\n\n\
## Conventions\n";
                match std::fs::write("SPINDLE.md", init_content) {
                    Ok(_) => {
                        println!("{}", dim("Wrote SPINDLE.md. Fill it in; it is added to the system prompt."));
                        engine.set_tools_enabled(engine.enable_tools);
                    }
                    Err(e) => println!("{}", crate::ui::warn(&format!("could not write SPINDLE.md: {e}"))),
                }
            }
            "/compact" => {
                println!("{}", dim("Summarizing the conversation…"));
                let prompt = "Summarize our conversation so far in at most 8 short bullet points: goals, files touched, decisions, open tasks.";
                let (summary, _, _) = unsafe { engine.generate_reply(prompt, 256, false, &mut |_| {}) };
                // Re-seed a fresh context with the summary, in the model's own chat format.
                let base = engine.system_prompt.clone();
                engine.set_system_prompt(&format!("{base}\n\n# Summary of the conversation so far\n{}", summary.trim()));
                engine.reset_kv();
                unsafe { engine.prefill_system(); }
                engine.set_system_prompt(&base);
                println!("{}", dim(&format!("Conversation replaced by its summary ({} tokens).", engine.cur_pos)));
            }
            "/think" | "/think toggle" => {
                allow_think = !allow_think;
                show_think = allow_think;
                println!("{}", dim(if allow_think { THINK_ON } else { THINK_OFF }));
            }
            "/think on" => {
                allow_think = true;
                show_think = true;
                println!("{}", dim(THINK_ON));
            }
            "/think off" => {
                allow_think = false;
                show_think = false;
                println!("{}", dim(THINK_OFF));
            }
            "/think show" => {
                show_think = true;
                println!("{}", dim("Reasoning will be shown in full."));
            }
            "/think hide" => {
                show_think = false;
                println!("{}", dim("Reasoning will be collapsed to one line."));
            }
            "/timing" => match *last_timing.borrow() {
                Some((b, t)) => println!("{}", dim(&b.table(t))),
                None => println!("{}", dim("No reply yet. Send a message, then run /timing.")),
            },
            "/stats" => {
                detailed = !detailed;
                println!("{}", dim(if detailed { "Detailed stats on." } else { "Detailed stats off." }));
            }
            cmd if cmd.starts_with("/max ") => {
                if let Ok(n) = cmd[5..].trim().parse::<usize>() {
                    max_tokens = n;
                    println!("{}", dim(&format!("Replies capped at {n} tokens.")));
                } else {
                    println!("{}", dim("usage: /max <tokens>"));
                }
            }
            cmd if cmd.starts_with('/') => println!("{}", dim(&format!("Unknown command {cmd}. Type /help."))),
            _ => {
                let prompt = if plan_mode {
                    format!("Before changing anything, write a short step-by-step plan and wait for my go-ahead.\n\n{}", input)
                } else {
                    input.to_string()
                };
                engine.save_kv_branch("pre_turn");
                reply(engine, &prompt, allow_think, show_think, detailed, max_tokens);
            }
        }
    }
    let _ = engine.save_expert_stats();
}
