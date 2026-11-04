//! The interactive chat: header, prompt, streamed replies and slash commands.

use std::io::{BufRead, Write};

use crate::{
    memory_guard::{self, Governor},
    smollm_chat::SmolLm2Engine,
    ui::{accent, bold, bold_accent, dim, green},
};

const HELP: &str = "\
╭─ Available Slash Commands ─────────────────────────────────────────────╮
│  /tools [on|off]             toggle agentic tools (files, search, cmd) │
│  /heal [test_cmd]            autonomous test-driven self-healing loop  │
│  /branch [save|revert|list]  zero-cost KV-cache hypothesis branching   │
│  /review                     semantic AI review of uncommitted git diff│
│  /undo                       revert the last file change made by agent │
│  /diff                       view uncommitted git diff of current edits│
│  /git <args>                 execute native git command in repository  │
│  /plan [on|off]              toggle architect planning mode            │
│  /init                       scaffold SPINDLE.md project memory file   │
│  /compact                    compact conversation history to free RAM  │
│  /think [on|off|show|hide]   toggle reasoning mode & thought visibility│
│  /system [prompt|reset]      inspect or dynamically change system prompt│
│  /moe-prune [threshold]      inspect or dynamically tune expert cutoff │
│  /draft-len [n]              set speculative draft token batch length  │
│  /cluster                    view distributed mesh status & ping RTT   │
│  /model                      inspect model tensors, layers & RAM       │
│  /bench [n]                  run on-demand speed benchmark (default 32)│
│  /clear                      reset KV-cache & start fresh conversation │
│  /stats                      toggle detailed memory & MoE stats        │
│  /max <n>                    set maximum response token limit          │
│  /help                       show this command reference               │
│  /exit                       quit interactive session                  │
╰────────────────────────────────────────────────────────────────────────╯";

/// Prints model output as it streams. Reasoning between `<think>` and `</think>` is rendered
/// inside an OpenCode-style Thought box, or collapsed into an elegant summary badge.
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
                let _ = writeln!(out, "{}", dim("┌ Thought ──────────────────────────────────────────────────────────"));
            } else {
                let _ = write!(out, "{}", dim("[*] Thinking..."));
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
                let _ = writeln!(out, "{}", dim(&format!("└ Done thinking ({elapsed:.1}s)\n")));
            } else {
                if crate::ui::color() {
                    let _ = write!(out, "\r\x1b[2K");
                }
                let _ = writeln!(out, "{}", dim(&format!("[*] Thought for {elapsed:.1}s (collapsed)\n")));
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
}

pub fn print_header(engine: &SmolLm2Engine, opts: &ChatOptions) {
    let mib = engine.raw_bytes.len() >> 20;
    let mode = if engine.is_streaming() { "low-mem" } else { "ram-resident" };
    let arch = &engine.idx.architecture;
    let name = &engine.idx.model_name;
    println!("{}", dim("╭─ spindle ──────────────────────────────────────────────────────────────╮"));
    println!(
        "{}  {} {} · {} · {} MiB · {}",
        dim("│"),
        bold("Engine:"),
        bold_accent(&format!("Spindle v{}", env!("CARGO_PKG_VERSION"))),
        arch,
        mib,
        mode
    );
    println!(
        "{}  {} {}",
        dim("│"),
        bold("Model: "),
        accent(name)
    );
    println!(
        "{}  {} threads · {}-token context window · tools: {}",
        dim("│"),
        opts.threads,
        engine.max_seq,
        if engine.enable_tools { green("ON") } else { dim("OFF") }
    );
    if opts.interactive {
        println!("{}  type a prompt or {} for slash commands", dim("│"), accent("/help"));
    }
    println!("{}\n", dim("╰────────────────────────────────────────────────────────────────────────╯"));
}

fn stats_line(secs: f64, tokens: usize, engine: &mut SmolLm2Engine, governor: Option<&Governor>, detailed: bool) -> String {
    let mut parts = vec![format!("{:.1} tok/s", tokens as f64 / secs), format!("{tokens} tokens"), format!("{secs:.1} s")];
    if let Some(client) = engine.cluster_client.as_mut() {
        if let Ok(rtt) = client.ping() {
            parts.push(format!("cluster mesh (RTT {:.2} ms)", rtt.as_secs_f64() * 1000.0));
        } else {
            parts.push("cluster mesh".to_string());
        }
    } else if !engine.is_streaming() {
        parts.push("100% RAM resident".to_string());
    } else {
        parts.push("streaming from disk".to_string());
    }
    if let Some((w_ms, l_ms)) = engine.take_cluster_timing() {
        if detailed {
            parts.push(format!("pipeline: worker {:.1}ms, local {:.1}ms", w_ms, l_ms));
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
            "spec: {}/{} ({:.1}%) accepted, {} batches",
            acc,
            tot,
            (acc as f64 / tot as f64) * 100.0,
            b
        ));
    }
    parts.join(" · ")
}

/// Runs `opts.first_prompt` (if any), then the interactive loop.
pub fn run(engine: &mut SmolLm2Engine, governor: Option<&Governor>, opts: ChatOptions) {
    if opts.tui && opts.interactive {
        crate::tui::run_tui(engine, governor, opts);
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
        let mut line = stats_line(stats.gen_secs.max(1e-3), stats.tokens, engine, governor, detailed);
        if stats.tool_calls > 0 {
            line.push_str(&format!(" · {} tool call(s)", stats.tool_calls));
        }
        println!("{}", dim(&format!("  [*] {line}")));
    };

    if engine.n_keep == 0 {
        print!("{}", dim("[*] Pre-warming system prompt and tools KV-cache... "));
        let _ = std::io::stdout().flush();
        let t_warm = std::time::Instant::now();
        unsafe { engine.prefill_system(); }
        let dt = t_warm.elapsed().as_secs_f64();
        println!("{}", dim(&format!("ready ({dt:.1}s, {} tokens cached)", engine.cur_pos)));
    }

    if let Some(p) = &opts.first_prompt {
        if opts.interactive {
            println!("\n{} {p}", bold_accent(">"));
        }
        reply(engine, p, allow_think, show_think, detailed, max_tokens);
    }
    if !opts.interactive {
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
                println!("{}", dim("[*] KV-cache cleared. Starting fresh conversation."));
            }
            "/model" | "/info" => {
                let mib = engine.raw_bytes.len() >> 20;
                let c = engine.cfg;
                println!("{}  {}", bold("Model:"), accent(&engine.idx.model_name));
                println!(
                    "  Architecture : {} (dim: {}, layers: {}/{}, heads: {}/{})",
                    engine.idx.architecture, c.dim,
                    engine.layer_range.len(), c.n_layers,
                    c.n_heads, c.n_kv_heads
                );
                let (ws, private, _) = memory_guard::process_counters();
                let free_ram = crate::mmap_file::available_ram_bytes() >> 20;
                println!(
                    "  Memory       : {mib} MiB file · {} MiB resident · {} MiB private · {free_ram} MiB free RAM",
                    ws >> 20, private >> 20
                );
                println!(
                    "  KV-Cache     : {}/{} tokens ({:.1} MiB allocated)",
                    engine.cur_pos, engine.max_seq,
                    ((engine.k_cache.len() + engine.v_cache.len()) * 2) as f64 / (1024.0 * 1024.0)
                );
            }
            "/cluster" | "/nodes" => {
                if let Some(client) = engine.cluster_client.as_mut() {
                    let rtt = client.ping().unwrap_or(std::time::Duration::from_millis(1));
                    println!("{}  {}", bold("Cluster Mesh:"), green("ONLINE"));
                    println!(
                        "  Partition   : Worker (layers {}..{}) + Local (layers {}..{}) = {} total layers",
                        client.worker_range.0, client.worker_range.1,
                        engine.layer_range.start, engine.layer_range.end,
                        engine.cfg.n_layers
                    );
                    println!(
                        "  Transport   : Wire {:?} · Network Ping RTT: {:.2} ms ({:.1} µs)",
                        client.wire_format,
                        rtt.as_secs_f64() * 1e3,
                        rtt.as_secs_f64() * 1e6
                    );
                } else {
                    println!("{}  {}", bold("Cluster Mesh:"), dim("standalone (no peer connected)"));
                    println!(
                        "  Pass '--peer <IP:PORT> --layer-range <start>..<end>' to join a cluster"
                    );
                }
            }
            cmd if cmd == "/bench" || cmd.starts_with("/bench ") => {
                let n_tokens = if cmd.starts_with("/bench ") {
                    cmd[7..].trim().parse::<usize>().unwrap_or(32).clamp(1, 256)
                } else {
                    32
                };
                println!("{}", dim(&format!("[*] Running {n_tokens}-token benchmark...")));
                let t_bench0 = std::time::Instant::now();
                let prompt = "Explain in one sentence why systems programming in Rust is reliable and performant.";
                let (_, secs, tokens) = unsafe { engine.generate_reply(prompt, n_tokens, false, &mut |_| {}) };
                let t_total = t_bench0.elapsed().as_secs_f64();
                let speed = tokens as f64 / secs.max(0.001);
                println!(
                    "  [{:.1} tok/s · {} tokens in {:.2}s (wall: {:.2}s)]",
                    speed, tokens, secs, t_total
                );
            }
            "/system" => {
                println!("{}  {}", bold("System Prompt:"), accent(&engine.system_prompt));
                println!("{}", dim("  use '/system <prompt>' to set custom, or '/system reset' for model default"));
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
                    println!("{}", dim("[*] System prompt reset to model default."));
                } else if rest == "clear" || rest == "none" {
                    engine.set_system_prompt("");
                    println!("{}", dim("[*] System prompt cleared (empty)."));
                } else {
                    engine.set_system_prompt(rest);
                    println!("{}  {}", dim("[*] System prompt updated:"), accent(rest));
                }
            }
            "/moe-prune" => {
                let p = engine.moe_prune();
                println!("{}  {:.3} ({})", bold("MoE Pruning Cutoff:"), p, if p > 0.0 { "active" } else { "disabled" });
                println!("{}", dim(&format!("  {}", describe_prune(p))));
                println!("{}", dim("  '/moe-prune 4' keeps the top 4 experts, '/moe-prune 0.1' keeps 90% of router mass, '0' disables"));
            }
            cmd if cmd.starts_with("/moe-prune ") => {
                if let Ok(p) = cmd[11..].trim().parse::<f32>() {
                    engine.set_moe_prune(p);
                    println!("{}", dim(&format!("[*] {}", describe_prune(p))));
                } else {
                    println!("{}", dim("usage: /moe-prune <threshold, e.g. 0.03>"));
                }
            }
            "/draft-len" => {
                println!("{}  {} tokens", bold("Speculative Draft Length:"), engine.draft_len);
                println!("{}", dim("  use '/draft-len <1..16>' to set draft speculation depth"));
            }
            cmd if cmd.starts_with("/draft-len ") => {
                if let Ok(n) = cmd[11..].trim().parse::<usize>() {
                    engine.set_draft_len(n);
                    println!("{}", dim(&format!("[*] Speculative draft length set to {} tokens", engine.draft_len)));
                } else {
                    println!("{}", dim("usage: /draft-len <number between 1 and 16>"));
                }
            }
            "/tools" | "/tools toggle" => {
                let now = !engine.enable_tools;
                engine.set_tools_enabled(now);
                println!("{}", dim(if now { "[*] Agentic tools: ON (files, web search, shell enabled)" } else { "[*] Agentic tools: OFF (pure chat mode)" }));
            }
            "/tools on" => {
                engine.set_tools_enabled(true);
                println!("{}", dim("[*] Agentic tools: ON (files, web search, shell enabled)"));
            }
            "/tools off" => {
                engine.set_tools_enabled(false);
                println!("{}", dim("[*] Agentic tools: OFF (pure chat mode)"));
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

                println!("{} Validating project via '{}'...", bold_accent("[*]"), test_cmd);
                let (initial_out, initial_code) = crate::guard::run_sandboxed(test_cmd, std::time::Duration::from_secs(60))
                    .unwrap_or_else(|e| (format!("Execution error: {e}"), 1));

                if initial_code == 0 {
                    println!("{} All tests passed! Repository is healthy and compiling cleanly.", bold_accent("[OK]"));
                } else {
                    println!("{} Build/test failure detected (exit code {}). Initiating autonomous self-healing...", crate::ui::bold_magenta("[FAIL]"), initial_code);
                    engine.save_kv_branch("pre_heal");

                    let mut current_diag = initial_out;
                    let mut healed = false;

                    for attempt in 1..=3 {
                        println!("\n{} Self-Healing Pass {}/3:", bold_accent("[*]"), attempt);
                        let diag_lines: Vec<&str> = current_diag.lines().collect();
                        let snippet = if diag_lines.len() > 30 {
                            diag_lines[diag_lines.len() - 30..].join("\n")
                        } else {
                            diag_lines.join("\n")
                        };

                        let heal_prompt = format!(
                            "Autonomous Self-Healing Loop (Attempt {attempt}/3):\n\
                            The command `{test_cmd}` failed with the following diagnostics:\n\n\
                            ```\n{snippet}\n```\n\n\
                            Inspect the code with `grep` and `read_file`, then fix the error with `edit_file`. Make only necessary, minimal surgical corrections."
                        );

                        reply(engine, &heal_prompt, allow_think, show_think, detailed, max_tokens);

                        println!("\n{} Re-validating via '{}'...", dim("[*]"), test_cmd);
                        let (re_out, re_code) = crate::guard::run_sandboxed(test_cmd, std::time::Duration::from_secs(60))
                            .unwrap_or_else(|e| (format!("Execution error: {e}"), 1));

                        if re_code == 0 {
                            println!("\n{} Self-healing SUCCESSFUL on pass {attempt}! All tests now pass cleanly.", bold_accent("[OK]"));
                            println!("{}", dim("  use '/diff' to review changes, or '/undo' if you want to revert."));
                            healed = true;
                            break;
                        } else {
                            current_diag = re_out;
                        }
                    }

                    if !healed {
                        println!("\n{} Self-healing could not resolve errors after 3 passes.", crate::ui::bold_magenta("[FAIL]"));
                        print!("{} Auto-rollback modifications and restore KV-cache? [Y/n]: ", dim("│"));
                        let _ = std::io::stdout().flush();
                        let mut ans = String::new();
                        let _ = std::io::stdin().read_line(&mut ans);
                        let t = ans.trim().to_lowercase();
                        if t.is_empty() || t == "y" || t == "yes" {
                            while let Some(_) = crate::tools::pop_undo() {}
                            let _ = engine.restore_kv_branch("pre_heal");
                            println!("{} Workspace and KV-cache restored to pre-heal state.", bold_accent("[OK]"));
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
                        println!("{}  {} active branch(es)", bold("KV-Cache Branches:"), branches.len());
                        for (name, pos) in branches {
                            println!("  - {} (cur_pos: {} tokens)", accent(&name), pos);
                        }
                        println!("{}", dim("  use '/branch save <name>' to checkpoint, or '/branch revert <name>' to rewind"));
                    }
                    ["save", name] => {
                        let pos = engine.save_kv_branch(name);
                        println!("{} Saved KV-cache branch '{}' at token pos {}.", bold_accent("[OK]"), name, pos);
                    }
                    ["revert", name] | ["restore", name] => {
                        match engine.restore_kv_branch(name) {
                            Ok(pos) => println!("{} Reverted KV-cache to branch '{}' (pos: {} tokens) in 0 ms.", bold_accent("[OK]"), name, pos),
                            Err(e) => println!("{} {}", dim("[FAIL]"), e),
                        }
                    }
                    _ => println!("{}", dim("usage: /branch [list | save <name> | revert <name>]")),
                }
            }
            "/review" => {
                println!("{}", dim("[*] Analyzing uncommitted changes via git diff..."));
                let output = std::process::Command::new("git")
                    .args(["diff", "HEAD"])
                    .output();
                match output {
                    Ok(out) => {
                        let diff_str = String::from_utf8_lossy(&out.stdout);
                        let trimmed = diff_str.trim();
                        if trimmed.is_empty() {
                            println!("{} No uncommitted changes found to review.", bold_accent("[OK]"));
                        } else {
                            let max_diff = crate::tools::truncate_middle(trimmed, 8000);
                            let review_prompt = format!(
                                "Please provide a rigorous, concise code review of the following git diff. Analyze:\n\
                                1. Correctness, bugs, and edge cases\n\
                                2. Performance / memory implications\n\
                                3. Security guardrail compliance\n\
                                Provide clear, constructive recommendations:\n\n\
                                ```diff\n{}\n```",
                                max_diff
                            );
                            reply(engine, &review_prompt, allow_think, show_think, detailed, max_tokens);
                        }
                    }
                    Err(e) => println!("{} Failed to run git diff: {e}", dim("[FAIL]")),
                }
            }
            "/undo" => {
                match crate::tools::pop_undo() {
                    Some(Ok(msg)) => println!("{} {}", bold_accent("[OK]"), msg),
                    Some(Err(e)) => println!("{} {}", dim("[FAIL]"), e),
                    None => println!("{}", dim("[*] No recent file modifications to undo.")),
                }
            }
            "/diff" => {
                println!("{}", dim("[*] Checking uncommitted git diff..."));
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
                println!("{}", dim(&format!("[*] Executing 'git {}'...", args.join(" "))));
                let _ = std::process::Command::new("git")
                    .args(&args)
                    .status();
            }
            "/plan" | "/plan toggle" => {
                plan_mode = !plan_mode;
                println!("{}", dim(if plan_mode { "[*] Architect Planning mode: ON (model will produce plans before edits)" } else { "[*] Architect Planning mode: OFF (direct execution)" }));
            }
            "/plan on" => {
                plan_mode = true;
                println!("{}", dim("[*] Architect Planning mode: ON (model will produce plans before edits)"));
            }
            "/plan off" => {
                plan_mode = false;
                println!("{}", dim("[*] Architect Planning mode: OFF (direct execution)"));
            }
            "/init" => {
                let init_content = r#"# Spindle Project Instructions

## Overview
This repository uses Spindle as its local high-performance inference engine and agentic coding assistant.

## Build and Test Commands
- Build: `cargo build --release`
- Tests: `cargo test --release`
- Check: `cargo check`

## Architecture Guidelines
- Keep implementations zero-dependency and high-performance.
- Follow pure Rust idioms without panics in production code paths.
- Guard against system destruction and respect host security boundaries.
"#;
                match std::fs::write("SPINDLE.md", init_content) {
                    Ok(_) => {
                        println!("{} {}", bold_accent("[OK]"), "Created SPINDLE.md project memory file.");
                        engine.set_tools_enabled(engine.enable_tools);
                    }
                    Err(e) => println!("{} Failed to create SPINDLE.md: {e}", dim("[FAIL]")),
                }
            }
            "/compact" => {
                println!("{}", dim("[*] Compacting conversation memory..."));
                let prompt = "Please provide a concise 3-sentence summary of the key tasks, accomplishments, and decisions made in our conversation so far.";
                let (summary, _, _) = unsafe { engine.generate_reply(prompt, 256, false, &mut |_| {}) };
                // Re-seed a fresh context with the summary, in the model's own chat format.
                let base = engine.system_prompt.clone();
                engine.set_system_prompt(&format!("{base}\n\n# Summary of the conversation so far\n{}", summary.trim()));
                engine.reset_kv();
                unsafe { engine.prefill_system(); }
                engine.set_system_prompt(&base);
                println!("{} Conversation compacted. KV-cache reset and context restored.", bold_accent("[OK]"));
            }
            "/think" | "/think toggle" => {
                allow_think = !allow_think;
                show_think = allow_think;
                println!("{}", dim(if allow_think { "[*] Reasoning mode: ON (chain-of-thought enabled)" } else { "[*] Reasoning mode: OFF (direct fast answer)" }));
            }
            "/think on" => {
                allow_think = true;
                show_think = true;
                println!("{}", dim("[*] Reasoning mode: ON (chain-of-thought enabled)"));
            }
            "/think off" => {
                allow_think = false;
                show_think = false;
                println!("{}", dim("[*] Reasoning mode: OFF (direct fast answer)"));
            }
            "/think show" => {
                show_think = true;
                println!("{}", dim("[*] Thought visibility: SHOWN (full reasoning box)"));
            }
            "/think hide" => {
                show_think = false;
                println!("{}", dim("[*] Thought visibility: COLLAPSED (summary badge)"));
            }
            "/stats" => {
                detailed = !detailed;
                println!("{}", dim(if detailed { "[*] Detailed memory & MoE stats: ON" } else { "[*] Detailed memory & MoE stats: OFF" }));
            }
            cmd if cmd.starts_with("/max ") => {
                if let Ok(n) = cmd[5..].trim().parse::<usize>() {
                    max_tokens = n;
                    println!("{}", dim(&format!("[*] Max tokens set to {n}")));
                } else {
                    println!("{}", dim("usage: /max <number of tokens>"));
                }
            }
            cmd if cmd.starts_with('/') => println!("{}", dim(&format!("unknown command {cmd}, try /help"))),
            _ => {
                let prompt = if plan_mode {
                    format!("Architect Mode: First provide a detailed technical plan. Do not execute destructive edits yet.\n\n{}", input)
                } else {
                    input.to_string()
                };
                engine.save_kv_branch("pre_turn");
                reply(engine, &prompt, allow_think, show_think, detailed, max_tokens);
            }
        }
    }
}
