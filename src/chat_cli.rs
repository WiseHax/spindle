//! The interactive chat: header, prompt, streamed replies and slash commands.

use std::io::{BufRead, Write};

use crate::{
    memory_guard::{self, Governor},
    smollm_chat::SmolLm2Engine,
    ui::{accent, bold, dim},
};

const HELP: &str = "\
/clear       start a new conversation
/model       display model metadata, architecture, and memory footprint
/cluster     check distributed cluster peer status and network ping RTT
/bench       run a fast 32-token generation benchmark (tok/s)
/think       toggle reasoning mode on/off (/think on, /think off)
/stats       show or hide memory details after each reply
/max <n>     set maximum generation tokens (default: 2048)
/help        this list
/exit        quit (also: exit, quit, Ctrl+Z then Enter)";

/// Prints model output as it streams. Reasoning between `<think>` and `</think>` is shown
/// dimmed, or collapsed into a single "thinking..." line that is erased once the answer starts.
struct Render {
    show_think: bool,
    in_think: bool,
    /// Drop the blank lines a model puts between its reasoning and its answer.
    skip_newlines: bool,
    wrote: bool,
}

impl Render {
    fn new(show_think: bool) -> Self {
        Self { show_think, in_think: false, skip_newlines: true, wrote: false }
    }

    fn push(&mut self, piece: &str) {
        let mut out = std::io::stdout();
        if piece.trim() == "<think>" {
            self.in_think = true;
            self.skip_newlines = true;
            let _ = write!(out, "{}", dim(if self.show_think { "thinking\n" } else { "thinking..." }));
        } else if piece.trim() == "</think>" {
            self.in_think = false;
            self.skip_newlines = true;
            if self.show_think {
                let _ = write!(out, "\n");
            } else if crate::ui::color() {
                let _ = write!(out, "\r\x1b[2K"); // erase the "thinking..." line
            } else {
                let _ = writeln!(out);
            }
        } else if self.in_think {
            let text = if self.skip_newlines { piece.trim_start_matches(['\n', '\r']) } else { piece };
            if self.show_think && !text.is_empty() {
                self.skip_newlines = false;
                let _ = write!(out, "{}", dim(text));
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

pub struct ChatOptions {
    pub first_prompt: Option<String>,
    pub interactive: bool,
    pub threads: usize,
    pub max_tokens: usize,
    pub allow_think: bool,
}

pub fn print_header(engine: &SmolLm2Engine, opts: &ChatOptions) {
    let mib = engine.raw_bytes.len() >> 20;
    println!("{}  {}", bold("spindle"), accent(&engine.idx.model_name));
    let mode = if engine.is_streaming() { " · low-memory mode" } else { "" };
    println!(
        "{}",
        dim(&format!(
            "{} · {mib} MiB · {} threads · {}-token context{mode}",
            engine.idx.architecture, opts.threads, engine.max_seq
        ))
    );
    if opts.interactive {
        println!("{}", dim("type a message, or /help"));
    }
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
    let mut allow_think = opts.allow_think;
    let mut show_think = opts.allow_think;
    let mut detailed = crate::ui::verbose();
    let mut max_tokens = if opts.max_tokens > 0 { opts.max_tokens } else { 2048 };

    let reply = |engine: &mut SmolLm2Engine, text: &str, allow_think: bool, show_think: bool, detailed: bool, max: usize| {
        let mut r = Render::new(show_think);
        let (_, secs, tokens) = unsafe { engine.generate_reply(text, max, allow_think, &mut |p| r.push(p)) };
        if r.wrote {
            println!();
        } else {
            // e.g. the model reasoned and then stopped without answering
            println!("{}", dim("(no answer)"));
        }
        println!("{}", dim(&format!("  {}", stats_line(secs, tokens, engine, governor, detailed))));
    };

    if let Some(p) = &opts.first_prompt {
        if opts.interactive {
            println!("\n{} {p}", bold(&accent(">")));
        }
        reply(engine, p, allow_think, show_think, detailed, max_tokens);
    }
    if !opts.interactive {
        return;
    }

    let stdin = std::io::stdin();
    loop {
        print!("\n{} ", bold(&accent(">")));
        let _ = std::io::stdout().flush();
        let mut line = String::new();
        match stdin.lock().read_line(&mut line) {
            Ok(0) | Err(_) => break, // EOF / closed stdin
            Ok(_) => {}
        }
        // PowerShell pipes may prepend a UTF-8 BOM to the first line.
        let input = line.trim_start_matches('\u{feff}').trim();
        match input.to_ascii_lowercase().as_str() {
            "" => continue,
            "exit" | "quit" | "/exit" | "/quit" => break,
            "/help" | "/?" => println!("{}", dim(HELP)),
            "/clear" | "/reset" => {
                engine.reset_kv();
                println!("{}", dim("new conversation"));
            }
            "/model" | "/info" => {
                let mib = engine.raw_bytes.len() >> 20;
                let c = engine.cfg;
                println!("{}  {}", bold("model:"), accent(&engine.idx.model_name));
                println!(
                    "  arch: {} · dim: {} · layers: {}/{} (range: {}..{}) · heads: {}/{}",
                    engine.idx.architecture, c.dim,
                    engine.layer_range.len(), c.n_layers,
                    engine.layer_range.start, engine.layer_range.end,
                    c.n_heads, c.n_kv_heads
                );
                let (ws, private, _) = memory_guard::process_counters();
                let free_ram = crate::mmap_file::available_ram_bytes() >> 20;
                println!(
                    "  memory: {mib} MiB file · {} MiB resident · {} MiB private · {free_ram} MiB free RAM",
                    ws >> 20, private >> 20
                );
                println!(
                    "  kv-cache: {}/{} tokens ({:.1} MiB allocated)",
                    engine.cur_pos, engine.max_seq,
                    ((engine.k_cache.len() + engine.v_cache.len()) * 2) as f64 / (1024.0 * 1024.0)
                );
            }
            "/cluster" | "/nodes" => {
                if let Some(client) = engine.cluster_client.as_mut() {
                    let rtt = client.ping().unwrap_or(std::time::Duration::from_millis(1));
                    println!("{}  {}", bold("spindle cluster:"), accent("online"));
                    println!(
                        "  partition: worker (layers {}..{}) + local (layers {}..{}) = {} total",
                        client.worker_range.0, client.worker_range.1,
                        engine.layer_range.start, engine.layer_range.end,
                        engine.cfg.n_layers
                    );
                    println!(
                        "  transport: wire {:?} · LAN ping RTT: {:.2} µs ({:.2} ms)",
                        client.wire_format,
                        rtt.as_secs_f64() * 1e6,
                        rtt.as_secs_f64() * 1e3
                    );
                } else {
                    println!("{}  {}", bold("spindle cluster:"), dim("standalone (no peer connected)"));
                    println!(
                        "  run with '--peer <IP:PORT> --layer-range <start>..<end>' to join a cluster"
                    );
                }
            }
            "/bench" => {
                println!("{}", dim("running quick benchmark (32 tokens)..."));
                let t_bench0 = std::time::Instant::now();
                let prompt = "Explain in one sentence why systems programming in Rust is reliable and performant.";
                let (_, secs, tokens) = unsafe { engine.generate_reply(prompt, 32, false, &mut |_| {}) };
                let t_total = t_bench0.elapsed().as_secs_f64();
                let speed = tokens as f64 / secs.max(0.001);
                println!(
                    "{} benchmark: {} tokens in {:.2}s -> {:.1} tok/s (wall: {:.2}s)",
                    accent("✓"), tokens, secs, speed, t_total
                );
            }
            "/think" | "/think toggle" => {
                allow_think = !allow_think;
                show_think = allow_think;
                println!("{}", dim(if allow_think { "thinking mode: ON (reasoning enabled)" } else { "thinking mode: OFF (direct fast answer)" }));
            }
            "/think on" => {
                allow_think = true;
                show_think = true;
                println!("{}", dim("thinking mode: ON (reasoning enabled)"));
            }
            "/think off" => {
                allow_think = false;
                show_think = false;
                println!("{}", dim("thinking mode: OFF (direct fast answer)"));
            }
            "/think show" => {
                show_think = true;
                println!("{}", dim("reasoning visibility: shown"));
            }
            "/think hide" => {
                show_think = false;
                println!("{}", dim("reasoning visibility: hidden"));
            }
            "/stats" => {
                detailed = !detailed;
                println!("{}", dim(if detailed { "memory details: on" } else { "memory details: off" }));
            }
            cmd if cmd.starts_with("/max ") => {
                if let Ok(n) = cmd[5..].trim().parse::<usize>() {
                    max_tokens = n;
                    println!("{}", dim(&format!("max tokens set to {n}")));
                } else {
                    println!("{}", dim("usage: /max <number of tokens>"));
                }
            }
            cmd if cmd.starts_with('/') => println!("{}", dim(&format!("unknown command {cmd}, try /help"))),
            _ => reply(engine, input, allow_think, show_think, detailed, max_tokens),
        }
    }
}
