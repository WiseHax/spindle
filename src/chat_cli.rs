//! The interactive chat: header, prompt, streamed replies and slash commands.

use std::io::{BufRead, Write};

use crate::{
    memory_guard::{self, Governor},
    smollm_chat::SmolLm2Engine,
    ui::{accent, bold, dim},
};

const HELP: &str = "\
/clear    start a new conversation
/think    show or hide the model's reasoning (Qwen3 <think> blocks)
/stats    show or hide memory details after each reply
/help     this list
/exit     quit (also: exit, quit, Ctrl+Z then Enter)";

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
    if detailed {
        let (ws, private, _) = memory_guard::process_counters();
        parts.push(format!("{} MiB RAM ({} MiB private)", ws >> 20, private >> 20));
        if let Some(g) = governor {
            let ord = std::sync::atomic::Ordering::Relaxed;
            parts.push(format!("cap {} MiB", g.status.cap.load(ord) >> 20));
            parts.push(format!("{} MiB free", g.status.available.load(ord) >> 20));
        }
    }
    if let Some((mib, touched, total)) = engine.take_moe_stats() {
        if detailed {
            parts.push(format!("{mib:.0} MiB experts/token, {touched}/{total} experts used"));
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
    parts.join(" · ")
}

/// Runs `opts.first_prompt` (if any), then the interactive loop.
pub fn run(engine: &mut SmolLm2Engine, governor: Option<&Governor>, opts: ChatOptions) {
    let mut show_think = false;
    let mut detailed = crate::ui::verbose();

    let reply = |engine: &mut SmolLm2Engine, text: &str, show_think: bool, detailed: bool, max: usize| {
        let mut r = Render::new(show_think);
        let (_, secs, tokens) = unsafe { engine.generate_reply(text, max, &mut |p| r.push(p)) };
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
        reply(engine, p, show_think, detailed, 256);
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
            "/think" => {
                show_think = !show_think;
                println!("{}", dim(if show_think { "reasoning: shown" } else { "reasoning: hidden" }));
            }
            "/stats" => {
                detailed = !detailed;
                println!("{}", dim(if detailed { "memory details: on" } else { "memory details: off" }));
            }
            cmd if cmd.starts_with('/') => println!("{}", dim(&format!("unknown command {cmd}, try /help"))),
            _ => reply(engine, input, show_think, detailed, 512),
        }
    }
}
