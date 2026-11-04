//! Startup progress display.
//!
//! Each startup phase (reading the model, loading weights, connecting to the worker, ...)
//! is one line. While a phase runs its line shows a spinner, and a progress bar when the
//! phase knows how far along it is; when it finishes the line is replaced by a check mark,
//! a short result and the time it took. Without a color terminal (piped output, `NO_COLOR`,
//! `--verbose`) each phase prints one plain line when it finishes instead.

use std::{
    io::Write,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
    time::{Duration, Instant},
};

const SPINNER: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
const BAR_WIDTH: usize = 24;
const LABEL_WIDTH: usize = 9;

/// Set while a phase line is being redrawn, so warnings can clear it before printing.
static LINE_ACTIVE: AtomicBool = AtomicBool::new(false);

/// Clears an in-progress phase line (if any) so other output starts at column 0.
pub fn clear_active_line() {
    if LINE_ACTIVE.load(Ordering::Relaxed) {
        let mut out = std::io::stdout().lock();
        let _ = write!(out, "\r\x1b[2K");
        let _ = out.flush();
    }
}

pub struct Loader {
    animate: bool,
    started: Instant,
}

impl Loader {
    /// Prints the title line. `title` is usually "spindle 1.0.0".
    pub fn start(title: &str, subtitle: &str) -> Self {
        let animate = crate::ui::color() && !crate::ui::verbose();
        println!();
        println!("  {}  {}", crate::ui::bold(title), crate::ui::dim(subtitle));
        println!();
        Self { animate, started: Instant::now() }
    }

    /// Begins a phase. `detail` is shown next to the spinner until the phase ends.
    pub fn step(&self, label: &str, detail: &str) -> Step {
        let state = Arc::new(Mutex::new(StepState { detail: detail.to_string(), frac: None }));
        let stop = Arc::new(AtomicBool::new(false));
        let started = Instant::now();
        let ticker = self.animate.then(|| {
            let (state, stop, label) = (state.clone(), stop.clone(), label.to_string());
            LINE_ACTIVE.store(true, Ordering::Relaxed);
            std::thread::spawn(move || {
                let mut frame = 0usize;
                while !stop.load(Ordering::Relaxed) {
                    let s = state.lock().unwrap().clone();
                    draw_running(&label, &s, SPINNER[frame % SPINNER.len()]);
                    frame += 1;
                    std::thread::sleep(Duration::from_millis(80));
                }
            })
        });
        Step { label: label.to_string(), state, stop, ticker, started, finished: false }
    }

    /// Prints the closing line ("Ready in 12.4 s").
    pub fn finish(self, message: &str) {
        let secs = self.started.elapsed().as_secs_f64();
        println!();
        println!("  {} {}", crate::ui::bold(message), crate::ui::dim(&format!("in {}", fmt_secs(secs))));
        println!();
    }
}

#[derive(Clone)]
struct StepState {
    detail: String,
    frac: Option<f64>,
}

pub struct Step {
    label: String,
    state: Arc<Mutex<StepState>>,
    stop: Arc<AtomicBool>,
    ticker: Option<std::thread::JoinHandle<()>>,
    started: Instant,
    finished: bool,
}

impl Step {
    /// Updates the progress bar (`done` of `total`) and the text next to it.
    pub fn progress(&self, done: u64, total: u64, detail: String) {
        let mut s = self.state.lock().unwrap();
        s.frac = (total > 0).then(|| (done as f64 / total as f64).clamp(0.0, 1.0));
        s.detail = detail;
    }

    /// Ends the phase with a check mark and `result`.
    pub fn done(mut self, result: &str) {
        self.end();
        let secs = self.started.elapsed().as_secs_f64();
        let mark = crate::ui::green("✓");
        println!("  {mark} {}  {}  {}", pad(&self.label), result, crate::ui::dim(&fmt_secs(secs)));
    }

    /// Ends the phase with a cross, `error` and an optional hint on the next line.
    pub fn fail(mut self, error: &str, hint: Option<&str>) {
        self.end();
        let mark = crate::ui::red("✗");
        println!("  {mark} {}  {}", pad(&self.label), crate::ui::red(error));
        if let Some(h) = hint {
            for line in h.lines() {
                println!("    {}{}", " ".repeat(LABEL_WIDTH + 2), crate::ui::dim(line));
            }
        }
    }

    fn end(&mut self) {
        self.finished = true;
        self.stop.store(true, Ordering::Relaxed);
        if let Some(t) = self.ticker.take() {
            let _ = t.join();
            LINE_ACTIVE.store(false, Ordering::Relaxed);
            let mut out = std::io::stdout().lock();
            let _ = write!(out, "\r\x1b[2K");
            let _ = out.flush();
        }
    }
}

impl Drop for Step {
    fn drop(&mut self) {
        if !self.finished {
            self.end();
        }
    }
}

fn pad(label: &str) -> String {
    crate::ui::bold(&format!("{label:<LABEL_WIDTH$}"))
}

fn draw_running(label: &str, s: &StepState, spin: &str) {
    let cols = crate::tui::terminal_size().0 as usize;
    let mut plain_len = 2 + 1 + 1 + LABEL_WIDTH + 2;
    let mut line = format!("  {} {}  ", crate::ui::accent(spin), pad(label));
    if let Some(f) = s.frac {
        let filled = (f * BAR_WIDTH as f64).round() as usize;
        line.push_str(&crate::ui::accent(&"█".repeat(filled)));
        line.push_str(&crate::ui::dim(&"░".repeat(BAR_WIDTH - filled)));
        let pct = format!(" {:>3.0}%  ", f * 100.0);
        line.push_str(&pct);
        plain_len += BAR_WIDTH + pct.chars().count();
    }
    // Keep the line shorter than the window: a wrapped line can't be redrawn in place.
    let room = cols.saturating_sub(plain_len + 1);
    let detail: String = s.detail.chars().take(room).collect();
    line.push_str(&crate::ui::dim(&detail));
    let mut out = std::io::stdout().lock();
    let _ = write!(out, "\r\x1b[2K{line}");
    let _ = out.flush();
}

pub fn fmt_secs(secs: f64) -> String {
    if secs < 1.0 {
        format!("{:.0} ms", secs * 1000.0)
    } else if secs < 60.0 {
        format!("{secs:.1} s")
    } else {
        format!("{}m {:02}s", secs as u64 / 60, secs as u64 % 60)
    }
}

/// Bytes as "812 MiB" / "3.9 GiB".
pub fn fmt_bytes(b: u64) -> String {
    const GIB: f64 = (1u64 << 30) as f64;
    if b as f64 >= GIB {
        format!("{:.1} GiB", b as f64 / GIB)
    } else {
        format!("{} MiB", b >> 20)
    }
}

/// Integer with thousands separators ("1,204").
pub fn fmt_count(n: usize) -> String {
    let s = n.to_string();
    let mut out = String::new();
    for (i, ch) in s.chars().enumerate() {
        if i > 0 && (s.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(ch);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats() {
        assert_eq!(fmt_count(1204), "1,204");
        assert_eq!(fmt_count(12), "12");
        assert_eq!(fmt_count(1_000_000), "1,000,000");
        assert_eq!(fmt_bytes(3 << 20), "3 MiB");
        assert_eq!(fmt_bytes(4 << 30), "4.0 GiB");
        assert_eq!(fmt_secs(0.25), "250 ms");
        assert_eq!(fmt_secs(12.34), "12.3 s");
        assert_eq!(fmt_secs(75.0), "1m 15s");
    }
}
