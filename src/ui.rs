//! Terminal output helpers: optional color and a verbose switch for diagnostics.
//!
//! Color is used only when stdout is a real console that accepts ANSI sequences and
//! `NO_COLOR` isn't set, so piped output stays plain text.

use std::{
    io::IsTerminal,
    sync::atomic::{AtomicBool, Ordering},
};

use windows::Win32::System::Console::{GetConsoleMode, GetStdHandle, SetConsoleMode, CONSOLE_MODE, ENABLE_VIRTUAL_TERMINAL_PROCESSING, STD_OUTPUT_HANDLE};

static VERBOSE: AtomicBool = AtomicBool::new(true);
static COLOR: AtomicBool = AtomicBool::new(false);

pub fn init(verbose: bool) {
    VERBOSE.store(verbose, Ordering::Relaxed);
    let color = std::env::var_os("NO_COLOR").is_none() && std::io::stdout().is_terminal() && enable_vt();
    COLOR.store(color, Ordering::Relaxed);
}

fn enable_vt() -> bool {
    unsafe {
        let Ok(h) = GetStdHandle(STD_OUTPUT_HANDLE) else { return false };
        let mut mode = CONSOLE_MODE(0);
        GetConsoleMode(h, &mut mode).is_ok() && SetConsoleMode(h, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING).is_ok()
    }
}

pub fn verbose() -> bool {
    VERBOSE.load(Ordering::Relaxed)
}

pub fn color() -> bool {
    COLOR.load(Ordering::Relaxed)
}

fn paint(code: &str, s: &str) -> String {
    if color() {
        format!("\x1b[{code}m{s}\x1b[0m")
    } else {
        s.to_string()
    }
}

pub fn dim(s: &str) -> String {
    paint("2", s)
}

pub fn bold(s: &str) -> String {
    paint("1", s)
}

pub fn accent(s: &str) -> String {
    paint("36", s)
}

pub fn cyan(s: &str) -> String {
    paint("36", s)
}

pub fn magenta(s: &str) -> String {
    paint("35", s)
}

pub fn green(s: &str) -> String {
    paint("32", s)
}

pub fn bold_accent(s: &str) -> String {
    paint("1;36", s)
}

pub fn bold_magenta(s: &str) -> String {
    paint("1;35", s)
}

pub fn warn(s: &str) -> String {
    paint("33", s)
}

/// Something the user should know even without `--verbose` (low memory, fallbacks).
pub fn note(msg: &str) {
    eprintln!("{}", warn(msg));
}

/// Diagnostics, printed only with `--verbose` (or outside chat mode).
#[macro_export]
macro_rules! vlog {
    ($($arg:tt)*) => {
        if $crate::ui::verbose() {
            eprintln!("{}", $crate::ui::dim(&format!($($arg)*)));
        }
    };
}
