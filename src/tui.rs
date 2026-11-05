//! Full-screen terminal UI: the pixel logo and prompt box on the home screen, a loading
//! screen while the system prompt is processed, then the conversation. User messages are
//! marked `›`. Tool calls appear as one-line rows (`▸ Read`, `⌕ Grep`, ...) or as blocks
//! for shell output and edit diffs. Each reply ends with a `◆` footer (mode, model, time,
//! speed). A status row sits under the prompt, and a session sidebar shows on wide terminals.
//!
//! Rendering is a cell canvas flushed in one write per frame (no flicker). Generation runs
//! on a worker thread so the UI keeps scrolling/typing at ~60 FPS; tool permission prompts
//! come back to the UI thread over a channel.

use std::{
    io::{self, Read, Write},
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc, Arc,
    },
    time::{Duration, Instant},
};

use windows::Win32::System::Console::{
    GetConsoleMode, GetConsoleScreenBufferInfo, GetNumberOfConsoleInputEvents, GetStdHandle, SetConsoleMode, CONSOLE_MODE,
    CONSOLE_SCREEN_BUFFER_INFO, ENABLE_ECHO_INPUT, ENABLE_LINE_INPUT, ENABLE_PROCESSED_INPUT, ENABLE_VIRTUAL_TERMINAL_INPUT,
    ENABLE_VIRTUAL_TERMINAL_PROCESSING, STD_INPUT_HANDLE, STD_OUTPUT_HANDLE,
};

use crate::{
    agent::{self, AgentEvent, AgentOpts, TurnStats},
    smollm_chat::SmolLm2Engine,
};

// ------------------------------------------------------------------------------ theme

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Rgb(pub u8, pub u8, pub u8);

/// Spindle's colors: a deep navy background with blue accents, matching the logo.
pub mod theme {
    use super::Rgb;
    pub const BG: Rgb = Rgb(0x0b, 0x0f, 0x17);
    pub const PANEL: Rgb = Rgb(0x11, 0x18, 0x26);
    pub const ELEMENT: Rgb = Rgb(0x18, 0x22, 0x35);
    pub const BORDER: Rgb = Rgb(0x2a, 0x3a, 0x57);
    pub const BORDER_SUBTLE: Rgb = Rgb(0x1e, 0x2a, 0x40);
    pub const PRIMARY: Rgb = Rgb(0x4c, 0x8d, 0xff);
    pub const SECONDARY: Rgb = Rgb(0x5e, 0xc8, 0xff);
    pub const ACCENT: Rgb = Rgb(0x9c, 0xc4, 0xff);
    pub const TEXT: Rgb = Rgb(0xe3, 0xea, 0xf5);
    pub const MUTED: Rgb = Rgb(0x7a, 0x87, 0x9e);
    pub const ERROR: Rgb = Rgb(0xef, 0x6b, 0x73);
    pub const WARNING: Rgb = Rgb(0xf2, 0xb7, 0x4e);
    pub const SUCCESS: Rgb = Rgb(0x5f, 0xd3, 0x9a);
    pub const INFO: Rgb = Rgb(0x5e, 0xc8, 0xff);
    pub const DIFF_ADD: Rgb = Rgb(0x6e, 0xe7, 0xb7);
    pub const DIFF_DEL: Rgb = Rgb(0xf0, 0x8a, 0x8a);
    pub const DIFF_ADD_BG: Rgb = Rgb(0x10, 0x2c, 0x28);
    pub const DIFF_DEL_BG: Rgb = Rgb(0x33, 0x17, 0x20);
    /// Top-to-bottom colors of the logo's five pixel rows.
    pub const LOGO_ROWS: [Rgb; 5] = [
        Rgb(0xee, 0xf4, 0xff),
        Rgb(0xb6, 0xd3, 0xff),
        Rgb(0x7f, 0xae, 0xff),
        Rgb(0x4c, 0x8d, 0xff),
        Rgb(0x2a, 0x63, 0xe6),
    ];
}
use theme::*;

// --------------------------------------------------------------------------- terminal

/// Restores the console modes and screen when dropped (also on early return).
pub struct TerminalGuard {
    in_handle: windows::Win32::Foundation::HANDLE,
    out_handle: windows::Win32::Foundation::HANDLE,
    orig_in_mode: CONSOLE_MODE,
    orig_out_mode: CONSOLE_MODE,
}

impl TerminalGuard {
    pub fn enter() -> io::Result<Self> {
        let e = |e: windows::core::Error| io::Error::other(e.to_string());
        unsafe {
            let in_handle = GetStdHandle(STD_INPUT_HANDLE).map_err(e)?;
            let out_handle = GetStdHandle(STD_OUTPUT_HANDLE).map_err(e)?;
            let mut orig_in_mode = CONSOLE_MODE(0);
            let mut orig_out_mode = CONSOLE_MODE(0);
            GetConsoleMode(in_handle, &mut orig_in_mode).map_err(e)?;
            GetConsoleMode(out_handle, &mut orig_out_mode).map_err(e)?;
            let _ = SetConsoleMode(out_handle, CONSOLE_MODE(orig_out_mode.0 | ENABLE_VIRTUAL_TERMINAL_PROCESSING.0));
            // Raw VT input: no line buffering/echo, and Ctrl+C arrives as a key (byte 3).
            // Quick-edit mode off (with the extended-flags bit that makes that stick), so the
            // classic console passes mouse events to us instead of starting a selection.
            const QUICK_EDIT: u32 = 0x0040;
            const EXTENDED_FLAGS: u32 = 0x0080;
            let new_in = ((orig_in_mode.0 | ENABLE_VIRTUAL_TERMINAL_INPUT.0 | EXTENDED_FLAGS) & !QUICK_EDIT)
                & !(ENABLE_LINE_INPUT.0 | ENABLE_ECHO_INPUT.0 | ENABLE_PROCESSED_INPUT.0);
            let _ = SetConsoleMode(in_handle, CONSOLE_MODE(new_in));
            let mut out = io::stdout();
            // Alternate screen, hidden cursor, no autowrap, mouse reporting in SGR form (so the
            // wheel arrives as wheel events, not as arrow keys that would recall prompt
            // history), bracketed paste. Hold Shift to select text with the mouse.
            let _ = write!(out, "\x1b[?1049h\x1b[?25l\x1b[?7l\x1b[?1000h\x1b[?1006h\x1b[?2004h");
            let _ = out.flush();
            Ok(Self { in_handle, out_handle, orig_in_mode, orig_out_mode })
        }
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        unsafe {
            let mut out = io::stdout();
            let _ = write!(out, "\x1b[?2004l\x1b[?1007l\x1b[?1006l\x1b[?1000l\x1b[?7h\x1b[?25h\x1b[?1049l\x1b[0m");
            let _ = out.flush();
            let _ = SetConsoleMode(self.out_handle, self.orig_out_mode);
            let _ = SetConsoleMode(self.in_handle, self.orig_in_mode);
        }
    }
}

/// Console size in cells (columns, rows).
pub fn terminal_size() -> (u16, u16) {
    unsafe {
        if let Ok(h) = GetStdHandle(STD_OUTPUT_HANDLE) {
            let mut csbi = CONSOLE_SCREEN_BUFFER_INFO::default();
            if GetConsoleScreenBufferInfo(h, &mut csbi).is_ok() {
                let w = (csbi.srWindow.Right - csbi.srWindow.Left + 1).max(20) as u16;
                let h = (csbi.srWindow.Bottom - csbi.srWindow.Top + 1).max(10) as u16;
                return (w, h);
            }
        }
    }
    (100, 30)
}

fn has_console_input() -> bool {
    unsafe {
        if let Ok(h) = GetStdHandle(STD_INPUT_HANDLE) {
            let mut n = 0u32;
            if GetNumberOfConsoleInputEvents(h, &mut n).is_ok() {
                return n > 0;
            }
        }
    }
    false
}

// ----------------------------------------------------------------------------- canvas

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Cell {
    pub ch: char,
    pub fg: Rgb,
    pub bg: Rgb,
    pub bold: bool,
    pub italic: bool,
}

impl Default for Cell {
    fn default() -> Self {
        Self { ch: ' ', fg: TEXT, bg: BG, bold: false, italic: false }
    }
}

/// Columns a character takes in the terminal: 2 for CJK and most emoji, 0 for combining
/// marks and zero-width joiners, 1 otherwise.
pub fn char_width(c: char) -> u16 {
    let u = c as u32;
    match u {
        0x0300..=0x036F | 0x200B..=0x200F | 0xFE00..=0xFE0F | 0x20D0..=0x20FF => 0,
        0x1100..=0x115F
        | 0x2E80..=0x303E
        | 0x3041..=0x33FF
        | 0x3400..=0x4DBF
        | 0x4E00..=0x9FFF
        | 0xA000..=0xA4CF
        | 0xAC00..=0xD7A3
        | 0xF900..=0xFAFF
        | 0xFE30..=0xFE4F
        | 0xFF00..=0xFF60
        | 0xFFE0..=0xFFE6
        | 0x1F300..=0x1F64F
        | 0x1F680..=0x1F6FF
        | 0x1F900..=0x1F9FF
        | 0x1FA70..=0x1FAFF
        | 0x20000..=0x3FFFD => 2,
        _ => 1,
    }
}

/// Display width of a string in terminal columns.
pub fn str_width(s: &str) -> usize {
    s.chars().map(|c| char_width(c) as usize).sum()
}

/// Marks the second cell of a double-width character; never written to the terminal.
const WIDE_TAIL: char = '\0';

/// Cell grid flushed to the terminal in a single write.
pub struct Canvas {
    pub width: u16,
    pub height: u16,
    needs_clear: bool,
    grid: Vec<Cell>,
}

impl Canvas {
    pub fn new(width: u16, height: u16) -> Self {
        Self { width, height, needs_clear: true, grid: vec![Cell::default(); width as usize * height as usize] }
    }

    /// Repaints the whole screen on the next flush (ctrl+l).
    pub fn force_clear(&mut self) {
        self.needs_clear = true;
    }

    pub fn resize(&mut self, width: u16, height: u16) {
        if (self.width, self.height) != (width, height) {
            *self = Self::new(width, height);
        }
    }

    pub fn clear(&mut self) {
        self.grid.iter_mut().for_each(|c| *c = Cell::default());
    }

    #[inline]
    pub fn cell(&self, x: u16, y: u16) -> Option<&Cell> {
        (x < self.width && y < self.height).then(|| &self.grid[y as usize * self.width as usize + x as usize])
    }

    #[inline]
    pub fn put(&mut self, x: u16, y: u16, ch: char, fg: Rgb, bg: Rgb, bold: bool) {
        if x < self.width && y < self.height {
            self.grid[y as usize * self.width as usize + x as usize] = Cell { ch, fg, bg, bold, italic: false };
        }
    }

    /// Prints `text` from `x`, clipped at `max_x` (exclusive). Returns the next column.
    /// Double-width characters (emoji, CJK) take two cells, so later columns stay aligned.
    pub fn text(&mut self, mut x: u16, y: u16, max_x: u16, text: &str, fg: Rgb, bg: Rgb, bold: bool) -> u16 {
        let limit = max_x.min(self.width);
        for ch in text.chars() {
            let ch = if ch == '\t' || ch.is_control() { ' ' } else { ch };
            let w = char_width(ch);
            if w == 0 {
                continue;
            }
            if x + w > limit {
                break;
            }
            self.put(x, y, ch, fg, bg, bold);
            if w == 2 {
                self.put(x + 1, y, WIDE_TAIL, fg, bg, bold);
            }
            x += w;
        }
        x
    }

    pub fn fill(&mut self, x: u16, y: u16, w: u16, h: u16, bg: Rgb) {
        for row in y..y.saturating_add(h).min(self.height) {
            for col in x..x.saturating_add(w).min(self.width) {
                self.put(col, row, ' ', TEXT, bg, false);
            }
        }
    }

    pub fn flush(&mut self) -> io::Result<()> {
        use std::fmt::Write as _;
        let mut buf = String::with_capacity(self.grid.len() * 4);
        // Synchronized update: terminals that support it show the frame all at once.
        buf.push_str("\x1b[?2026h");
        buf.push_str(if self.needs_clear { "\x1b[2J\x1b[H" } else { "\x1b[H" });
        self.needs_clear = false;
        let mut cur: Option<(Rgb, Rgb, bool, bool)> = None;
        for y in 0..self.height {
            let row = &self.grid[y as usize * self.width as usize..(y as usize + 1) * self.width as usize];
            let last = row.iter().rposition(|c| c.ch != ' ' || c.bg != BG).map_or(0, |i| i + 1);
            for c in &row[..last] {
                if c.ch == WIDE_TAIL {
                    continue;
                }
                let style = (c.fg, c.bg, c.bold, c.italic);
                if cur != Some(style) {
                    let _ = write!(buf, "\x1b[0;38;2;{};{};{};48;2;{};{};{}m", c.fg.0, c.fg.1, c.fg.2, c.bg.0, c.bg.1, c.bg.2);
                    if c.bold {
                        buf.push_str("\x1b[1m");
                    }
                    if c.italic {
                        buf.push_str("\x1b[3m");
                    }
                    cur = Some(style);
                }
                buf.push(c.ch);
            }
            let _ = write!(buf, "\x1b[0;48;2;{};{};{}m\x1b[K", BG.0, BG.1, BG.2);
            cur = None;
            if y + 1 < self.height {
                buf.push_str("\r\n");
            }
        }
        buf.push_str("\x1b[?2026l");
        let mut out = io::stdout();
        out.write_all(buf.as_bytes())?;
        out.flush()
    }
}

// ------------------------------------------------------------------------------ input

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum KeyAction {
    Char(char),
    Paste(String),
    Backspace,
    Delete,
    Enter,
    /// Ctrl+J / pasted LF: insert a newline instead of sending.
    Newline,
    Left,
    Right,
    Home,
    End,
    Up,
    Down,
    PageUp,
    PageDown,
    ScrollUp(usize),
    ScrollDown(usize),
    Tab,
    CtrlC,
    CtrlD,
    CtrlP,
    /// Delete from the cursor to the start of the line.
    CtrlU,
    /// Delete from the cursor to the end of the line.
    CtrlK,
    /// Delete the word before the cursor (also alt+backspace).
    CtrlW,
    /// Undo the last edit to the prompt.
    CtrlZ,
    /// Repaint the screen.
    CtrlL,
    /// Start / end of the current line (ctrl+a / ctrl+e).
    LineStart,
    LineEnd,
    /// Word left / right (ctrl+←/→, alt+b/f).
    WordLeft,
    WordRight,
    /// Jump to the top / bottom of the conversation (ctrl+home / ctrl+end).
    ScrollTop,
    ScrollBottom,
    Escape,
    None,
}

/// Reads stdin on its own thread so the UI loop never blocks.
pub struct InputReader {
    rx: mpsc::Receiver<KeyAction>,
}

impl InputReader {
    pub fn new() -> Self {
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let mut stdin = io::stdin();
            let mut chunk = [0u8; 4096];
            let mut pending: Vec<u8> = Vec::new();
            loop {
                let n = match stdin.read(&mut chunk) {
                    Ok(0) | Err(_) => break,
                    Ok(n) => n,
                };
                pending.extend_from_slice(&chunk[..n]);
                // A lone ESC may be the start of a sequence split across reads.
                if pending.ends_with(&[27]) {
                    std::thread::sleep(Duration::from_millis(25));
                    if has_console_input() {
                        if let Ok(m) = stdin.read(&mut chunk) {
                            pending.extend_from_slice(&chunk[..m]);
                        }
                    }
                }
                let (actions, used) = parse_input_bytes(&pending);
                pending.drain(..used);
                for a in actions {
                    if a != KeyAction::None && tx.send(a).is_err() {
                        return;
                    }
                }
            }
        });
        Self { rx }
    }

    pub fn poll(&self) -> KeyAction {
        self.rx.try_recv().unwrap_or(KeyAction::None)
    }
}

impl Default for InputReader {
    fn default() -> Self {
        Self::new()
    }
}

/// Parses raw VT input into actions. Returns `(actions, bytes_consumed)`; an incomplete
/// trailing escape sequence (or unfinished bracketed paste) is left for the next read.
/// Mouse clicks and unknown sequences are swallowed, never typed into the prompt.
pub fn parse_input_bytes(buf: &[u8]) -> (Vec<KeyAction>, usize) {
    const PASTE_START: &[u8] = b"\x1b[200~";
    const PASTE_END: &[u8] = b"\x1b[201~";
    let mut out = Vec::new();
    let mut i = 0;
    while i < buf.len() {
        let b = buf[i];
        match b {
            27 if buf[i..].starts_with(PASTE_START) => {
                let body = i + PASTE_START.len();
                let Some(end) = buf[body..].windows(PASTE_END.len()).position(|w| w == PASTE_END) else { break };
                let text = String::from_utf8_lossy(&buf[body..body + end]).replace("\r\n", "\n").replace('\r', "\n");
                out.push(KeyAction::Paste(text));
                i = body + end + PASTE_END.len();
            }
            3 => {
                out.push(KeyAction::CtrlC);
                i += 1;
            }
            4 => {
                out.push(KeyAction::CtrlD);
                i += 1;
            }
            13 => {
                out.push(KeyAction::Enter);
                i += 1;
            }
            10 => {
                out.push(KeyAction::Newline);
                i += 1;
            }
            8 | 127 => {
                out.push(KeyAction::Backspace);
                i += 1;
            }
            9 => {
                out.push(KeyAction::Tab);
                i += 1;
            }
            1 | 5 | 11 | 12 | 16 | 21 | 23 | 26 => {
                out.push(match b {
                    1 => KeyAction::LineStart,
                    5 => KeyAction::LineEnd,
                    11 => KeyAction::CtrlK,
                    12 => KeyAction::CtrlL,
                    16 => KeyAction::CtrlP,
                    21 => KeyAction::CtrlU,
                    23 => KeyAction::CtrlW,
                    _ => KeyAction::CtrlZ,
                });
                i += 1;
            }
            27 => {
                if i + 1 >= buf.len() {
                    out.push(KeyAction::Escape);
                    i += 1;
                    continue;
                }
                let next = buf[i + 1];
                // Alt+key arrives as ESC followed by the key.
                let alt = match next {
                    127 | 8 => Some(KeyAction::CtrlW),
                    13 => Some(KeyAction::Newline),
                    b'b' => Some(KeyAction::WordLeft),
                    b'f' => Some(KeyAction::WordRight),
                    _ => None,
                };
                if let Some(a) = alt {
                    out.push(a);
                    i += 2;
                    continue;
                }
                if next != b'[' && next != b'O' {
                    out.push(KeyAction::Escape);
                    i += 1;
                    continue;
                }
                let mut j = i + 2;
                while j < buf.len() && (0x20..=0x3F).contains(&buf[j]) {
                    j += 1;
                }
                if j >= buf.len() {
                    break;
                }
                let fin = buf[j];
                let params = std::str::from_utf8(&buf[i + 2..j]).unwrap_or("");
                let action = if let Some(sgr) = params.strip_prefix('<') {
                    let btn = sgr.split(';').next().and_then(|s| s.parse::<u32>().ok()).unwrap_or(0);
                    match btn & !(4 | 8 | 16) {
                        64 => KeyAction::ScrollUp(3),
                        65 => KeyAction::ScrollDown(3),
                        _ => KeyAction::None,
                    }
                } else if fin == b'M' && params.is_empty() {
                    if j + 3 >= buf.len() {
                        break;
                    }
                    let btn = buf[j + 1].saturating_sub(32);
                    j += 3;
                    match btn {
                        64 => KeyAction::ScrollUp(3),
                        65 => KeyAction::ScrollDown(3),
                        _ => KeyAction::None,
                    }
                } else {
                    let modified = params.contains(';');
                    match fin {
                        b'A' if modified => KeyAction::ScrollUp(2),
                        b'B' if modified => KeyAction::ScrollDown(2),
                        b'C' if modified => KeyAction::WordRight,
                        b'D' if modified => KeyAction::WordLeft,
                        b'H' if modified => KeyAction::ScrollTop,
                        b'F' if modified => KeyAction::ScrollBottom,
                        b'A' => KeyAction::Up,
                        b'B' => KeyAction::Down,
                        b'C' => KeyAction::Right,
                        b'D' => KeyAction::Left,
                        b'H' => KeyAction::Home,
                        b'F' => KeyAction::End,
                        b'~' if modified => match params.split(';').next() {
                            Some("1") | Some("7") => KeyAction::ScrollTop,
                            Some("4") | Some("8") => KeyAction::ScrollBottom,
                            Some("3") => KeyAction::Delete,
                            Some("5") => KeyAction::PageUp,
                            Some("6") => KeyAction::PageDown,
                            _ => KeyAction::None,
                        },
                        b'~' => match params {
                            "1" | "7" => KeyAction::Home,
                            "4" | "8" => KeyAction::End,
                            "3" => KeyAction::Delete,
                            "5" => KeyAction::PageUp,
                            "6" => KeyAction::PageDown,
                            _ => KeyAction::None,
                        },
                        _ => KeyAction::None,
                    }
                };
                out.push(action);
                i = j + 1;
            }
            b'[' if i + 1 < buf.len() && (buf[i + 1] == b'<' || buf[i + 1] == b'?') => {
                // Orphaned CSI body (its ESC was eaten): swallow it.
                let mut j = i + 2;
                while j < buf.len() && (0x20..=0x3F).contains(&buf[j]) {
                    j += 1;
                }
                if j >= buf.len() {
                    break;
                }
                i = j + 1;
            }
            b';' if i + 1 < buf.len() && buf[i + 1].is_ascii_digit() => {
                // Orphaned SGR mouse coordinates like ";77;27M".
                let mut j = i + 1;
                while j < buf.len() && (buf[j].is_ascii_digit() || buf[j] == b';') {
                    j += 1;
                }
                if j < buf.len() && (buf[j] == b'M' || buf[j] == b'm') {
                    i = j + 1;
                } else {
                    out.push(KeyAction::Char(';'));
                    i += 1;
                }
            }
            b if b >= 32 => {
                let len = match b {
                    0xC0..=0xDF => 2,
                    0xE0..=0xEF => 3,
                    0xF0..=0xF7 => 4,
                    _ => 1,
                };
                if i + len > buf.len() {
                    break;
                }
                if let Some(c) = std::str::from_utf8(&buf[i..i + len]).ok().and_then(|s| s.chars().next()) {
                    out.push(KeyAction::Char(c));
                }
                i += len;
            }
            _ => i += 1,
        }
    }
    (out, i)
}

/// Prompt text with a cursor (in chars). May hold newlines. Keeps an undo history of
/// earlier states for ctrl+z.
#[derive(Default)]
pub struct InputBuffer {
    pub text: String,
    pub cursor: usize,
    undo: Vec<(String, usize)>,
    /// True while a run of typed characters continues (one undo step per word).
    typing: bool,
}

impl InputBuffer {
    fn byte_at(&self, char_idx: usize) -> usize {
        self.text.char_indices().nth(char_idx).map_or(self.text.len(), |(i, _)| i)
    }
    fn chars(&self) -> Vec<char> {
        self.text.chars().collect()
    }
    /// Saves the current state as an undo step (skipped if nothing changed).
    pub fn snapshot(&mut self) {
        self.typing = false;
        if self.undo.last().map_or(true, |(t, _)| *t != self.text) {
            self.undo.push((self.text.clone(), self.cursor));
            if self.undo.len() > 200 {
                self.undo.remove(0);
            }
        }
    }
    pub fn undo(&mut self) {
        self.typing = false;
        while let Some((t, c)) = self.undo.pop() {
            if t != self.text {
                self.text = t;
                self.cursor = c;
                return;
            }
        }
    }
    pub fn insert(&mut self, ch: char) {
        // A new undo step at the start of each word.
        if !self.typing || ch.is_whitespace() {
            self.snapshot();
            self.typing = true;
        }
        let b = self.byte_at(self.cursor);
        self.text.insert(b, ch);
        self.cursor += 1;
    }
    pub fn insert_str(&mut self, s: &str) {
        self.snapshot();
        let b = self.byte_at(self.cursor);
        self.text.insert_str(b, s);
        self.cursor += s.chars().count();
    }
    pub fn backspace(&mut self) {
        if self.cursor > 0 {
            if !self.typing {
                self.snapshot();
                self.typing = true;
            }
            self.cursor -= 1;
            let b = self.byte_at(self.cursor);
            self.text.remove(b);
        }
    }
    pub fn delete(&mut self) {
        if self.cursor < self.text.chars().count() {
            self.snapshot();
            let b = self.byte_at(self.cursor);
            self.text.remove(b);
        }
    }
    /// Removes chars `from..to` (char indices) and leaves the cursor at `from`.
    fn remove_range(&mut self, from: usize, to: usize) {
        if from >= to {
            return;
        }
        self.snapshot();
        let (a, b) = (self.byte_at(from), self.byte_at(to));
        self.text.replace_range(a..b, "");
        self.cursor = from;
    }
    fn word_start_before(&self, mut i: usize) -> usize {
        let c = self.chars();
        while i > 0 && c[i - 1].is_whitespace() {
            i -= 1;
        }
        while i > 0 && !c[i - 1].is_whitespace() {
            i -= 1;
        }
        i
    }
    fn word_end_after(&self, mut i: usize) -> usize {
        let c = self.chars();
        while i < c.len() && c[i].is_whitespace() {
            i += 1;
        }
        while i < c.len() && !c[i].is_whitespace() {
            i += 1;
        }
        i
    }
    pub fn delete_word_back(&mut self) {
        let start = self.word_start_before(self.cursor);
        self.remove_range(start, self.cursor);
    }
    /// Start and end (char indices) of the line the cursor is on.
    fn line_bounds(&self) -> (usize, usize) {
        let c = self.chars();
        let mut s = self.cursor.min(c.len());
        while s > 0 && c[s - 1] != '\n' {
            s -= 1;
        }
        let mut e = self.cursor.min(c.len());
        while e < c.len() && c[e] != '\n' {
            e += 1;
        }
        (s, e)
    }
    pub fn kill_to_line_start(&mut self) {
        let (s, _) = self.line_bounds();
        self.remove_range(s, self.cursor);
    }
    pub fn kill_to_line_end(&mut self) {
        let (_, e) = self.line_bounds();
        self.remove_range(self.cursor, e);
    }
    pub fn move_left(&mut self) {
        self.typing = false;
        self.cursor = self.cursor.saturating_sub(1);
    }
    pub fn move_right(&mut self) {
        self.typing = false;
        self.cursor = (self.cursor + 1).min(self.text.chars().count());
    }
    pub fn move_word_left(&mut self) {
        self.typing = false;
        self.cursor = self.word_start_before(self.cursor);
    }
    pub fn move_word_right(&mut self) {
        self.typing = false;
        self.cursor = self.word_end_after(self.cursor);
    }
    pub fn move_line_start(&mut self) {
        self.typing = false;
        self.cursor = self.line_bounds().0;
    }
    pub fn move_line_end(&mut self) {
        self.typing = false;
        self.cursor = self.line_bounds().1;
    }
    pub fn move_home(&mut self) {
        self.move_line_start();
    }
    pub fn move_end(&mut self) {
        self.move_line_end();
    }
    pub fn on_first_line(&self) -> bool {
        !self.chars()[..self.cursor.min(self.text.chars().count())].contains(&'\n')
    }
    pub fn on_last_line(&self) -> bool {
        let c = self.chars();
        !c[self.cursor.min(c.len())..].contains(&'\n')
    }
    /// Moves the cursor to the same column on the previous (`up`) or next logical line.
    pub fn move_line(&mut self, up: bool) {
        self.typing = false;
        let (s, e) = self.line_bounds();
        let col = self.cursor - s;
        let c = self.chars();
        if up && s > 0 {
            let prev_end = s - 1;
            let mut prev_start = prev_end;
            while prev_start > 0 && c[prev_start - 1] != '\n' {
                prev_start -= 1;
            }
            self.cursor = prev_start + col.min(prev_end - prev_start);
        } else if !up && e < c.len() {
            let next_start = e + 1;
            let mut next_end = next_start;
            while next_end < c.len() && c[next_end] != '\n' {
                next_end += 1;
            }
            self.cursor = next_start + col.min(next_end - next_start);
        }
    }
    pub fn clear(&mut self) {
        if !self.text.is_empty() {
            self.snapshot();
        }
        self.text.clear();
        self.cursor = 0;
    }
    pub fn set_text(&mut self, s: &str) {
        self.snapshot();
        self.text = s.to_string();
        self.cursor = self.text.chars().count();
    }
}

// ------------------------------------------------------------------------- text utils

/// Word-wraps prose to `max_width` columns (long words are split).
pub fn wrap_text(text: &str, max_width: usize) -> Vec<String> {
    let w = max_width.max(8);
    let mut out = Vec::new();
    for raw in text.lines() {
        let line = raw.trim_end().replace('\t', "    ");
        if line.trim().is_empty() {
            out.push(String::new());
            continue;
        }
        let indent: String = line.chars().take_while(|c| *c == ' ').collect();
        let mut cur = indent.clone();
        let mut cur_len = indent.chars().count();
        for word in line.split_whitespace() {
            let wl = str_width(word);
            if cur_len > indent.len() && cur_len + 1 + wl > w {
                out.push(std::mem::take(&mut cur));
                cur = indent.clone();
                cur_len = indent.chars().count();
            }
            if cur_len > indent.len() {
                cur.push(' ');
                cur_len += 1;
            }
            if wl > w.saturating_sub(cur_len) && wl > w / 2 {
                for ch in word.chars() {
                    let cw = char_width(ch) as usize;
                    if cur_len + cw > w {
                        out.push(std::mem::take(&mut cur));
                        cur_len = 0;
                    }
                    cur.push(ch);
                    cur_len += cw;
                }
            } else {
                cur.push_str(word);
                cur_len += wl;
            }
        }
        if !cur.trim().is_empty() {
            out.push(cur);
        }
    }
    out
}

/// Hard-wraps one line at exactly `w` columns, keeping spaces (for code and command output).
fn hard_wrap(line: &str, w: usize) -> Vec<String> {
    let line = line.replace('\t', "    ");
    let chars: Vec<char> = line.chars().collect();
    if chars.is_empty() {
        return vec![String::new()];
    }
    chars.chunks(w.max(1)).map(|c| c.iter().collect()).collect()
}

fn fmt_tokens(n: usize) -> String {
    if n >= 10_000 {
        format!("{:.0}K", n as f64 / 1000.0)
    } else if n >= 1000 {
        format!("{:.1}K", n as f64 / 1000.0)
    } else {
        n.to_string()
    }
}

/// The "SPINDLE" pixel logo: five rows per letter, colored top to bottom with
/// `theme::LOGO_ROWS`.
const LOGO: [[&str; 5]; 7] = [
    ["████", "█   ", "████", "   █", "████"],
    ["████", "█  █", "████", "█   ", "█   "],
    ["███", " █ ", " █ ", " █ ", "███"],
    ["█  █", "██ █", "█ ██", "█  █", "█  █"],
    ["███ ", "█  █", "█  █", "█  █", "███ "],
    ["█   ", "█   ", "█   ", "█   ", "████"],
    ["████", "█   ", "███ ", "█   ", "████"],
];
const LOGO_H: u16 = 5;

/// What the loading screen shows.
struct LoadingView {
    title: String,
    /// Phases finished before the full-screen UI opened (label, result).
    done_steps: Vec<(String, String)>,
    label: String,
    what: String,
    done: usize,
    total: usize,
    started: Instant,
}

/// Processes the system prompt while the loading screen shows its real progress. The work
/// runs on a scoped thread so the screen keeps animating between 128-token chunks (one
/// chunk can take a second or more on a two-PC setup).
fn prefill_with_loading(tui: &mut SpindleTui, engine: &mut SmolLm2Engine, title: &str, done_steps: &[(String, String)]) {
    if engine.n_keep != 0 {
        return;
    }
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    let done = AtomicUsize::new(0);
    let total = AtomicUsize::new(0);
    let finished = AtomicBool::new(false);
    let mut view = LoadingView {
        title: title.to_string(),
        done_steps: done_steps.to_vec(),
        label: "Prompt".into(),
        what: if engine.enable_tools { "system prompt and tool descriptions".into() } else { "system prompt".into() },
        done: 0,
        total: 0,
        started: Instant::now(),
    };
    let eng = &mut *engine;
    std::thread::scope(|sc| {
        let (done, total, finished) = (&done, &total, &finished);
        sc.spawn(move || {
            unsafe {
                eng.prefill_system_with(&mut |d, t| {
                    total.store(t, Ordering::Relaxed);
                    done.store(d, Ordering::Relaxed);
                });
            }
            finished.store(true, Ordering::Release);
        });
        while !finished.load(Ordering::Acquire) {
            view.done = done.load(Ordering::Relaxed);
            view.total = total.load(Ordering::Relaxed);
            tui.render_loading(&view);
            std::thread::sleep(Duration::from_millis(60));
        }
    });
    tui.info.ctx_used = engine.cur_pos;
}

fn spinner_frame(t0: Instant) -> char {
    const FRAMES: [char; 10] = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
    FRAMES[(t0.elapsed().as_millis() / 80) as usize % FRAMES.len()]
}

/// Splits a raw (streaming) reply into `(thinking, visible)` and drops tool-call markup;
/// `pending_tool` is true while a `<tool_call>` block is still being written.
fn split_reply(raw: &str) -> (String, String, bool) {
    let mut thinking = String::new();
    let mut visible = String::new();
    let mut rest = raw;
    if let Some(s) = rest.find("<think>") {
        visible.push_str(&rest[..s]);
        let after = &rest[s + 7..];
        match after.find("</think>") {
            Some(e) => {
                thinking.push_str(&after[..e]);
                rest = &after[e + 8..];
            }
            None => {
                thinking.push_str(after);
                rest = "";
            }
        }
    }
    let mut pending = false;
    let mut r = rest;
    while let Some(s) = r.find("<tool_call>") {
        visible.push_str(&r[..s]);
        match r[s..].find("</tool_call>") {
            Some(e) => r = &r[s + e + 12..],
            None => {
                pending = true;
                r = "";
            }
        }
    }
    visible.push_str(r);
    // A tag still being streamed token by token ("<tool_c…") is not shown yet.
    let partial = crate::tools::partial_tag_suffix(&visible, "<tool_call>");
    if partial >= 2 {
        visible.truncate(visible.len() - partial);
        pending = true;
    }
    (thinking.trim().to_string(), visible.trim().to_string(), pending)
}

// ------------------------------------------------------------------- conversation model

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ToolState {
    Running,
    Ok,
    Failed,
}

#[derive(Clone, Debug)]
enum Entry {
    User(String),
    /// Raw reply text (with think / tool-call markup; filtered at render time).
    Assistant { raw: String, streaming: bool },
    Tool { icon: &'static str, title: String, state: ToolState, error: Option<String>, preview: Option<String>, block: bool },
    /// `◆ Agent · model · 3.2s · 11.4 tok/s · ...`
    Footer { agent: String, detail: String },
    Notice { text: String, error: bool },
}

/// One styled run of text.
#[derive(Clone, Debug)]
struct Span {
    text: String,
    fg: Rgb,
    bold: bool,
    italic: bool,
}

fn span(text: impl Into<String>, fg: Rgb) -> Span {
    Span { text: text.into(), fg, bold: false, italic: false }
}

/// One rendered row of the transcript.
#[derive(Clone, Debug, Default)]
struct Line {
    spans: Vec<Span>,
    /// Background for the whole row (panels: user messages, tool blocks, code).
    bg: Option<Rgb>,
    /// `│` drawn in the first column in this color.
    bar: Option<Rgb>,
}

impl Line {
    fn blank() -> Self {
        Self::default()
    }
    fn of(spans: Vec<Span>) -> Self {
        Self { spans, ..Default::default() }
    }
}

/// Minimal markdown: fenced code, headings, bullets, `inline code` and **bold**.
fn markdown_lines(text: &str, width: usize, out: &mut Vec<Line>) {
    let mut in_code = false;
    for raw in text.lines() {
        let trimmed = raw.trim_start();
        if trimmed.starts_with("```") {
            in_code = !in_code;
            let lang = trimmed.trim_start_matches('`').trim();
            if in_code && !lang.is_empty() {
                out.push(Line { spans: vec![span(format!(" {lang}"), MUTED)], bg: Some(PANEL), bar: None });
            }
            continue;
        }
        if in_code {
            for part in hard_wrap(raw, width.saturating_sub(2)) {
                out.push(Line { spans: vec![span(format!(" {part}"), TEXT)], bg: Some(PANEL), bar: None });
            }
            continue;
        }
        if let Some(h) = trimmed.strip_prefix('#') {
            let title = h.trim_start_matches('#').trim();
            for part in wrap_text(title, width) {
                out.push(Line::of(vec![Span { text: part, fg: ACCENT, bold: true, italic: false }]));
            }
            continue;
        }
        let (lead, body) = match trimmed.strip_prefix("- ").or_else(|| trimmed.strip_prefix("* ")) {
            Some(b) => (format!("{}• ", " ".repeat(raw.len() - trimmed.len())), b),
            None => (String::new(), raw),
        };
        let lead_w = lead.chars().count();
        for (i, part) in wrap_text(body, width.saturating_sub(lead_w)).into_iter().enumerate() {
            let mut spans = vec![span(if i == 0 { lead.clone() } else { " ".repeat(lead_w) }, MUTED)];
            spans.extend(inline_markdown(&part));
            out.push(Line::of(spans));
        }
        if body.trim().is_empty() {
            out.push(Line::blank());
        }
    }
}

/// `inline code` in cyan, **bold** in bold.
fn inline_markdown(s: &str) -> Vec<Span> {
    let mut out = Vec::new();
    let mut code = false;
    let mut bold = false;
    let mut cur = String::new();
    let mut chars = s.chars().peekable();
    let flush = |cur: &mut String, out: &mut Vec<Span>, code: bool, bold: bool| {
        if !cur.is_empty() {
            out.push(Span { text: std::mem::take(cur), fg: if code { SECONDARY } else { TEXT }, bold, italic: false });
        }
    };
    while let Some(c) = chars.next() {
        if c == '`' {
            flush(&mut cur, &mut out, code, bold);
            code = !code;
        } else if c == '*' && chars.peek() == Some(&'*') && !code {
            chars.next();
            flush(&mut cur, &mut out, code, bold);
            bold = !bold;
        } else {
            cur.push(c);
        }
    }
    flush(&mut cur, &mut out, code, bold);
    out
}

impl Entry {
    fn layout(&self, width: usize, show_think: bool, t0: Instant, out: &mut Vec<Line>) {
        let w = width.max(10);
        match self {
            Entry::User(text) => {
                out.push(Line::blank());
                for (i, l) in wrap_text(text, w.saturating_sub(4)).into_iter().enumerate() {
                    let lead = Span { text: if i == 0 { " › ".into() } else { "   ".into() }, fg: PRIMARY, bold: true, italic: false };
                    out.push(Line::of(vec![lead, Span { text: l, fg: TEXT, bold: true, italic: false }]));
                }
            }
            Entry::Assistant { raw, streaming } => {
                let (thinking, visible, pending_tool) = split_reply(raw);
                let ind = 3;
                let mut body = Vec::new();
                if !thinking.is_empty() {
                    body.push(Line::blank());
                    let lines = wrap_text(&thinking, w - ind - 10);
                    let keep = if show_think { lines.len() } else { 3.min(lines.len()) };
                    for (i, l) in lines.iter().take(keep).enumerate() {
                        let lead = if i == 0 { "Thinking: " } else { "          " };
                        body.push(Line::of(vec![span(lead, MUTED), Span { text: l.clone(), fg: MUTED, bold: false, italic: true }]));
                    }
                    if lines.len() > keep {
                        body.push(Line::of(vec![span(format!("          … {} more lines (/think show)", lines.len() - keep), MUTED)]));
                    }
                }
                if !visible.is_empty() {
                    body.push(Line::blank());
                    markdown_lines(&visible, w - ind, &mut body);
                } else if *streaming && thinking.is_empty() && !pending_tool {
                    body.push(Line::blank());
                    body.push(Line::of(vec![span(format!("{} ", spinner_frame(t0)), PRIMARY), span("Thinking…", MUTED)]));
                }
                if pending_tool && *streaming {
                    body.push(Line::blank());
                    body.push(Line::of(vec![span(format!("{} ", spinner_frame(t0)), MUTED), span("Preparing tool call…", MUTED)]));
                }
                for mut l in body {
                    if l.bg.is_some() {
                        // Code panel: indent by padding inside the panel area.
                        l.spans.insert(0, span("", TEXT));
                    }
                    l.spans.insert(0, span(" ".repeat(ind), TEXT));
                    out.push(l);
                }
            }
            Entry::Tool { icon, title, state, error, preview, block } => {
                let color = match state {
                    ToolState::Failed => ERROR,
                    _ => MUTED,
                };
                let glyph = if *state == ToolState::Running { spinner_frame(t0).to_string() } else { icon.to_string() };
                if *block {
                    out.push(Line::blank());
                    let row = |spans: Vec<Span>| Line { spans, bg: Some(PANEL), bar: Some(BORDER) };
                    out.push(row(vec![]));
                    let head = if *icon == "❯" {
                        vec![span(format!("  {glyph} "), if *state == ToolState::Running { MUTED } else { TEXT }), span(title.clone(), TEXT)]
                    } else {
                        vec![span(format!("  {glyph} "), color), span(title.clone(), color)]
                    };
                    let head_text: String = head.iter().map(|s| s.text.as_str()).collect();
                    for (i, l) in hard_wrap(&head_text, w.saturating_sub(4)).into_iter().take(3).enumerate() {
                        out.push(row(if i == 0 { head.clone() } else { vec![span(format!("    {l}"), TEXT)] }));
                    }
                    if let Some(p) = preview {
                        out.push(row(vec![]));
                        for pl in p.lines().take(24) {
                            let (fg, bg) = match pl.chars().next() {
                                Some('+') if *icon != "❯" => (DIFF_ADD, Some(DIFF_ADD_BG)),
                                Some('-') if *icon != "❯" => (DIFF_DEL, Some(DIFF_DEL_BG)),
                                _ => (MUTED, None),
                            };
                            for part in hard_wrap(pl, w.saturating_sub(6)) {
                                out.push(Line { spans: vec![span("  ", TEXT), span(part, fg)], bg: bg.or(Some(PANEL)), bar: Some(BORDER) });
                            }
                        }
                    }
                    if let Some(e) = error {
                        for l in wrap_text(e, w.saturating_sub(6)).into_iter().take(4) {
                            out.push(row(vec![span(format!("  {l}"), ERROR)]));
                        }
                    }
                    out.push(row(vec![]));
                } else {
                    out.push(Line::of(vec![span(format!("   {glyph} "), color), span(title.clone(), color)]));
                    if let Some(e) = error {
                        for l in wrap_text(e, w.saturating_sub(8)).into_iter().take(3) {
                            out.push(Line::of(vec![span(format!("     {l}"), ERROR)]));
                        }
                    }
                }
            }
            Entry::Footer { agent, detail } => {
                out.push(Line::blank());
                out.push(Line::of(vec![span("   ◆ ", PRIMARY), span(agent.clone(), ACCENT), span(format!(" · {detail}"), MUTED)]));
            }
            Entry::Notice { text, error } => {
                out.push(Line::blank());
                for l in text.lines() {
                    for part in wrap_text(l, w.saturating_sub(4)) {
                        out.push(Line::of(vec![span(format!("   {part}"), if *error { ERROR } else { MUTED })]));
                    }
                }
            }
        }
    }
}

// --------------------------------------------------------------------- slash commands

/// `(command, description)`, shown in `/help` and the autocomplete popup.
/// Key reference shown by /help.
const KEYS_HELP: &str = "
Keys
  enter              send                 ctrl+j / alt+enter  new line
  tab                complete a command   ↑ / ↓               history (from an empty prompt)
  ctrl+a / ctrl+e    start / end of line  ctrl+← / ctrl+→     previous / next word
  ctrl+w             delete word          ctrl+u / ctrl+k     delete to line start / end
  ctrl+z             undo an edit         esc                 interrupt reply / clear prompt
  wheel, pgup/pgdn   scroll               ctrl+home / ctrl+end  top / bottom
  ctrl+l             redraw               ctrl+c              quit (on an empty prompt)
  shift+drag         select text with the mouse";

const COMMANDS: &[(&str, &str)] = &[
    ("/help", "show commands and keys"),
    ("/new", "start a new conversation (clears the KV-cache)"),
    ("/tools", "toggle agent tools: /tools on|off"),
    ("/think", "reasoning: /think on|off|show|hide"),
    ("/undo", "revert the last file the agent changed"),
    ("/diff", "git diff --stat of the working tree"),
    ("/review", "ask the model to review the uncommitted diff"),
    ("/compact", "summarize the conversation and restart from the summary"),
    ("/init", "create an AGENTS.md / SPINDLE.md with project instructions"),
    ("/model", "model, memory and KV-cache details"),
    ("/cluster", "cluster status and round-trip latency"),
    ("/timing", "where the last reply's time went, per forward pass"),
    ("/spec", "speculative decoding: /spec on|off|<draft len>"),
    ("/moe-prune", "MoE expert pruning: /moe-prune <n experts | fraction>"),
    ("/system", "show or set the system prompt"),
    ("/max", "max tokens per reply: /max <n>"),
    ("/bench", "measure decode speed: /bench [tokens]"),
    ("/exit", "quit (also ctrl+c on an empty prompt)"),
];

fn command_matches(input: &str) -> Vec<(&'static str, &'static str)> {
    if !input.starts_with('/') || input.contains(char::is_whitespace) {
        return Vec::new();
    }
    let q = input.to_ascii_lowercase();
    COMMANDS.iter().copied().filter(|(c, _)| c.starts_with(&q)).collect()
}

// ---------------------------------------------------------------------------------- UI

/// Facts about the session shown in the prompt metadata, status row and sidebar.
struct SessionInfo {
    model: String,
    cluster: Option<String>,
    local_layers: String,
    worker_layers: Option<String>,
    wire: Option<String>,
    ctx_max: usize,
    ctx_used: usize,
    tools: bool,
    think: bool,
    speculate: bool,
    last_tps: Option<f64>,
    spec: (usize, usize),
    branch: Option<String>,
    /// Timing of the last reply, for /timing: (breakdown, tokens generated).
    last_timing: Option<(crate::agent::Breakdown, usize)>,
}

enum UiEvent {
    Agent(AgentEvent),
    Confirm(String, mpsc::Sender<bool>),
    Done(TurnStats, usize),
}

pub struct SpindleTui {
    canvas: Canvas,
    input: InputBuffer,
    entries: Vec<Entry>,
    prompt_history: Vec<String>,
    history_idx: Option<usize>,
    pub scroll_offset: usize,
    pub running: bool,
    info: SessionInfo,
    show_think: bool,
    reader: InputReader,
    t0: Instant,
    /// Highlighted row in the slash-command popup.
    popup_sel: usize,
    /// Pending permission request from the agent: (description, reply channel).
    confirm: Option<(String, mpsc::Sender<bool>)>,
    generating: bool,
    /// A prompt typed while the model was busy; sent when the turn ends.
    queued: Option<String>,
    /// What was in the prompt before ↑ started browsing history (restored by ↓).
    draft: String,
    /// Transcript rows visible last frame (page size for pgup/pgdn).
    view_h: usize,
    /// Transcript length last frame, to keep a scrolled-up view still while text streams in.
    last_lines: usize,
}

impl SpindleTui {
    fn new(info: SessionInfo) -> Self {
        let (w, h) = terminal_size();
        Self {
            canvas: Canvas::new(w, h),
            input: InputBuffer::default(),
            entries: Vec::new(),
            prompt_history: Vec::new(),
            history_idx: None,
            scroll_offset: 0,
            running: true,
            info,
            show_think: false,
            reader: InputReader::new(),
            t0: Instant::now(),
            popup_sel: 0,
            confirm: None,
            generating: false,
            queued: None,
            draft: String::new(),
            view_h: 20,
            last_lines: 0,
        }
    }

    fn notice(&mut self, text: &str) {
        self.entries.push(Entry::Notice { text: text.to_string(), error: false });
        self.scroll_offset = 0;
    }

    fn error(&mut self, text: &str) {
        self.entries.push(Entry::Notice { text: text.to_string(), error: true });
        self.scroll_offset = 0;
    }

    fn git_branch() -> Option<String> {
        let out = std::process::Command::new("git").args(["rev-parse", "--abbrev-ref", "HEAD"]).output().ok()?;
        let b = String::from_utf8_lossy(&out.stdout).trim().to_string();
        (out.status.success() && !b.is_empty()).then_some(b)
    }

    fn cwd_label(&self) -> String {
        let cwd = std::env::current_dir().map(|p| p.display().to_string()).unwrap_or_else(|_| ".".into());
        let cwd = match std::env::var("USERPROFILE") {
            Ok(home) if cwd.starts_with(&home) => format!("~{}", &cwd[home.len()..]),
            _ => cwd,
        };
        match &self.info.branch {
            Some(b) => format!("{cwd}:{b}"),
            None => cwd,
        }
    }

    // ------------------------------------------------------------------- rendering

    fn draw_line(&mut self, x0: u16, y: u16, width: u16, line: &Line) {
        let max_x = x0 + width;
        if let Some(bg) = line.bg {
            self.canvas.fill(x0, y, width, 1, bg);
        }
        let bg = line.bg.unwrap_or(BG);
        let mut x = x0;
        if let Some(bar) = line.bar {
            self.canvas.put(x0, y, '│', bar, BG, false);
            x += 1;
        }
        for s in &line.spans {
            let start = x;
            x = self.canvas.text(x, y, max_x, &s.text, s.fg, bg, s.bold);
            if s.italic {
                for cx in start..x {
                    if let Some(i) = (cx < self.canvas.width).then(|| y as usize * self.canvas.width as usize + cx as usize) {
                        self.canvas.grid[i].italic = true;
                    }
                }
            }
        }
    }

    /// Prompt box: a rounded border whose top edge names the mode, model and where it runs;
    /// a `›` marker before the input. Turns amber while a permission request is open.
    /// Returns its height (input rows + 4). `y` is its top row.
    fn draw_prompt(&mut self, x0: u16, y: u16, w: u16, rows: &[String], cursor: (usize, usize)) -> u16 {
        let h = rows.len() as u16 + 4;
        let tx = x0 + 4;
        let max_x = x0 + w.saturating_sub(2);
        if let Some((what, _)) = self.confirm.clone() {
            self.draw_box(x0, y, w, h, WARNING, PANEL, &[("Permission required".into(), WARNING, true)]);
            let line: String = what.chars().take((max_x - tx) as usize).collect();
            self.canvas.text(x0 + 2, y + 1, max_x, "! ", WARNING, PANEL, true);
            self.canvas.text(tx, y + 1, max_x, &line, TEXT, PANEL, false);
            let ky = y + h - 2;
            let mut x = self.canvas.text(tx, ky, max_x, "enter", TEXT, PANEL, true);
            x = self.canvas.text(x, ky, max_x, " allow   ", MUTED, PANEL, false);
            x = self.canvas.text(x, ky, max_x, "esc", TEXT, PANEL, true);
            self.canvas.text(x, ky, max_x, " deny", MUTED, PANEL, false);
            return h;
        }
        let mode = if self.info.tools { "Agent" } else { "Chat" };
        let mut label = vec![(mode.to_string(), PRIMARY, true), (self.info.model.clone(), TEXT, false)];
        label.push(match &self.info.cluster {
            Some(peer) => (format!("cluster {peer}"), MUTED, false),
            None => ("local".into(), MUTED, false),
        });
        if self.info.think {
            label.push(("think".into(), ACCENT, false));
        }
        let border = if self.generating { BORDER_SUBTLE } else { BORDER };
        self.draw_box(x0, y, w, h, border, PANEL, &label);
        let iy = y + 2;
        self.canvas.text(x0 + 2, iy, max_x, "›", PRIMARY, PANEL, true);
        if self.input.text.is_empty() {
            self.canvas.put(tx, iy, ' ', PANEL, TEXT, false);
            let ph = if self.entries.is_empty() { "Ask anything… \"What does this project do?\"" } else { "Ask anything…" };
            self.canvas.text(tx + 1, iy, max_x, ph, MUTED, PANEL, false);
        } else {
            for (i, r) in rows.iter().enumerate() {
                self.canvas.text(tx, iy + i as u16, max_x, r, TEXT, PANEL, false);
            }
            let (cr, cc) = cursor;
            let cx = tx + cc as u16;
            let cy = iy + cr as u16;
            let under = self.canvas.cell(cx, cy).map_or(' ', |c| c.ch);
            self.canvas.put(cx, cy, under, PANEL, TEXT, false);
        }
        h
    }

    /// Wraps the prompt text to `w` columns; returns the visible rows (max 6) and the
    /// cursor's (row, col) within them.
    fn prompt_rows(&self, w: usize) -> (Vec<String>, (usize, usize)) {
        let w = w.max(4);
        let mut rows: Vec<String> = Vec::new();
        let mut cursor = (0, 0);
        let mut idx = 0;
        for line in self.input.text.split('\n') {
            let chars: Vec<char> = line.chars().collect();
            let chunks: Vec<&[char]> = if chars.is_empty() { vec![&[][..]] } else { chars.chunks(w).collect() };
            for (ci, chunk) in chunks.iter().enumerate() {
                let start = idx + ci * w;
                let end = start + chunk.len();
                let last_chunk = ci + 1 == chunks.len();
                if self.input.cursor >= start && (self.input.cursor < end || (last_chunk && self.input.cursor == end)) {
                    cursor = (rows.len(), self.input.cursor - start);
                    if cursor.1 >= w {
                        cursor = (rows.len() + 1, 0);
                    }
                }
                rows.push(chunk.iter().collect());
            }
            idx += chars.len() + 1;
        }
        if cursor.0 >= rows.len() {
            rows.push(String::new());
        }
        let max_rows = 6;
        let first = cursor.0.saturating_sub(max_rows - 1);
        let visible: Vec<String> = rows.into_iter().skip(first).take(max_rows).collect();
        (visible, (cursor.0 - first, cursor.1))
    }

    /// Row under the prompt: spinner + `esc interrupt` while busy, else the directory;
    /// context usage on the right.
    fn draw_status(&mut self, x0: u16, y: u16, w: u16) {
        let max_x = x0 + w;
        if self.generating {
            let mut x = self.canvas.text(x0 + 1, y, max_x, &format!("{} ", spinner_frame(self.t0)), PRIMARY, BG, false);
            x = self.canvas.text(x, y, max_x, "esc", TEXT, BG, true);
            x = self.canvas.text(x, y, max_x, " interrupt", MUTED, BG, false);
            if self.queued.is_some() {
                self.canvas.text(x, y, max_x, "   1 message queued", ACCENT, BG, false);
            }
        } else {
            let cwd = self.cwd_label();
            self.canvas.text(x0 + 1, y, max_x.saturating_sub(24), &cwd, MUTED, BG, false);
        }
        let used = self.info.ctx_used;
        let pct = used * 100 / self.info.ctx_max.max(1);
        let right = match self.info.last_tps {
            Some(t) => format!("{} ({pct}%) · {t:.1} tok/s", fmt_tokens(used)),
            None => format!("{} ({pct}%)", fmt_tokens(used)),
        };
        let rx = (x0 + w).saturating_sub(right.chars().count() as u16 + 1);
        self.canvas.text(rx, y, max_x, &right, MUTED, BG, false);
    }

    /// Slash-command popup drawn just above the prompt.
    fn draw_popup(&mut self, x0: u16, bottom: u16, w: u16) {
        let matches = command_matches(&self.input.text);
        if matches.is_empty() || self.confirm.is_some() {
            return;
        }
        self.popup_sel = self.popup_sel.min(matches.len() - 1);
        let n = matches.len().min(8) as u16;
        let top = bottom.saturating_sub(n);
        let first = self.popup_sel.saturating_sub(7);
        for (i, (cmd, desc)) in matches.iter().skip(first).take(8).enumerate() {
            let y = top + i as u16;
            let sel = first + i == self.popup_sel;
            let bg = if sel { PRIMARY } else { ELEMENT };
            self.canvas.fill(x0, y, w, 1, bg);
            let x = self.canvas.text(x0 + 2, y, x0 + w, cmd, if sel { BG } else { TEXT }, bg, true);
            self.canvas.text(x.max(x0 + 16), y, x0 + w - 1, desc, if sel { BG } else { MUTED }, bg, false);
        }
    }

    fn draw_sidebar(&mut self, x0: u16, w: u16, h: u16) {
        self.canvas.fill(x0, 0, w, h, PANEL);
        let tx = x0 + 2;
        let max_x = x0 + w - 2;
        let mut y = 1;
        let title = self.entries.iter().find_map(|e| if let Entry::User(t) = e { Some(t.clone()) } else { None }).unwrap_or_else(|| "New session".into());
        for l in wrap_text(&title, (w - 4) as usize).into_iter().take(2) {
            self.canvas.text(tx, y, max_x, &l, TEXT, PANEL, true);
            y += 1;
        }
        y += 1;
        let section = |c: &mut Canvas, y: &mut u16, head: &str, rows: Vec<(String, Rgb)>| {
            if *y + 2 >= h {
                return;
            }
            c.text(tx, *y, max_x, head, ACCENT, PANEL, true);
            *y += 1;
            for (r, fg) in rows {
                if *y + 2 >= h {
                    break;
                }
                c.text(tx, *y, max_x, &r, fg, PANEL, false);
                *y += 1;
            }
            *y += 1;
        };
        let used = self.info.ctx_used;
        let pct = used * 100 / self.info.ctx_max.max(1);
        section(&mut self.canvas, &mut y, "Context", vec![
            (format!("{} / {} tokens", fmt_tokens(used), fmt_tokens(self.info.ctx_max)), MUTED),
            (format!("{pct}% used"), if pct > 85 { WARNING } else { MUTED }),
        ]);
        let mut speed = vec![(self.info.last_tps.map_or("—".into(), |t| format!("{t:.1} tok/s")), MUTED)];
        if self.info.speculate {
            let (a, d) = self.info.spec;
            speed.push((if d > 0 { format!("spec {a}/{d} ({}%)", a * 100 / d) } else { "spec on".into() }, MUTED));
        }
        section(&mut self.canvas, &mut y, "Speed", speed);
        let mut node = vec![(format!("local {}", self.info.local_layers), MUTED)];
        if let Some(peer) = &self.info.cluster {
            node.insert(0, (peer.clone(), SUCCESS));
            if let Some(wl) = &self.info.worker_layers {
                node.push((format!("worker {wl}"), MUTED));
            }
            if let Some(wire) = &self.info.wire {
                node.push((format!("wire {wire}"), MUTED));
            }
        }
        section(&mut self.canvas, &mut y, if self.info.cluster.is_some() { "Cluster" } else { "Node" }, node);
        let files = crate::tools::modified_files();
        if !files.is_empty() {
            let rows = files.into_iter().take(8).map(|f| (f, MUTED)).collect();
            section(&mut self.canvas, &mut y, "Modified Files", rows);
        }
        let ver = format!("spindle v{}", env!("CARGO_PKG_VERSION"));
        self.canvas.text(tx, h - 2, max_x, "• ", SUCCESS, PANEL, false);
        self.canvas.text(tx + 2, h - 2, max_x, &ver, MUTED, PANEL, false);
    }

    fn transcript(&self, width: usize) -> Vec<Line> {
        let mut lines = Vec::new();
        for e in &self.entries {
            e.layout(width, self.show_think, self.t0, &mut lines);
        }
        lines
    }

    fn render(&mut self) {
        let (w, h) = terminal_size();
        self.canvas.resize(w, h);
        self.canvas.clear();
        if self.entries.is_empty() {
            self.render_home(w, h);
        } else {
            self.render_session(w, h);
        }
        let _ = self.canvas.flush();
    }

    /// The pixel logo centered at row `top`, `LOGO_H` rows tall.
    fn draw_logo(&mut self, w: u16, top: u16) {
        let logo_w: u16 = LOGO.iter().map(|g| g[0].chars().count() as u16 + 1).sum::<u16>() - 1;
        let lx = w.saturating_sub(logo_w) / 2;
        for row in 0..LOGO_H {
            let fg = theme::LOGO_ROWS[row as usize];
            let mut x = lx;
            for g in LOGO.iter() {
                x = self.canvas.text(x, top + row, w, g[row as usize], fg, BG, false) + 1;
            }
        }
    }

    /// A rounded box filled with `bg`. `label` (if any) is set into the top border.
    fn draw_box(&mut self, x0: u16, y0: u16, w: u16, h: u16, border: Rgb, bg: Rgb, label: &[(String, Rgb, bool)]) {
        if w < 4 || h < 2 {
            return;
        }
        self.canvas.fill(x0 + 1, y0 + 1, w - 2, h - 2, bg);
        let (x1, y1) = (x0 + w - 1, y0 + h - 1);
        self.canvas.put(x0, y0, '╭', border, BG, false);
        self.canvas.put(x1, y0, '╮', border, BG, false);
        self.canvas.put(x0, y1, '╰', border, BG, false);
        self.canvas.put(x1, y1, '╯', border, BG, false);
        for x in x0 + 1..x1 {
            self.canvas.put(x, y0, '─', border, BG, false);
            self.canvas.put(x, y1, '─', border, BG, false);
        }
        for y in y0 + 1..y1 {
            self.canvas.put(x0, y, '│', border, bg, false);
            self.canvas.put(x1, y, '│', border, bg, false);
        }
        if !label.is_empty() {
            let mut x = self.canvas.text(x0 + 2, y0, x1 - 1, " ", border, BG, false);
            for (i, (t, fg, bold)) in label.iter().enumerate() {
                if i > 0 {
                    x = self.canvas.text(x, y0, x1 - 1, " · ", MUTED, BG, false);
                }
                x = self.canvas.text(x, y0, x1 - 1, t, *fg, BG, *bold);
            }
            self.canvas.text(x, y0, x1 - 1, " ", border, BG, false);
        }
    }

    fn draw_footer(&mut self, w: u16, h: u16) {
        let fy = h - 1;
        let cwd = self.cwd_label();
        self.canvas.text(2, fy, w.saturating_sub(20), &cwd, MUTED, BG, false);
        let ver = format!("v{}", env!("CARGO_PKG_VERSION"));
        self.canvas.text(w.saturating_sub(ver.len() as u16 + 2), fy, w, &ver, MUTED, BG, false);
    }

    /// Full-screen loading view: the startup phases already done, the current one with a
    /// spinner, and a progress bar driven by real counts (tokens processed).
    fn render_loading(&mut self, view: &LoadingView) {
        let (w, h) = terminal_size();
        self.canvas.resize(w, h);
        self.canvas.clear();
        let card_w = w.saturating_sub(4).min(80);
        let inner = card_w.saturating_sub(6);
        let steps = view.done_steps.len() as u16;
        // card: border, blank, steps, current, blank, bar, detail, blank, border
        let card_h = 2 + 1 + steps + 1 + 1 + 2 + 1;
        let top = h.saturating_sub(LOGO_H + 2 + card_h + 2) / 2;
        self.draw_logo(w, top);
        let bx = w.saturating_sub(card_w) / 2;
        let cy = top + LOGO_H + 2;
        self.draw_box(bx, cy, card_w, card_h, BORDER, PANEL, &[(view.title.clone(), TEXT, true)]);
        let x0 = bx + 3;
        let max_x = bx + card_w - 3;
        let mut y = cy + 2;
        for (label, result) in &view.done_steps {
            let x = self.canvas.text(x0, y, max_x, "✓ ", SUCCESS, PANEL, false);
            let x = self.canvas.text(x, y, max_x, &format!("{label:<9}"), MUTED, PANEL, false);
            self.canvas.text(x, y, max_x, result, MUTED, PANEL, false);
            y += 1;
        }
        let x = self.canvas.text(x0, y, max_x, &format!("{} ", spinner_frame(self.t0)), PRIMARY, PANEL, false);
        let x = self.canvas.text(x, y, max_x, &format!("{:<9}", view.label), TEXT, PANEL, true);
        self.canvas.text(x, y, max_x, &view.what, TEXT, PANEL, false);
        y += 2;
        // Progress bar: thin line, filled part in the primary color.
        let frac = if view.total > 0 { (view.done as f64 / view.total as f64).clamp(0.0, 1.0) } else { 0.0 };
        let pct = format!(" {:>3.0}%", frac * 100.0);
        let bar_w = inner.saturating_sub(pct.len() as u16) as usize;
        let filled = (frac * bar_w as f64).round() as usize;
        let bar: String = "█".repeat(filled);
        let rest: String = "░".repeat(bar_w - filled);
        let x = self.canvas.text(x0, y, max_x, &bar, PRIMARY, PANEL, false);
        let x = self.canvas.text(x, y, max_x, &rest, BORDER, PANEL, false);
        self.canvas.text(x, y, max_x, &pct, TEXT, PANEL, false);
        y += 1;
        let secs = view.started.elapsed().as_secs_f64();
        let mut detail = format!("{} / {} tokens", crate::loading::fmt_count(view.done), crate::loading::fmt_count(view.total));
        if view.done > 0 && secs > 0.5 {
            let rate = view.done as f64 / secs;
            detail.push_str(&format!(" · {rate:.0} tok/s"));
            let left = view.total.saturating_sub(view.done) as f64 / rate;
            if left >= 1.0 {
                detail.push_str(&format!(" · {} left", crate::loading::fmt_secs(left)));
            }
        } else {
            detail.push_str(&format!(" · {}", crate::loading::fmt_secs(secs)));
        }
        self.canvas.text(x0, y, max_x, &detail, MUTED, PANEL, false);
        let hint = "processed once per session; replies only process your new message";
        if cy + card_h + 2 < h - 1 {
            let hx = w.saturating_sub(hint.chars().count() as u16) / 2;
            self.canvas.text(hx, cy + card_h + 1, w, hint, MUTED, BG, false);
        }
        self.draw_footer(w, h);
        let _ = self.canvas.flush();
    }

    fn render_home(&mut self, w: u16, h: u16) {
        let box_w = w.saturating_sub(4).min(75);
        let (rows, cursor) = self.prompt_rows(box_w.saturating_sub(6) as usize);
        let prompt_h = rows.len() as u16 + 4;
        let block_h = LOGO_H + 2 + prompt_h;
        let top = h.saturating_sub(block_h + 2) / 2;
        self.draw_logo(w, top);
        let bx = w.saturating_sub(box_w) / 2;
        let py = top + LOGO_H + 2;
        self.draw_popup(bx, py, box_w);
        let ph = self.draw_prompt(bx, py, box_w, &rows, cursor);
        let hint_y = py + ph + 1;
        if hint_y + 2 < h {
            let hint = "enter send · ctrl+j newline · /help commands · ctrl+c quit";
            let hx = w.saturating_sub(hint.chars().count() as u16) / 2;
            self.canvas.text(hx, hint_y, w, hint, MUTED, BG, false);
        }
        self.draw_footer(w, h);
    }

    fn render_session(&mut self, w: u16, h: u16) {
        let sidebar_w: u16 = if w >= 110 { 42 } else { 0 };
        if sidebar_w > 0 {
            self.draw_sidebar(w - sidebar_w, sidebar_w, h);
        }
        let main_w = w - sidebar_w;
        let x0 = 2u16;
        let col_w = main_w.saturating_sub(4);
        let (rows, cursor) = self.prompt_rows(col_w.saturating_sub(6) as usize);
        let prompt_h = rows.len() as u16 + 4;
        let status_y = h - 1;
        let prompt_y = status_y.saturating_sub(prompt_h);
        let view_h = prompt_y.saturating_sub(1) as usize;

        let lines = self.transcript(col_w as usize);
        // Scrolled up while new text arrives below: keep the same rows on screen.
        if self.scroll_offset > 0 && lines.len() > self.last_lines {
            self.scroll_offset += lines.len() - self.last_lines;
        }
        self.last_lines = lines.len();
        self.view_h = view_h;
        let max_scroll = lines.len().saturating_sub(view_h);
        self.scroll_offset = self.scroll_offset.min(max_scroll);
        let end = lines.len() - self.scroll_offset;
        let start = end.saturating_sub(view_h);
        for (i, line) in lines[start..end].iter().enumerate() {
            self.draw_line(x0, i as u16, col_w, line);
        }
        if self.scroll_offset > 0 {
            let s = format!(" ↓ {} more lines · ctrl+end to jump back ", self.scroll_offset);
            let sx = (x0 + col_w).saturating_sub(s.chars().count() as u16);
            self.canvas.text(sx, 0, x0 + col_w, &s, BG, MUTED, false);
        }
        self.draw_popup(x0, prompt_y, col_w);
        self.draw_prompt(x0, prompt_y, col_w, &rows, cursor);
        self.draw_status(x0, status_y, col_w);
    }

    // ------------------------------------------------------------------------ input

    /// Applies a key to the prompt editor. Returns `Some(text)` when the user submits.
    fn handle_key(&mut self, key: KeyAction) -> Option<String> {
        let popup = command_matches(&self.input.text);
        let page = self.view_h.saturating_sub(2).max(1);
        // Any edit makes the text the user's own: ↓ no longer replaces it with history.
        let edits = matches!(
            key,
            KeyAction::Char(_)
                | KeyAction::Paste(_)
                | KeyAction::Newline
                | KeyAction::Backspace
                | KeyAction::Delete
                | KeyAction::CtrlU
                | KeyAction::CtrlK
                | KeyAction::CtrlW
                | KeyAction::CtrlZ
        );
        if edits {
            self.history_idx = None;
        }
        match key {
            KeyAction::Char(c) => {
                self.input.insert(c);
                self.popup_sel = 0;
            }
            KeyAction::Paste(s) => self.input.insert_str(&s),
            KeyAction::Newline => self.input.insert('\n'),
            KeyAction::Backspace => self.input.backspace(),
            KeyAction::Delete => self.input.delete(),
            KeyAction::Left => self.input.move_left(),
            KeyAction::Right => self.input.move_right(),
            KeyAction::Home | KeyAction::LineStart => self.input.move_line_start(),
            KeyAction::End | KeyAction::LineEnd => self.input.move_line_end(),
            KeyAction::WordLeft => self.input.move_word_left(),
            KeyAction::WordRight => self.input.move_word_right(),
            KeyAction::CtrlU => self.input.kill_to_line_start(),
            KeyAction::CtrlK => self.input.kill_to_line_end(),
            KeyAction::CtrlW => self.input.delete_word_back(),
            KeyAction::CtrlZ => self.input.undo(),
            KeyAction::CtrlL => self.canvas.force_clear(),
            KeyAction::ScrollUp(n) => self.scroll_offset += n,
            KeyAction::ScrollDown(n) => self.scroll_offset = self.scroll_offset.saturating_sub(n),
            KeyAction::PageUp => self.scroll_offset += page,
            KeyAction::PageDown => self.scroll_offset = self.scroll_offset.saturating_sub(page),
            KeyAction::ScrollTop => self.scroll_offset = usize::MAX / 2,
            KeyAction::ScrollBottom => self.scroll_offset = 0,
            KeyAction::Tab => {
                if let Some((cmd, _)) = popup.get(self.popup_sel) {
                    self.input.set_text(&format!("{cmd} "));
                }
            }
            KeyAction::CtrlP => {
                if self.input.text.is_empty() {
                    self.input.set_text("/");
                }
            }
            KeyAction::Up if !popup.is_empty() && self.history_idx.is_none() => {
                self.popup_sel = self.popup_sel.saturating_sub(1)
            }
            KeyAction::Down if !popup.is_empty() && self.history_idx.is_none() => {
                self.popup_sel = (self.popup_sel + 1).min(popup.len() - 1)
            }
            // In a multi-line prompt, ↑/↓ move between its lines first.
            KeyAction::Up if !self.input.on_first_line() => self.input.move_line(true),
            KeyAction::Down if !self.input.on_last_line() => self.input.move_line(false),
            KeyAction::Up => {
                // History only from an empty prompt (or while already browsing), so ↑ never
                // throws away something you typed.
                let browsing = self.history_idx.is_some();
                if !self.prompt_history.is_empty() && (browsing || self.input.text.is_empty()) {
                    if !browsing {
                        self.draft = self.input.text.clone();
                    }
                    let i = self.history_idx.map_or(self.prompt_history.len() - 1, |i| i.saturating_sub(1));
                    self.history_idx = Some(i);
                    let t = self.prompt_history[i].clone();
                    self.input.set_text(&t);
                    self.popup_sel = 0;
                }
            }
            KeyAction::Down => match self.history_idx {
                Some(i) if i + 1 < self.prompt_history.len() => {
                    self.history_idx = Some(i + 1);
                    let t = self.prompt_history[i + 1].clone();
                    self.input.set_text(&t);
                }
                Some(_) => {
                    self.history_idx = None;
                    let d = std::mem::take(&mut self.draft);
                    self.input.set_text(&d);
                }
                None => {}
            },
            KeyAction::Escape => {
                // Clears the prompt (ctrl+z brings it back).
                self.input.clear();
                self.history_idx = None;
            }
            KeyAction::Enter => {
                if let Some((cmd, _)) = popup.get(self.popup_sel) {
                    // Picking a command from the popup completes it; a complete one runs.
                    if self.input.text.trim() != *cmd {
                        self.input.set_text(cmd);
                        return None;
                    }
                }
                let text = self.input.text.trim().to_string();
                if !text.is_empty() {
                    if self.prompt_history.last() != Some(&text) {
                        self.prompt_history.push(text.clone());
                    }
                    self.history_idx = None;
                    self.draft.clear();
                    self.input.clear();
                    self.scroll_offset = 0;
                    return Some(text);
                }
            }
            KeyAction::CtrlC | KeyAction::CtrlD => {
                if self.input.text.is_empty() {
                    self.running = false;
                } else {
                    self.input.clear();
                }
            }
            KeyAction::None => {}
        }
        None
    }

    /// Idle loop: renders and edits the prompt until the user submits (or quits).
    fn read_prompt(&mut self) -> Option<String> {
        if let Some(q) = self.queued.take() {
            return Some(q);
        }
        let mut last = Instant::now() - Duration::from_secs(1);
        loop {
            // Handle everything that is waiting (fast typing, a burst of wheel events) before
            // drawing, so the screen never falls behind the keyboard.
            let mut changed = false;
            for _ in 0..512 {
                let key = self.reader.poll();
                if key == KeyAction::None {
                    break;
                }
                changed = true;
                if let Some(text) = self.handle_key(key) {
                    return Some(text);
                }
                if !self.running {
                    return None;
                }
            }
            if changed || last.elapsed() >= Duration::from_millis(250) {
                self.render();
                last = Instant::now();
            } else {
                std::thread::sleep(Duration::from_millis(12));
            }
        }
    }
}

// ------------------------------------------------------------------------- turn driver

/// Runs one agent turn on a worker thread while the UI stays live.
fn run_turn(tui: &mut SpindleTui, engine: &mut SmolLm2Engine, prompt: &str, opts: AgentOpts) {
    tui.entries.push(Entry::User(prompt.to_string()));
    tui.entries.push(Entry::Assistant { raw: String::new(), streaming: true });
    tui.scroll_offset = 0;
    tui.generating = true;
    let abort = Arc::new(AtomicBool::new(false));
    engine.abort_signal = Some(abort.clone());
    let (tx, rx) = mpsc::channel::<UiEvent>();

    std::thread::scope(|s| {
        let worker_abort = abort.clone();
        let eng = &mut *engine;
        let prompt = prompt.to_string();
        s.spawn(move || {
            crate::pin_current_thread_to_pcore();
            let tx_ev = tx.clone();
            let tx_cf = tx.clone();
            let mut confirm = move |what: &str| -> bool {
                let (rtx, rrx) = mpsc::channel();
                if tx_cf.send(UiEvent::Confirm(what.to_string(), rtx)).is_err() {
                    return false;
                }
                rrx.recv().unwrap_or(false)
            };
            let stats = agent::run_turn(eng, &prompt, opts, Some(&worker_abort), &mut |ev| {
                let _ = tx_ev.send(UiEvent::Agent(ev));
            }, &mut confirm);
            let _ = tx.send(UiEvent::Done(stats, eng.cur_pos));
        });

        let mut last = Instant::now();
        loop {
            let mut changed = false;
            let mut finished = None;
            while let Ok(ev) = rx.try_recv() {
                changed = true;
                match ev {
                    UiEvent::Agent(AgentEvent::Token(p)) => {
                        tui.info.ctx_used += 1;
                        if let Some(Entry::Assistant { raw, .. }) = tui.entries.last_mut() {
                            raw.push_str(&p);
                        }
                    }
                    UiEvent::Agent(AgentEvent::ReplyDone { .. }) => {
                        match tui.entries.last_mut() {
                            Some(Entry::Assistant { raw, streaming }) => {
                                *streaming = false;
                                let (t, v, _) = split_reply(raw);
                                if t.is_empty() && v.is_empty() {
                                    tui.entries.pop();
                                }
                            }
                            _ => {}
                        }
                    }
                    UiEvent::Agent(AgentEvent::ToolStart { icon, title }) => {
                        let block = matches!(icon, "❯");
                        tui.entries.push(Entry::Tool { icon, title, state: ToolState::Running, error: None, preview: None, block });
                    }
                    UiEvent::Agent(AgentEvent::ToolEnd(o)) => {
                        let block = o.icon == "❯" || (o.ok && o.preview.is_some());
                        let error = (!o.ok).then(|| o.output.lines().next().unwrap_or("").trim_start_matches("Error: ").to_string());
                        let entry = Entry::Tool {
                            icon: o.icon,
                            title: o.title.clone(),
                            state: if o.ok { ToolState::Ok } else { ToolState::Failed },
                            error: if o.icon == "❯" { None } else { error },
                            preview: o.preview.clone(),
                            block,
                        };
                        match tui.entries.last_mut() {
                            Some(e @ Entry::Tool { state: ToolState::Running, .. }) => *e = entry,
                            _ => tui.entries.push(entry),
                        }
                        // The model replies again after the tool results.
                        tui.entries.push(Entry::Assistant { raw: String::new(), streaming: true });
                    }
                    UiEvent::Confirm(what, reply) => tui.confirm = Some((what, reply)),
                    UiEvent::Done(stats, cur_pos) => finished = Some((stats, cur_pos)),
                }
            }
            if let Some((stats, cur_pos)) = finished {
                // Drop an empty trailing placeholder.
                if let Some(Entry::Assistant { raw, .. }) = tui.entries.last() {
                    let (t, v, _) = split_reply(raw);
                    if t.is_empty() && v.is_empty() {
                        tui.entries.pop();
                    }
                }
                if let Some(Entry::Assistant { streaming, .. }) = tui.entries.last_mut() {
                    *streaming = false;
                }
                tui.info.ctx_used = cur_pos;
                tui.info.last_tps = (stats.tokens > 0).then(|| stats.tok_per_sec());
                if stats.breakdown.passes > 0 {
                    tui.info.last_timing = Some((stats.breakdown, stats.tokens));
                }
                tui.info.spec = (tui.info.spec.0 + stats.spec_accepted, tui.info.spec.1 + stats.spec_drafted);
                let mut detail = vec![tui.info.model.clone(), format!("{:.1}s", stats.wall_secs)];
                if stats.tokens > 0 {
                    detail.push(format!("{:.1} tok/s", stats.tok_per_sec()));
                }
                if stats.spec_drafted > 0 {
                    detail.push(format!("spec {}/{}", stats.spec_accepted, stats.spec_drafted));
                }
                if stats.round_trips > 0 {
                    detail.push(format!("w {:.0}ms · l {:.0}ms", stats.worker_ms, stats.local_ms));
                }
                if stats.aborted {
                    detail.push("interrupted".into());
                }
                let agent = if tui.info.tools { "Agent" } else { "Chat" }.to_string();
                tui.entries.push(Entry::Footer { agent, detail: detail.join(" · ") });
                break;
            }

            for _ in 0..512 {
                let key = tui.reader.poll();
                if key == KeyAction::None {
                    break;
                }
                changed = true;
                if let Some((_, reply)) = tui.confirm.as_ref() {
                    match key {
                        KeyAction::Enter | KeyAction::Char('y') | KeyAction::Char('Y') => {
                            let _ = reply.send(true);
                            tui.confirm = None;
                        }
                        KeyAction::Escape | KeyAction::CtrlC | KeyAction::Char('n') | KeyAction::Char('N') => {
                            let _ = reply.send(false);
                            tui.confirm = None;
                        }
                        KeyAction::ScrollUp(_)
                        | KeyAction::ScrollDown(_)
                        | KeyAction::PageUp
                        | KeyAction::PageDown
                        | KeyAction::ScrollTop
                        | KeyAction::ScrollBottom => {
                            tui.handle_key(key);
                        }
                        _ => {}
                    }
                } else {
                    match key {
                        KeyAction::Escape | KeyAction::CtrlC => abort.store(true, Ordering::SeqCst),
                        other => {
                            if let Some(text) = tui.handle_key(other) {
                                tui.queued = Some(text);
                            }
                            tui.running = true; // ctrl+d during a turn does not quit
                        }
                    }
                }
            }
            if changed || last.elapsed() >= Duration::from_millis(80) {
                tui.render();
                last = Instant::now();
            } else {
                std::thread::sleep(Duration::from_millis(8));
            }
        }
    });
    engine.abort_signal = None;
    tui.generating = false;
    tui.render();
}

// ------------------------------------------------------------------------- entry point

pub fn run_tui(engine: &mut SmolLm2Engine, _governor: Option<&crate::memory_guard::Governor>, opts: crate::chat_cli::ChatOptions) {
    let guard = match TerminalGuard::enter() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("could not start the full-screen UI: {e}");
            return;
        }
    };
    let client = engine.cluster_client.as_ref();
    let info = SessionInfo {
        model: engine.idx.model_name.clone(),
        cluster: client.and_then(|c| c.stream.peer_addr().ok()).map(|a| a.to_string()),
        local_layers: format!("layers {}..{}", engine.layer_range.start, engine.layer_range.end),
        worker_layers: client.map(|c| format!("layers {}..{}", c.worker_range.0, c.worker_range.1)),
        wire: client.map(|c| format!("{:?}", c.wire_format).to_lowercase()),
        ctx_max: engine.max_seq,
        ctx_used: engine.cur_pos,
        tools: engine.enable_tools,
        think: opts.allow_think,
        speculate: engine.speculate,
        last_tps: None,
        last_timing: None,
        spec: (0, 0),
        branch: SpindleTui::git_branch(),
    };
    let mut tui = SpindleTui::new(info);
    let mut agent_opts = AgentOpts {
        max_tokens: if opts.max_tokens > 0 { opts.max_tokens } else { (engine.max_seq / 2).clamp(256, 4096) },
        allow_think: opts.allow_think,
        ..Default::default()
    };

    prefill_with_loading(&mut tui, engine, "Starting the session", &opts.startup);
    tui.render();
    if let Some(p) = opts.first_prompt.clone() {
        run_turn(&mut tui, engine, &p, agent_opts);
    }
    while tui.running {
        let Some(text) = tui.read_prompt() else { break };
        if matches!(text.as_str(), "/exit" | "/quit" | "/q") {
            break;
        }
        if text.starts_with('/') {
            if let Some(prompt) = handle_slash_command(&mut tui, engine, &text, &mut agent_opts) {
                run_turn(&mut tui, engine, &prompt, agent_opts);
            }
            continue;
        }
        run_turn(&mut tui, engine, &text, agent_opts);
    }
    drop(guard);
}

/// Runs a slash command. Returns a prompt to send to the model, if the command produces one.
fn handle_slash_command(tui: &mut SpindleTui, engine: &mut SmolLm2Engine, line: &str, opts: &mut AgentOpts) -> Option<String> {
    let (head, rest) = match line.split_once(char::is_whitespace) {
        Some((h, r)) => (h.to_ascii_lowercase(), r.trim().to_string()),
        None => (line.to_ascii_lowercase(), String::new()),
    };
    let restart_context = |tui: &mut SpindleTui, engine: &mut SmolLm2Engine| {
        engine.reset_kv();
        prefill_with_loading(tui, engine, "Starting a new conversation", &[]);
    };
    match head.as_str() {
        "/help" | "/?" => {
            let mut s = String::from("Commands\n");
            for (c, d) in COMMANDS {
                s.push_str(&format!("  {c:<12} {d}\n"));
            }
            s.push_str(KEYS_HELP);
            tui.notice(&s);
        }
        "/new" | "/clear" | "/reset" => {
            tui.entries.clear();
            restart_context(tui, engine);
        }
        "/tools" => {
            let on = match rest.as_str() {
                "on" => true,
                "off" => false,
                _ => !engine.enable_tools,
            };
            engine.set_tools_enabled(on);
            tui.info.tools = on;
            restart_context(tui, engine);
            tui.notice(if on { "Agent tools on (files, search, shell, web). Context restarted." } else { "Agent tools off. Context restarted." });
        }
        "/think" => match rest.as_str() {
            "show" => {
                tui.show_think = true;
                tui.notice("Reasoning shown in full.");
            }
            "hide" => {
                tui.show_think = false;
                tui.notice("Reasoning collapsed to 3 lines.");
            }
            r => {
                opts.allow_think = match r {
                    "on" => true,
                    "off" => false,
                    _ => !opts.allow_think,
                };
                tui.info.think = opts.allow_think;
                tui.notice(if opts.allow_think { "Reasoning on (<think> before answering)." } else { "Reasoning off (direct answers)." });
            }
        },
        "/undo" => match crate::tools::pop_undo() {
            Some(Ok(m)) => tui.notice(&m),
            Some(Err(e)) => tui.error(&e),
            None => tui.notice("Nothing to undo."),
        },
        "/diff" => match std::process::Command::new("git").args(["diff", "--stat"]).output() {
            Ok(o) if !o.stdout.is_empty() => tui.notice(String::from_utf8_lossy(&o.stdout).trim_end()),
            Ok(_) => tui.notice("Working tree clean."),
            Err(e) => tui.error(&format!("git: {e}")),
        },
        "/review" => match std::process::Command::new("git").args(["diff", "HEAD"]).output() {
            Ok(o) => {
                let diff = String::from_utf8_lossy(&o.stdout);
                if diff.trim().is_empty() {
                    tui.notice("No uncommitted changes to review.");
                } else {
                    return Some(format!(
                        "Review this uncommitted diff. List concrete bugs, regressions and risky changes first, then smaller suggestions.\n\n```diff\n{}\n```",
                        crate::tools::truncate_middle(diff.trim(), (engine.max_seq.saturating_sub(engine.cur_pos) * 2).clamp(2000, 12000))
                    ));
                }
            }
            Err(e) => tui.error(&format!("git: {e}")),
        },
        "/compact" => {
            tui.notice("Summarizing the conversation…");
            tui.render();
            let (summary, _, _) = unsafe {
                engine.generate_reply("Summarize our conversation so far in at most 8 short bullet points: goals, files touched, decisions, open tasks.", 400, false, &mut |_| {})
            };
            let base = engine.system_prompt.clone();
            engine.set_system_prompt(&format!("{base}\n\n# Summary of the earlier conversation\n{}", summary.trim()));
            restart_context(tui, engine);
            engine.set_system_prompt(&base);
            tui.entries.clear();
            tui.notice(&format!("Conversation compacted to {} tokens.\n\n{}", engine.cur_pos, summary.trim()));
        }
        "/init" => {
            let target = if std::path::Path::new("SPINDLE.md").exists() { "SPINDLE.md" } else { "AGENTS.md" };
            return Some(format!(
                "Look at this project (list_dir, read the build files and main sources), then write {target} with: what the project is, how to build/test/run it, the code layout, and conventions to follow. Keep it under 60 lines. If {target} exists, improve it instead of replacing useful content."
            ));
        }
        "/model" | "/info" => {
            let c = engine.cfg;
            let (ws, private, _) = crate::memory_guard::process_counters();
            tui.notice(&format!(
                "{}\n{} · dim {} · layers {}..{} of {} · heads {}/{}\nRAM: {} MiB file · {} MiB resident · {} MiB private · {} MiB free\nKV-cache: {}/{} tokens · {}",
                engine.idx.model_name,
                engine.idx.architecture,
                c.dim,
                engine.layer_range.start,
                engine.layer_range.end,
                c.n_layers,
                c.n_heads,
                c.n_kv_heads,
                engine.raw_bytes.len() >> 20,
                ws >> 20,
                private >> 20,
                crate::mmap_file::available_ram_bytes() >> 20,
                engine.cur_pos,
                engine.max_seq,
                crate::chat_cli::describe_prune(engine.moe_prune()),
            ));
        }
        "/timing" | "/stats" => match &tui.info.last_timing {
            Some((b, tokens)) => {
                let mut t = b.table(*tokens);
                if engine.speculate {
                    t.push_str(&format!("
Speculation: {}", engine.spec_model.summary()));
                }
                tui.notice(&t);
            }
            None => tui.notice("No reply yet. Send a message, then run /timing."),
        },
        "/cluster" | "/nodes" => match engine.cluster_client.as_mut() {
            Some(c) => {
                let rtt = c.ping().map(|d| format!("{:.2} ms", d.as_secs_f64() * 1e3)).unwrap_or_else(|e| format!("ping failed: {e}"));
                tui.notice(&format!(
                    "Worker layers {}..{} · local layers {}..{} · wire {:?}\nRound trip: {rtt} · context {} tokens (min of both nodes)",
                    c.worker_range.0, c.worker_range.1, engine.layer_range.start, engine.layer_range.end, c.wire_format, engine.max_seq
                ));
            }
            None => tui.notice("Standalone (no cluster peer). Set `peer` and `layer_range` in spindle.toml to use a worker."),
        },
        "/spec" => {
            match rest.as_str() {
                "on" => engine.speculate = true,
                "off" => engine.speculate = false,
                n => {
                    if let Ok(k) = n.parse::<usize>() {
                        engine.set_draft_len(k);
                        engine.speculate = true;
                    }
                }
            }
            tui.info.speculate = engine.speculate;
            tui.notice(&format!(
                "Speculative decoding (prompt lookup): {} · up to {} draft tokens per verify. Output is identical either way.",
                if engine.speculate { "on" } else { "off" },
                engine.draft_len
            ));
        }
        "/moe-prune" => {
            if let Ok(p) = rest.parse::<f32>() {
                engine.set_moe_prune(p);
            }
            tui.notice(&crate::chat_cli::describe_prune(engine.moe_prune()));
        }
        "/system" => {
            if rest.is_empty() {
                let shown = crate::tools::truncate_middle(&engine.system_prompt, 1500);
                tui.notice(&format!("System prompt:\n{shown}\n\n/system <text> to replace, /system reset for the default"));
            } else {
                let base = if rest == "reset" { engine.default_system_prompt() } else { rest.clone() };
                engine.set_system_prompt(&base);
                engine.set_tools_enabled(engine.enable_tools);
                restart_context(tui, engine);
                tui.notice("System prompt updated. Context restarted.");
            }
        }
        "/max" => match rest.parse::<usize>() {
            Ok(n) => {
                opts.max_tokens = n.clamp(16, engine.max_seq);
                tui.notice(&format!("Replies capped at {} tokens.", opts.max_tokens));
            }
            Err(_) => tui.notice(&format!("Replies capped at {} tokens. Usage: /max <n>", opts.max_tokens)),
        },
        "/bench" => {
            let n = rest.parse::<usize>().unwrap_or(64).clamp(8, 512);
            tui.notice(&format!("Generating {n} tokens…"));
            tui.render();
            let (_, secs, toks) = unsafe { engine.generate_reply("Write a short paragraph about how rivers shape valleys.", n, false, &mut |_| {}) };
            tui.info.ctx_used = engine.cur_pos;
            tui.notice(&format!("{:.1} tok/s ({toks} tokens in {secs:.2}s)", toks as f64 / secs.max(1e-3)));
        }
        other => tui.error(&format!("Unknown command {other}. Type /help.")),
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn info() -> SessionInfo {
        SessionInfo {
            model: "test-model".into(),
            cluster: Some("10.0.0.2:42424".into()),
            local_layers: "layers 24..48".into(),
            worker_layers: Some("layers 0..24".into()),
            wire: Some("fp8".into()),
            ctx_max: 4096,
            ctx_used: 100,
            tools: true,
            think: false,
            speculate: true,
            last_tps: Some(11.0),
            last_timing: None,
            spec: (3, 4),
            branch: None,
        }
    }

    #[test]
    fn input_buffer_utf8_editing() {
        let mut b = InputBuffer::default();
        for c in "kumusta".chars() {
            b.insert(c);
        }
        b.insert('ñ');
        assert_eq!((b.text.as_str(), b.cursor), ("kumustañ", 8));
        b.backspace();
        b.move_home();
        b.delete();
        assert_eq!(b.text, "umusta");
        b.move_end();
        b.insert_str("\nline 2");
        assert_eq!(b.text, "umusta\nline 2");
    }

    #[test]
    fn parse_input_handles_mouse_paste_and_ctrl_keys() {
        let (a, n) = parse_input_bytes(b"\x1b[<0;77;27M\x1b[<0;77;27m");
        assert_eq!(n, 22);
        assert!(a.iter().all(|k| *k == KeyAction::None));
        assert_eq!(parse_input_bytes(b"\x1b[<64;7;7M").0, vec![KeyAction::ScrollUp(3)]);
        assert_eq!(parse_input_bytes(b"[<0;77;27M").0.len(), 0);
        let text: String = parse_input_bytes(b"hello\x1b[<0;10;10M world").0.into_iter().filter_map(|a| if let KeyAction::Char(c) = a { Some(c) } else { None }).collect();
        assert_eq!(text, "hello world");
        // Bracketed paste: multi-line text arrives as one Paste, CRLF normalized; Enter not sent.
        let (a, n) = parse_input_bytes(b"\x1b[200~line1\r\nline2\x1b[201~\r");
        assert_eq!(n, 25);
        assert_eq!(a, vec![KeyAction::Paste("line1\nline2".into()), KeyAction::Enter]);
        // Unfinished paste waits for more bytes.
        assert_eq!(parse_input_bytes(b"\x1b[200~partial").1, 0);
        assert_eq!(parse_input_bytes(&[3, 4, 10]).0, vec![KeyAction::CtrlC, KeyAction::CtrlD, KeyAction::Newline]);
    }

    #[test]
    fn wheel_and_editing_keys() {
        // Wheel events never become arrow keys (which would recall history).
        assert_eq!(parse_input_bytes(b"\x1b[<65;7;7M").0, vec![KeyAction::ScrollDown(3)]);
        assert_eq!(parse_input_bytes(&[1, 5, 11, 21, 23, 26, 12]).0, vec![
            KeyAction::LineStart, KeyAction::LineEnd, KeyAction::CtrlK, KeyAction::CtrlU,
            KeyAction::CtrlW, KeyAction::CtrlZ, KeyAction::CtrlL,
        ]);
        assert_eq!(parse_input_bytes(b"\x1b[1;5D\x1b[1;5C\x1b[1;5H\x1b[1;5F").0, vec![
            KeyAction::WordLeft, KeyAction::WordRight, KeyAction::ScrollTop, KeyAction::ScrollBottom,
        ]);
        assert_eq!(parse_input_bytes(b"\x1b\x7f\x1b\r").0, vec![KeyAction::CtrlW, KeyAction::Newline]);
    }

    #[test]
    fn history_never_replaces_typed_text() {
        let mut tui = SpindleTui::new(info());
        tui.prompt_history = vec!["Hi".into(), "/model".into()];
        for c in "draft".chars() {
            tui.handle_key(KeyAction::Char(c));
        }
        // ↑ with text in the prompt does nothing.
        tui.handle_key(KeyAction::Up);
        assert_eq!(tui.input.text, "draft");
        // From an empty prompt ↑ browses, ↓ past the end restores what was there.
        tui.input.clear();
        tui.handle_key(KeyAction::Up);
        assert_eq!(tui.input.text, "/model");
        tui.handle_key(KeyAction::Up);
        assert_eq!(tui.input.text, "Hi");
        tui.handle_key(KeyAction::Down);
        tui.handle_key(KeyAction::Down);
        assert_eq!(tui.input.text, "");
        // Editing a recalled entry makes it yours: ↓ no longer replaces it.
        tui.handle_key(KeyAction::Up);
        tui.handle_key(KeyAction::Char('!'));
        tui.handle_key(KeyAction::Down);
        assert_eq!(tui.input.text, "/model!");
        // Wheel scrolling leaves the prompt alone.
        tui.handle_key(KeyAction::ScrollUp(3));
        assert_eq!(tui.input.text, "/model!");
    }

    #[test]
    fn prompt_editing_shortcuts_and_undo() {
        let mut b = InputBuffer::default();
        for c in "hello big world".chars() {
            b.insert(c);
        }
        b.delete_word_back();
        assert_eq!(b.text, "hello big ");
        b.move_word_left();
        assert_eq!(b.cursor, 6);
        b.kill_to_line_end();
        assert_eq!(b.text, "hello ");
        b.undo();
        assert_eq!(b.text, "hello big ");
        b.undo();
        assert_eq!(b.text, "hello big world");
        b.set_text("one\ntwo\nthree");
        b.move_line(true);
        assert!(!b.on_first_line() && !b.on_last_line());
        b.move_line_start();
        b.kill_to_line_end();
        assert_eq!(b.text, "one\n\nthree");
    }

    #[test]
    fn wide_characters_keep_columns_aligned() {
        let mut c = Canvas::new(20, 1);
        let end = c.text(0, 0, 20, "hi 😊 ok", TEXT, BG, false);
        assert_eq!(end, 8);
        assert_eq!(c.cell(4, 0).unwrap().ch, WIDE_TAIL);
        assert_eq!(c.cell(6, 0).unwrap().ch, 'o');
        assert_eq!(str_width("中文"), 4);
    }

    #[test]
    fn split_reply_hides_tool_markup_and_think() {
        let (t, v, p) = split_reply("<think>plan it</think>\n\nI'll read it.\n<tool_call>\n{\"name\":\"read_file\"");
        assert_eq!((t.as_str(), v.as_str(), p), ("plan it", "I'll read it.", true));
        let (_, v, p) = split_reply("a<tool_call>{}</tool_call>b");
        assert_eq!((v.as_str(), p), ("ab", false));
    }

    #[test]
    fn wrap_text_respects_width() {
        let s = "I am a large language model that can help with answering questions, generating code and automating developer workflows.";
        for l in wrap_text(s, 40) {
            assert!(l.chars().count() <= 40, "{l:?}");
        }
        assert_eq!(wrap_text("    indented code line", 40), vec!["    indented code line"]);
    }

    #[test]
    fn transcript_layout_and_full_render() {
        let mut tui = SpindleTui::new(info());
        tui.canvas.resize(120, 30);
        tui.entries.push(Entry::User("Fix the bug in `main.rs`".into()));
        tui.entries.push(Entry::Assistant { raw: "Sure.\n```rust\nfn main() {}\n```\n- one **two** `three`".into(), streaming: false });
        tui.entries.push(Entry::Tool { icon: "▸", title: "Read src/main.rs".into(), state: ToolState::Ok, error: None, preview: None, block: false });
        tui.entries.push(Entry::Tool { icon: "✎", title: "Edit src/main.rs".into(), state: ToolState::Ok, error: None, preview: Some("-old\n+new".into()), block: true });
        tui.entries.push(Entry::Tool { icon: "❯", title: "cargo test".into(), state: ToolState::Failed, error: None, preview: Some("error[E0425]\n(exit code 101)".into()), block: true });
        tui.entries.push(Entry::Footer { agent: "Agent".into(), detail: "test-model · 2.0s · 11.0 tok/s".into() });
        let lines = tui.transcript(80);
        let flat: Vec<String> = lines.iter().map(|l| l.spans.iter().map(|s| s.text.as_str()).collect()).collect();
        assert!(flat.iter().any(|l| l.starts_with(" › Fix the bug")), "user message marker");
        assert!(flat.iter().any(|l| l.contains("▸ Read src/main.rs")));
        assert!(lines.iter().any(|l| l.bg == Some(DIFF_ADD_BG)) && lines.iter().any(|l| l.bg == Some(DIFF_DEL_BG)));
        assert!(flat.iter().any(|l| l.contains("❯ cargo test")));
        assert!(flat.iter().any(|l| l.contains("◆ Agent")));
        // Code block rendered on a panel; inline code cyan.
        assert!(lines.iter().any(|l| l.bg == Some(PANEL) && flat_contains(l, "fn main() {}")));
        assert!(lines.iter().any(|l| l.spans.iter().any(|s| s.text == "three" && s.fg == SECONDARY)));
        // Widths never exceed the column.
        for l in &flat {
            assert!(l.chars().count() <= 82, "{l:?}");
        }
        tui.render_session(120, 30);
        tui.render_session(60, 20);
        let mut home = SpindleTui::new(info());
        home.canvas.resize(80, 24);
        home.render_home(80, 24);
    }

    /// Prints a rendered session as plain text (`cargo test --release dump_session -- --ignored --nocapture`).
    #[test]
    #[ignore]
    fn dump_session_screen() {
        for (w, h) in [(130u16, 40u16), (80, 34)] {
            let mut tui = SpindleTui::new(info());
            tui.canvas.resize(w, h);
            tui.entries.push(Entry::User("Rename the greeting in demo.rs to \"hello spindle\" and run the tests".into()));
            tui.entries.push(Entry::Assistant { raw: "<think>Need to read the file first.</think>\n\nI'll check the file first.".into(), streaming: false });
            tui.entries.push(Entry::Tool { icon: "▸", title: "Read demo.rs".into(), state: ToolState::Ok, error: None, preview: None, block: false });
            tui.entries.push(Entry::Tool { icon: "⌕", title: "Grep \"hello\" (2 matches)".into(), state: ToolState::Ok, error: None, preview: None, block: false });
            tui.entries.push(Entry::Tool { icon: "✎", title: "Edit demo.rs".into(), state: ToolState::Ok, error: None, preview: Some("-    println!(\"hello\");\n+    println!(\"hello spindle\");".into()), block: true });
            tui.entries.push(Entry::Tool { icon: "❯", title: "cargo test".into(), state: ToolState::Ok, error: None, preview: Some("running 3 tests\ntest a ... ok\ntest result: ok. 3 passed".into()), block: true });
            tui.entries.push(Entry::Assistant { raw: "Done. The greeting now reads **hello spindle** and all tests pass:\n- `demo.rs` line 2 updated\n- `cargo test`: 3 passed".into(), streaming: false });
            tui.entries.push(Entry::Footer { agent: "Agent".into(), detail: "Qwen3-30B-A3B · 14.2s · 11.8 tok/s · spec 41/52".into() });
            tui.input.set_text("/to");
            tui.render_session(w, h);
            println!("===== {w}x{h} =====");
            for y in 0..h {
                let row: String = (0..w).map(|x| tui.canvas.cell(x, y).unwrap().ch).collect();
                println!("{}", row.trim_end());
            }
        }
    }

    fn flat_contains(l: &Line, s: &str) -> bool {
        l.spans.iter().map(|x| x.text.as_str()).collect::<String>().contains(s)
    }

    #[test]
    fn prompt_wraps_and_tracks_cursor() {
        let mut tui = SpindleTui::new(info());
        tui.input.set_text("abcdefghij\nxy");
        let (rows, cur) = tui.prompt_rows(4);
        assert_eq!(rows, vec!["abcd", "efgh", "ij", "xy"]);
        assert_eq!(cur, (3, 2));
        tui.input.cursor = 4;
        assert_eq!(tui.prompt_rows(4).1, (1, 0));
    }

    #[test]
    fn slash_popup_and_completion() {
        assert!(command_matches("/t").iter().any(|(c, _)| *c == "/tools"));
        assert!(command_matches("/tools on").is_empty());
        let mut tui = SpindleTui::new(info());
        tui.input.set_text("/too");
        assert_eq!(tui.handle_key(KeyAction::Enter), None);
        assert_eq!(tui.input.text, "/tools");
        assert_eq!(tui.handle_key(KeyAction::Enter).as_deref(), Some("/tools"));
        // Ctrl+C clears first, quits on an empty prompt; Esc never quits.
        tui.input.set_text("draft");
        tui.handle_key(KeyAction::Escape);
        assert!(tui.running && tui.input.text.is_empty());
        tui.handle_key(KeyAction::CtrlC);
        assert!(!tui.running);
    }

    #[test]
    fn slash_commands_dispatch() {
        let path = "smollm2-135m-q4_k_m.gguf";
        if !std::path::Path::new(path).exists() {
            return;
        }
        let mut engine = SmolLm2Engine::load(path).expect("load");
        let mut tui = SpindleTui::new(info());
        let mut opts = AgentOpts::default();
        assert!(handle_slash_command(&mut tui, &mut engine, "/help", &mut opts).is_none());
        assert!(tui.entries.iter().any(|e| matches!(e, Entry::Notice { text, .. } if text.contains("/tools"))));
        handle_slash_command(&mut tui, &mut engine, "/think on", &mut opts);
        assert!(opts.allow_think);
        handle_slash_command(&mut tui, &mut engine, "/tools on", &mut opts);
        assert!(engine.enable_tools && engine.system_prompt.contains("<tools>"));
        // A custom system prompt keeps its case and survives the tools section swap.
        handle_slash_command(&mut tui, &mut engine, "/system Ikaw ay si Spindle.", &mut opts);
        assert!(engine.system_prompt.starts_with("Ikaw ay si Spindle.") && engine.system_prompt.contains("<tools>"));
        handle_slash_command(&mut tui, &mut engine, "/tools off", &mut opts);
        assert_eq!(engine.system_prompt, "Ikaw ay si Spindle.");
        assert!(handle_slash_command(&mut tui, &mut engine, "/init", &mut opts).is_some());
        handle_slash_command(&mut tui, &mut engine, "/max 300", &mut opts);
        assert_eq!(opts.max_tokens, 300);
    }

    #[test]
    fn engine_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<SmolLm2Engine>();
    }
}
