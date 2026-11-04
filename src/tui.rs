//! OpenCode-grade Full-Screen Terminal User Interface (TUI) for Spindle.
//!
//! Provides an alternate screen canvas, 24-bit TrueColor atomic frame rendering,
//! floating elevated card layout, responsive word-wrapping, non-blocking asynchronous
//! token streaming (user can type and scroll during generation), SGR 1006 mouse wheel scrolling,
//! arrow key scroll navigation, prompt history, and an electric blue theme.

use std::io::{self, Read, Write};
use windows::Win32::System::Console::{
    GetConsoleMode, GetConsoleScreenBufferInfo, GetStdHandle, SetConsoleMode,
    CONSOLE_MODE, CONSOLE_SCREEN_BUFFER_INFO, STD_INPUT_HANDLE, STD_OUTPUT_HANDLE,
    ENABLE_ECHO_INPUT, ENABLE_LINE_INPUT, ENABLE_PROCESSED_INPUT,
    ENABLE_VIRTUAL_TERMINAL_INPUT, ENABLE_VIRTUAL_TERMINAL_PROCESSING,
    GetNumberOfConsoleInputEvents,
};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Rgb(pub u8, pub u8, pub u8);

impl Rgb {
    pub const BG_DARK: Rgb = Rgb(10, 14, 23);           // Deep midnight navy
    pub const CARD_BG: Rgb = Rgb(17, 24, 39);          // Elevated dark sapphire slate
    pub const CARD_BORDER: Rgb = Rgb(30, 58, 110);      // Muted cobalt border
    pub const PRIMARY_BLUE: Rgb = Rgb(37, 99, 235);     // Signature Royal Blue
    pub const BRIGHT_BLUE: Rgb = Rgb(59, 130, 246);     // Electric Blue
    pub const SKY_BLUE: Rgb = Rgb(96, 165, 250);        // Sky Blue highlight
    pub const ICE_CYAN: Rgb = Rgb(56, 189, 248);        // Ice Cyan
    pub const WHITE: Rgb = Rgb(248, 250, 252);         // Crisp White
    pub const MUTED: Rgb = Rgb(148, 163, 184);         // Slate Gray
    pub const DIM: Rgb = Rgb(71, 85, 105);             // Subtle Dark Gray
    pub const GREEN: Rgb = Rgb(52, 211, 153);          // Emerald (mesh/success)
    pub const ORANGE: Rgb = Rgb(251, 146, 60);         // Warm Orange (prune/stats)
    pub const GOLD: Rgb = Rgb(251, 191, 36);           // Tool action gold
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Cell {
    pub ch: char,
    pub fg: Rgb,
    pub bg: Rgb,
    pub bold: bool,
    pub dim: bool,
}

impl Default for Cell {
    fn default() -> Self {
        Self {
            ch: ' ',
            fg: Rgb::WHITE,
            bg: Rgb::BG_DARK,
            bold: false,
            dim: false,
        }
    }
}

/// RAII Terminal protector that restores original terminal modes upon drop.
pub struct TerminalGuard {
    in_handle: windows::Win32::Foundation::HANDLE,
    out_handle: windows::Win32::Foundation::HANDLE,
    orig_in_mode: CONSOLE_MODE,
    orig_out_mode: CONSOLE_MODE,
}

impl TerminalGuard {
    pub fn enter() -> io::Result<Self> {
        unsafe {
            let in_handle = GetStdHandle(STD_INPUT_HANDLE)
                .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
            let out_handle = GetStdHandle(STD_OUTPUT_HANDLE)
                .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;

            let mut orig_in_mode = CONSOLE_MODE(0);
            let mut orig_out_mode = CONSOLE_MODE(0);
            GetConsoleMode(in_handle, &mut orig_in_mode)
                .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
            GetConsoleMode(out_handle, &mut orig_out_mode)
                .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;

            // Enable virtual terminal processing for output
            let new_out = orig_out_mode.0 | ENABLE_VIRTUAL_TERMINAL_PROCESSING.0;
            let _ = SetConsoleMode(out_handle, CONSOLE_MODE(new_out));

            // Set raw mode for input: disable line buffering and echo, enable VT input
            let new_in = (orig_in_mode.0 | ENABLE_VIRTUAL_TERMINAL_INPUT.0)
                & !(ENABLE_LINE_INPUT.0 | ENABLE_ECHO_INPUT.0 | ENABLE_PROCESSED_INPUT.0);
            let _ = SetConsoleMode(in_handle, CONSOLE_MODE(new_in));

            let mut stdout = io::stdout();
            // Enter alternate buffer (\x1b[?1049h), hide cursor (\x1b[?25l), disable auto-wrap (\x1b[?7l),
            // enable SGR extended mouse mode (\x1b[?1006h), enable alternate scroll mode (\x1b[?1007h).
            // NOTE: \x1b[?1000h (normal click tracking) is omitted so mouse clicks/taps never emit raw escape bytes!
            let _ = write!(stdout, "\x1b[?1049h\x1b[?25l\x1b[?7l\x1b[?1006h\x1b[?1007h");
            let _ = stdout.flush();

            Ok(Self {
                in_handle,
                out_handle,
                orig_in_mode,
                orig_out_mode,
            })
        }
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        unsafe {
            let mut stdout = io::stdout();
            // Disable mouse modes, enable auto-wrap (\x1b[?7h), show cursor (\x1b[?25h),
            // exit alternate screen buffer (\x1b[?1049l), reset styles
            let _ = write!(stdout, "\x1b[?1007l\x1b[?1006l\x1b[?1000l\x1b[?7h\x1b[?25h\x1b[?1049l\x1b[0m");
            let _ = stdout.flush();

            let _ = SetConsoleMode(self.out_handle, self.orig_out_mode);
            let _ = SetConsoleMode(self.in_handle, self.orig_in_mode);
        }
    }
}

/// Query current console dimensions (columns, rows).
pub fn terminal_size() -> (u16, u16) {
    unsafe {
        if let Ok(handle) = GetStdHandle(STD_OUTPUT_HANDLE) {
            let mut csbi = CONSOLE_SCREEN_BUFFER_INFO::default();
            if GetConsoleScreenBufferInfo(handle, &mut csbi).is_ok() {
                let width = (csbi.srWindow.Right - csbi.srWindow.Left + 1).max(20) as u16;
                let height = (csbi.srWindow.Bottom - csbi.srWindow.Top + 1).max(10) as u16;
                return (width, height);
            }
        }
    }
    (100, 30)
}

/// Query Windows console subsystem if there are pending input events.
pub fn has_console_input() -> bool {
    unsafe {
        if let Ok(handle) = GetStdHandle(STD_INPUT_HANDLE) {
            let mut count = 0u32;
            if GetNumberOfConsoleInputEvents(handle, &mut count).is_ok() {
                return count > 0;
            }
        }
    }
    false
}

/// High-performance 2D frame buffer with atomic full-screen sync.
pub struct Canvas {
    pub width: u16,
    pub height: u16,
    needs_clear: bool,
    grid: Vec<Cell>,
}

impl Canvas {
    pub fn new(width: u16, height: u16) -> Self {
        let size = (width as usize) * (height as usize);
        Self {
            width,
            height,
            needs_clear: false,
            grid: vec![Cell::default(); size],
        }
    }

    pub fn resize(&mut self, width: u16, height: u16) {
        if self.width == width && self.height == height {
            return;
        }
        self.width = width;
        self.height = height;
        self.needs_clear = true;
        let size = (width as usize) * (height as usize);
        self.grid = vec![Cell::default(); size];
    }

    pub fn clear(&mut self, bg: Rgb) {
        for c in self.grid.iter_mut() {
            c.ch = ' ';
            c.fg = Rgb::WHITE;
            c.bg = bg;
            c.bold = false;
            c.dim = false;
        }
    }

    #[inline]
    pub fn set(&mut self, x: u16, y: u16, ch: char, fg: Rgb, bg: Rgb, bold: bool, dim: bool) {
        if x < self.width && y < self.height {
            let idx = (y as usize) * (self.width as usize) + (x as usize);
            self.grid[idx] = Cell { ch, fg, bg, bold, dim };
        }
    }

    pub fn print(&mut self, mut x: u16, y: u16, text: &str, fg: Rgb, bg: Rgb, bold: bool, dim: bool) {
        if y >= self.height {
            return;
        }
        for ch in text.chars() {
            if x >= self.width {
                break;
            }
            self.set(x, y, ch, fg, bg, bold, dim);
            x += 1;
        }
    }

    pub fn fill_rect(&mut self, x: u16, y: u16, w: u16, h: u16, ch: char, fg: Rgb, bg: Rgb) {
        for row in y..y + h {
            if row >= self.height {
                break;
            }
            for col in x..x + w {
                if col >= self.width {
                    break;
                }
                self.set(col, row, ch, fg, bg, false, false);
            }
        }
    }

    /// Renders an OpenCode-style floating card with an electric blue left accent strip.
    pub fn draw_card(&mut self, x: u16, y: u16, w: u16, h: u16, bg: Rgb, accent_bar: Rgb) {
        // 1. Fill card interior surface
        self.fill_rect(x, y, w, h, ' ', Rgb::WHITE, bg);

        // 2. Left prominent accent bar (`▌`) running full height of the card
        for row in y..y + h {
            self.set(x, row, '▌', accent_bar, bg, true, false);
        }

        // 3. Subtle borders on top, bottom, and right for refined elevation
        let border_fg = Rgb::CARD_BORDER;
        for col in (x + 1)..(x + w.saturating_sub(1)) {
            self.set(col, y, '─', border_fg, bg, false, false);
            self.set(col, y + h.saturating_sub(1), '─', border_fg, bg, false, false);
        }
        for row in (y + 1)..(y + h.saturating_sub(1)) {
            self.set(x + w.saturating_sub(1), row, '│', border_fg, bg, false, false);
        }
        self.set(x + w.saturating_sub(1), y, '╮', border_fg, bg, false, false);
        self.set(x + w.saturating_sub(1), y + h.saturating_sub(1), '╯', border_fg, bg, false, false);
    }

    /// Atomic frame flush: emits the screen with zero flickering, no trailing space overhead,
    /// and zero risk of accidental terminal scrolling.
    pub fn flush(&mut self) -> io::Result<()> {
        let mut buf = String::with_capacity((self.width as usize) * (self.height as usize) * 8);
        if self.needs_clear {
            // Screen dimensions changed: clear entire screen to eliminate stale artifacts outside new viewport
            buf.push_str("\x1b[2J\x1b[H");
            self.needs_clear = false;
        } else {
            // Home cursor to (1, 1)
            buf.push_str("\x1b[H");
        }

        let mut cur_fg = Rgb(0, 0, 0);
        let mut cur_bg = Rgb(0, 0, 0);
        let mut cur_bold = false;
        let mut cur_dim = false;
        let mut in_escape = false;

        for y in 0..self.height {
            // Find rightmost cell on this row that isn't a blank BG_DARK space
            let mut last_col = 0;
            for x in (0..self.width).rev() {
                let idx = (y as usize) * (self.width as usize) + (x as usize);
                let cell = self.grid[idx];
                if cell.ch != ' ' || cell.bg != Rgb::BG_DARK {
                    last_col = x + 1;
                    break;
                }
            }

            for x in 0..last_col {
                let idx = (y as usize) * (self.width as usize) + (x as usize);
                let cell = self.grid[idx];

                if !in_escape || cell.fg != cur_fg || cell.bg != cur_bg || cell.bold != cur_bold || cell.dim != cur_dim {
                    buf.push_str(&format!(
                        "\x1b[0;38;2;{};{};{};48;2;{};{};{}m",
                        cell.fg.0, cell.fg.1, cell.fg.2,
                        cell.bg.0, cell.bg.1, cell.bg.2
                    ));
                    if cell.bold {
                        buf.push_str("\x1b[1m");
                    }
                    if cell.dim {
                        buf.push_str("\x1b[2m");
                    }
                    cur_fg = cell.fg;
                    cur_bg = cell.bg;
                    cur_bold = cell.bold;
                    cur_dim = cell.dim;
                    in_escape = true;
                }
                buf.push(cell.ch);
            }

            // Set base dark background and clear remainder of row to end of line
            buf.push_str(&format!(
                "\x1b[0;48;2;{};{};{}m\x1b[K",
                Rgb::BG_DARK.0, Rgb::BG_DARK.1, Rgb::BG_DARK.2
            ));
            in_escape = false;

            // Advance cursor to next line using CRLF only if not on the bottom row
            if y + 1 < self.height {
                buf.push_str("\r\n");
            }
        }

        let mut out = io::stdout();
        out.write_all(buf.as_bytes())?;
        out.flush()?;
        Ok(())
    }
}

/// Shaded Unicode block logo for Spindle.
pub const SPINDLE_LOGO: [&str; 4] = [
    "  ▄█████  ██████  ██  ███  ██  ██████   ██       ███████",
    "  ▀█▄     ██▄▄██  ██  ████ ██  ██   ██  ██       ██▄▄▄  ",
    "    ▀▀██  ██      ██  ██ ▀███  ██   ██  ██       ██     ",
    "  █████▀  ██      ██  ██   ██  ██████   ███████  ███████",
];

/// Gradient colors for each line of the Spindle logo (Ice White -> Royal Blue).
pub const LOGO_GRADIENT: [Rgb; 4] = [
    Rgb(240, 249, 255), // Ice White
    Rgb(147, 197, 253), // Soft Sky Blue
    Rgb(59, 130, 246),  // Electric Blue
    Rgb(29, 78, 216),   // Deep Cobalt Blue
];

/// Keys and mouse actions captured during the non-blocking event loop.
#[derive(Debug, PartialEq, Eq)]
pub enum KeyAction {
    Char(char),
    Backspace,
    Delete,
    Enter,
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
    CtrlP,
    Escape,
    None,
}

/// Dedicated asynchronous background input reader thread.
/// Completely prevents any blocking calls on the main UI thread!
pub struct InputReader {
    rx: std::sync::mpsc::Receiver<KeyAction>,
}

impl InputReader {
    pub fn new() -> Self {
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let mut stdin = io::stdin();
            let mut chunk = [0u8; 512];
            let mut pending: Vec<u8> = Vec::new();
            loop {
                // Read whatever bytes are available as one chunk
                let n = match stdin.read(&mut chunk) {
                    Ok(0) | Err(_) => break,
                    Ok(n) => n,
                };
                pending.extend_from_slice(&chunk[..n]);

                // If pending buffer ends with solitary ESC (0x1b), wait a brief 25ms window
                // to see if subsequent escape sequence bytes arrive together
                if pending.ends_with(&[27]) {
                    std::thread::sleep(std::time::Duration::from_millis(25));
                    if has_console_input() {
                        if let Ok(more) = stdin.read(&mut chunk) {
                            if more > 0 {
                                pending.extend_from_slice(&chunk[..more]);
                            }
                        }
                    }
                }

                let (actions, consumed) = parse_input_bytes(&pending);
                pending.drain(..consumed);
                for a in actions {
                    if a != KeyAction::None && tx.send(a).is_err() {
                        return;
                    }
                }
            }
        });
        Self { rx }
    }

    #[inline]
    pub fn poll(&self) -> KeyAction {
        self.rx.try_recv().unwrap_or(KeyAction::None)
    }
}

/// Parses a raw input byte buffer into key/mouse actions.
/// Returns `(actions, bytes_consumed)`. Incomplete trailing escape sequences are left
/// unconsumed so the next read can complete them. Unknown sequences and mouse tracking clicks
/// are swallowed whole and never leaked into the input box as text.
pub fn parse_input_bytes(buf: &[u8]) -> (Vec<KeyAction>, usize) {
    let mut out = Vec::new();
    let mut i = 0;
    while i < buf.len() {
        let b = buf[i];
        match b {
            3 => {
                out.push(KeyAction::Escape);
                i += 1;
            }
            13 | 10 => {
                out.push(KeyAction::Enter);
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
            16 => {
                out.push(KeyAction::CtrlP);
                i += 1;
            }
            27 => {
                // Lone ESC at end of chunk: treated as Escape key
                if i + 1 >= buf.len() {
                    out.push(KeyAction::Escape);
                    i += 1;
                    continue;
                }
                let next = buf[i + 1];
                if next != b'[' && next != b'O' {
                    // ESC followed by something else (Alt+key / double ESC): treat as Escape
                    out.push(KeyAction::Escape);
                    i += 1;
                    continue;
                }
                // CSI / SS3: params are bytes 0x20..=0x3F, final byte 0x40..=0x7E
                let mut j = i + 2;
                while j < buf.len() && (0x20..=0x3F).contains(&buf[j]) {
                    j += 1;
                }
                if j >= buf.len() {
                    // Incomplete sequence: leave in pending buffer to wait for next chunk
                    break;
                }
                let fin = buf[j];
                let params = std::str::from_utf8(&buf[i + 2..j]).unwrap_or("");
                let action = if let Some(sgr) = params.strip_prefix('<') {
                    // SGR extended mouse mode: \x1b[<btn;col;rowM or m
                    let btn = sgr.split(';').next().and_then(|s| s.parse::<u32>().ok()).unwrap_or(0);
                    match btn & !(4 | 8 | 16) {
                        64 => KeyAction::ScrollUp(3),
                        65 => KeyAction::ScrollDown(3),
                        _ => KeyAction::None, // clicks, releases, drags silently swallowed
                    }
                } else if fin == b'M' && params.is_empty() {
                    // Legacy X10 mouse: 3 raw bytes follow
                    if j + 3 >= buf.len() {
                        break; // wait for the 3 payload bytes
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
                        b'A' => KeyAction::Up,
                        b'B' => KeyAction::Down,
                        b'C' => KeyAction::Right,
                        b'D' => KeyAction::Left,
                        b'H' => KeyAction::Home,
                        b'F' => KeyAction::End,
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
                // Safeguard against orphaned CSI sequences missing leading ESC
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
                // Safeguard against orphaned SGR coordinate fragments like ";77;27M"
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
                // UTF-8 decode one character
                let len = match b {
                    0x00..=0x7F => 1,
                    0xC0..=0xDF => 2,
                    0xE0..=0xEF => 3,
                    0xF0..=0xF7 => 4,
                    _ => 1,
                };
                if i + len > buf.len() {
                    break;
                }
                if let Ok(s) = std::str::from_utf8(&buf[i..i + len]) {
                    if let Some(c) = s.chars().next() {
                        out.push(KeyAction::Char(c));
                    }
                }
                i += len;
            }
            _ => {
                i += 1;
            }
        }
    }
    (out, i)
}

/// Text input line editor state.
#[derive(Default)]
pub struct InputBuffer {
    pub text: String,
    pub cursor: usize,
}

impl InputBuffer {
    pub fn insert(&mut self, ch: char) {
        let byte_pos = self.text.char_indices().nth(self.cursor).map(|(i, _)| i).unwrap_or(self.text.len());
        self.text.insert(byte_pos, ch);
        self.cursor += 1;
    }

    pub fn backspace(&mut self) {
        if self.cursor > 0 {
            self.cursor -= 1;
            let byte_pos = self.text.char_indices().nth(self.cursor).map(|(i, _)| i).unwrap_or(self.text.len());
            self.text.remove(byte_pos);
        }
    }

    pub fn delete(&mut self) {
        let char_count = self.text.chars().count();
        if self.cursor < char_count {
            let byte_pos = self.text.char_indices().nth(self.cursor).map(|(i, _)| i).unwrap_or(self.text.len());
            self.text.remove(byte_pos);
        }
    }

    pub fn move_left(&mut self) {
        if self.cursor > 0 {
            self.cursor -= 1;
        }
    }

    pub fn move_right(&mut self) {
        if self.cursor < self.text.chars().count() {
            self.cursor += 1;
        }
    }

    pub fn move_home(&mut self) {
        self.cursor = 0;
    }

    pub fn move_end(&mut self) {
        self.cursor = self.text.chars().count();
    }

    pub fn clear(&mut self) {
        self.text.clear();
        self.cursor = 0;
    }

    pub fn set_text(&mut self, s: &str) {
        self.text = s.to_string();
        self.cursor = self.text.chars().count();
    }
}

/// Message turn stored for rendering conversation history.
pub struct ChatMessage {
    pub role: String,
    pub text: String,
    pub speed_badge: Option<String>,
}

/// Formatted line to be rendered inside the chat viewport.
struct ViewportLine {
    text: String,
    fg: Rgb,
    bold: bool,
    dim: bool,
}

/// Responsive word-wrapping algorithm that cleanly partitions paragraphs by whitespace
/// and breaks excessively long tokens without panicking or exceeding `max_width`.
pub fn wrap_text(text: &str, max_width: usize) -> Vec<String> {
    let mut out = Vec::new();
    let max_w = max_width.max(15);

    for raw_line in text.lines() {
        let trimmed = raw_line.trim_end();
        if trimmed.is_empty() {
            out.push(String::new());
            continue;
        }

        let words: Vec<&str> = trimmed.split_whitespace().collect();
        if words.is_empty() {
            out.push(String::new());
            continue;
        }

        let mut cur = String::new();
        for word in words {
            let w_len = word.chars().count();
            if cur.is_empty() {
                if w_len > max_w {
                    let mut char_iter = word.chars();
                    loop {
                        let chunk: String = char_iter.by_ref().take(max_w).collect();
                        if chunk.is_empty() {
                            break;
                        }
                        if char_iter.clone().next().is_none() {
                            cur = chunk;
                        } else {
                            out.push(chunk);
                        }
                    }
                } else {
                    cur.push_str(word);
                }
            } else if cur.chars().count() + 1 + w_len <= max_w {
                cur.push(' ');
                cur.push_str(word);
            } else {
                out.push(cur);
                cur = String::new();
                if w_len > max_w {
                    let mut char_iter = word.chars();
                    loop {
                        let chunk: String = char_iter.by_ref().take(max_w).collect();
                        if chunk.is_empty() {
                            break;
                        }
                        if char_iter.clone().next().is_none() {
                            cur = chunk;
                        } else {
                            out.push(chunk);
                        }
                    }
                } else {
                    cur.push_str(word);
                }
            }
        }
        if !cur.is_empty() {
            out.push(cur);
        }
    }
    out
}

/// The Full-Screen TUI Orchestrator.
pub struct SpindleTui {
    canvas: Canvas,
    input: InputBuffer,
    history: Vec<ChatMessage>,
    prompt_history: Vec<String>,
    history_idx: Option<usize>,
    pub scroll_offset: usize,
    pub running: bool,
    pub model_name: String,
    pub cluster_peer: Option<String>,
    pub wire_format: String,
    pub prune_val: f32,
    pub layers_info: String,
    pub git_branch: String,
    pub draft_name: Option<String>,
    input_reader: InputReader,
}

impl SpindleTui {
    pub fn new(
        model_name: &str,
        cluster_peer: Option<&str>,
        wire_format: &str,
        prune_val: f32,
        layers_info: &str,
    ) -> Self {
        let (w, h) = terminal_size();
        let branch = Self::detect_git_branch().unwrap_or_else(|| "main".to_string());
        Self {
            canvas: Canvas::new(w, h),
            input: InputBuffer::default(),
            history: Vec::new(),
            prompt_history: Vec::new(),
            history_idx: None,
            scroll_offset: 0,
            running: true,
            model_name: model_name.to_string(),
            cluster_peer: cluster_peer.map(str::to_string),
            wire_format: wire_format.to_string(),
            prune_val,
            layers_info: layers_info.to_string(),
            git_branch: branch,
            draft_name: None,
            input_reader: InputReader::new(),
        }
    }

    #[inline]
    pub fn poll_key(&self) -> KeyAction {
        self.input_reader.poll()
    }

    fn detect_git_branch() -> Option<String> {
        let output = std::process::Command::new("git")
            .args(["rev-parse", "--abbrev-ref", "HEAD"])
            .output()
            .ok()?;
        if output.status.success() {
            let b = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !b.is_empty() {
                return Some(b);
            }
        }
        None
    }

    /// Renders the signature OpenCode-style hero screen in electric blue when starting.
    pub fn render_welcome(&mut self) {
        let (w, h) = terminal_size();
        self.canvas.resize(w, h);
        self.canvas.clear(Rgb::BG_DARK);

        let center_x = w / 2;
        let card_w = (w.saturating_sub(6)).clamp(40, 84);
        let card_h = 5u16;
        let card_x = center_x.saturating_sub(card_w / 2);

        if h >= 22 {
            // Full height terminal: display stylish block logo and centered hero card
            let logo_y = ((h.saturating_sub(18)) / 2).max(2);
            for (i, line) in SPINDLE_LOGO.iter().enumerate() {
                let row = logo_y + (i as u16);
                let len = line.chars().count() as u16;
                let start_x = center_x.saturating_sub(len / 2);
                let color = LOGO_GRADIENT[i % LOGO_GRADIENT.len()];
                self.canvas.print(start_x, row, line, color, Rgb::BG_DARK, true, false);
            }

            // Subtitle
            let sub_prefix = "/help";
            let sub_mid = " for slash commands · ";
            let sub_cmd2 = "/heal";
            let sub_tail = " autonomous loop";
            let total_sub_len = (sub_prefix.len() + sub_mid.len() + sub_cmd2.len() + sub_tail.len()) as u16;
            let mut sub_x = center_x.saturating_sub(total_sub_len / 2);
            let sub_y = logo_y + 4;

            self.canvas.print(sub_x, sub_y, sub_prefix, Rgb::ORANGE, Rgb::BG_DARK, true, false);
            sub_x += sub_prefix.len() as u16;
            self.canvas.print(sub_x, sub_y, sub_mid, Rgb::MUTED, Rgb::BG_DARK, false, false);
            sub_x += sub_mid.len() as u16;
            self.canvas.print(sub_x, sub_y, sub_cmd2, Rgb::ORANGE, Rgb::BG_DARK, true, false);
            sub_x += sub_cmd2.len() as u16;
            self.canvas.print(sub_x, sub_y, sub_tail, Rgb::MUTED, Rgb::BG_DARK, false, false);

            let card_y = logo_y + 6;
            self.render_card_content(card_x, card_y, card_w, card_h, true);

            let hotkeys = "tab tools    ctrl+p commands    /help slash commands    esc quit";
            let hk_x = center_x.saturating_sub((hotkeys.len() as u16) / 2);
            self.canvas.print(hk_x, card_y + card_h + 1, hotkeys, Rgb::MUTED, Rgb::BG_DARK, false, false);
        } else {
            // Compact terminal: sleek single-line header with no overlapping rows
            let banner = "SPINDLE · Autonomous Distributed MoE Engine";
            let bx = center_x.saturating_sub(banner.len() as u16 / 2);
            self.canvas.print(bx, 1, banner, Rgb::BRIGHT_BLUE, Rgb::BG_DARK, true, false);

            let sub_prefix = "/help for slash commands · /heal loop";
            let sx = center_x.saturating_sub(sub_prefix.len() as u16 / 2);
            self.canvas.print(sx, 2, sub_prefix, Rgb::MUTED, Rgb::BG_DARK, false, false);

            let card_y = 4u16;
            self.render_card_content(card_x, card_y, card_w, card_h, true);

            if h >= 14 {
                let hotkeys = "tab tools   ctrl+p commands   esc quit";
                let hk_x = center_x.saturating_sub((hotkeys.len() as u16) / 2);
                self.canvas.print(hk_x, card_y + card_h + 1, hotkeys, Rgb::MUTED, Rgb::BG_DARK, false, false);
            }
        }

        self.render_docked_footer(w, h);
        let _ = self.canvas.flush();
    }

    /// Renders an input card with prompt editor, ghost text, and meta pills.
    fn render_card_content(&mut self, card_x: u16, card_y: u16, card_w: u16, card_h: u16, is_welcome: bool) {
        self.canvas.draw_card(card_x, card_y, card_w, card_h, Rgb::CARD_BG, Rgb::BRIGHT_BLUE);

        let prompt_x = card_x + 3;
        let prompt_y = card_y + 1;
        let visible_w = (card_w.saturating_sub(6)) as usize;

        if self.input.text.is_empty() {
            let ghost = if is_welcome {
                "Ask anything... \"delete hello.py\", \"/review\", \"/stats\""
            } else {
                "Ask anything... \"delete hello.py\", \"/review\""
            };
            let max_ghost_w = visible_w.saturating_sub(2);
            let clipped_ghost: String = ghost.chars().take(max_ghost_w).collect();

            self.canvas.set(prompt_x, prompt_y, '█', Rgb::BRIGHT_BLUE, Rgb::CARD_BG, true, false);
            self.canvas.print(prompt_x + 2, prompt_y, &clipped_ghost, Rgb::DIM, Rgb::CARD_BG, false, false);
        } else {
            let total_chars = self.input.text.chars().count();
            let cursor = self.input.cursor.min(total_chars);
            let window_start = if cursor >= visible_w { cursor - visible_w + 1 } else { 0 };

            let visible_text: String = self.input.text.chars().skip(window_start).take(visible_w).collect();
            self.canvas.print(prompt_x, prompt_y, &visible_text, Rgb::WHITE, Rgb::CARD_BG, false, false);

            let cursor_x = prompt_x + (cursor - window_start) as u16;
            let char_under = self.input.text.chars().nth(cursor).unwrap_or(' ');
            self.canvas.set(cursor_x, prompt_y, char_under, Rgb::CARD_BG, Rgb::BRIGHT_BLUE, true, false);
        }

        let meta_y = card_y + 3;
        let mut mx = prompt_x;
        let max_card_content_x = card_x + card_w - 2;

        if mx + 6 < max_card_content_x {
            self.canvas.print(mx, meta_y, "Build", Rgb::ICE_CYAN, Rgb::CARD_BG, true, false);
            mx += 6;
        }

        let model_badge = if is_welcome {
            format!("· {} ({})", self.model_name, self.layers_info)
        } else {
            format!("· {}", self.model_name)
        };
        if mx + (model_badge.len() as u16) < max_card_content_x {
            self.canvas.print(mx, meta_y, &model_badge, Rgb::WHITE, Rgb::CARD_BG, false, false);
            mx += model_badge.len() as u16 + 1;
        }

        let prune_badge = format!("· {:.2} prune", self.prune_val);
        if mx + (prune_badge.len() as u16) < max_card_content_x {
            self.canvas.print(mx, meta_y, &prune_badge, Rgb::ORANGE, Rgb::CARD_BG, false, false);
            mx += prune_badge.len() as u16 + 1;
        }

        if is_welcome {
            if let Some(peer) = &self.cluster_peer {
                let mesh_badge = format!("· mesh: {peer}");
                if mx + (mesh_badge.len() as u16) < max_card_content_x {
                    self.canvas.print(mx, meta_y, &mesh_badge, Rgb::GREEN, Rgb::CARD_BG, false, false);
                }
            }
        } else {
            let low_mem = "· low-mem";
            if mx + (low_mem.len() as u16) < max_card_content_x {
                self.canvas.print(mx, meta_y, low_mem, Rgb::MUTED, Rgb::CARD_BG, false, false);
                mx += low_mem.len() as u16 + 1;
            }
        }

        if let Some(draft) = &self.draft_name {
            let draft_badge = format!("· spec: {draft}");
            if mx + (draft_badge.len() as u16) < max_card_content_x {
                self.canvas.print(mx, meta_y, &draft_badge, Rgb::ICE_CYAN, Rgb::CARD_BG, false, false);
            }
        }
    }

    /// Renders a pre-warming notification while system prompt and KV-cache are initialized.
    pub fn render_warming(&mut self, text: &str) {
        let (w, h) = terminal_size();
        self.canvas.resize(w, h);
        self.canvas.clear(Rgb::BG_DARK);

        let center_x = w / 2;
        let card_w = 68.min(w.saturating_sub(6));
        let card_h = 5u16;
        let card_x = center_x.saturating_sub(card_w / 2);
        let card_y = (h / 2).saturating_sub(3);

        self.canvas.draw_card(card_x, card_y, card_w, card_h, Rgb::CARD_BG, Rgb::ICE_CYAN);
        let title = "INITIALIZING KV-CACHE & PRE-WARMING SYSTEM PROMPT";
        self.canvas.print(card_x + 3, card_y + 1, title, Rgb::ICE_CYAN, Rgb::CARD_BG, true, false);
        self.canvas.print(card_x + 3, card_y + 3, text, Rgb::MUTED, Rgb::CARD_BG, false, false);

        self.render_docked_footer(w, h);
        let _ = self.canvas.flush();
    }

    /// Renders the conversation view strictly CHRONOLOGICALLY with word wrapping and scroll support.
    pub fn render_chat(&mut self) {
        let (w, h) = terminal_size();
        self.canvas.resize(w, h);
        self.canvas.clear(Rgb::BG_DARK);

        // 1. Persistent Top Header Bar (Row 0)
        let header_bg = Rgb::CARD_BG;
        self.canvas.fill_rect(0, 0, w, 1, ' ', Rgb::WHITE, header_bg);
        self.canvas.print(2, 0, "SPINDLE v0.1.0", Rgb::BRIGHT_BLUE, header_bg, true, false);

        let status = if let Some(d) = &self.draft_name {
            format!("· {} · {} · spec: {d}", self.model_name, self.layers_info)
        } else {
            format!("· {} · {}", self.model_name, self.layers_info)
        };
        if let Some(p) = &self.cluster_peer {
            let mesh_str = format!("mesh: {p}");
            let mx = w.saturating_sub(mesh_str.len() as u16 + 4);
            if 18 + (status.len() as u16) < mx {
                self.canvas.print(18, 0, &status, Rgb::MUTED, header_bg, false, false);
            } else {
                let max_stat = (mx.saturating_sub(20)) as usize;
                let clip: String = status.chars().take(max_stat).collect();
                self.canvas.print(18, 0, &clip, Rgb::MUTED, header_bg, false, false);
            }
            self.canvas.print(mx, 0, &mesh_str, Rgb::GREEN, header_bg, true, false);
        } else {
            self.canvas.print(18, 0, &status, Rgb::MUTED, header_bg, false, false);
        }

        // Horizontal divider below header
        for col in 0..w {
            self.canvas.set(col, 1, '─', Rgb::CARD_BORDER, Rgb::BG_DARK, false, false);
        }

        // 2. Format conversation history chronologically with RESPONSIVE WORD WRAPPING
        let mut lines: Vec<ViewportLine> = Vec::new();

        for msg in &self.history {
            if msg.role == "user" {
                let user_max_w = (w.saturating_sub(14)).max(20) as usize;
                let wrapped_user = wrap_text(&msg.text, user_max_w);
                for (idx, u_line) in wrapped_user.into_iter().enumerate() {
                    let prefix = if idx == 0 { "> You: " } else { "       " };
                    lines.push(ViewportLine {
                        text: format!("{prefix}{u_line}"),
                        fg: Rgb::SKY_BLUE,
                        bold: true,
                        dim: false,
                    });
                }
            } else if msg.role == "tool" {
                let tool_max_w = (w.saturating_sub(14)).max(20) as usize;
                let wrapped_tool = wrap_text(&msg.text, tool_max_w);
                for (idx, t_line) in wrapped_tool.into_iter().enumerate() {
                    let prefix = if idx == 0 { "  [*] " } else { "      " };
                    lines.push(ViewportLine {
                        text: format!("{prefix}{t_line}"),
                        fg: Rgb::GOLD,
                        bold: false,
                        dim: false,
                    });
                }
            } else if msg.role == "system" {
                let sys_max_w = (w.saturating_sub(10)).max(20) as usize;
                for raw_line in msg.text.lines() {
                    let wrapped = wrap_text(raw_line, sys_max_w);
                    for s_line in wrapped {
                        lines.push(ViewportLine {
                            text: format!("  {s_line}"),
                            fg: Rgb::ICE_CYAN,
                            bold: false,
                            dim: false,
                        });
                    }
                }
            } else {
                // Assistant message
                let header = if let Some(speed) = &msg.speed_badge {
                    format!("Spindle · {speed}")
                } else {
                    "Spindle".to_string()
                };
                lines.push(ViewportLine {
                    text: header,
                    fg: Rgb::PRIMARY_BLUE,
                    bold: true,
                    dim: false,
                });
                let assistant_max_w = (w.saturating_sub(12)).max(20) as usize;
                let wrapped_body = wrap_text(&msg.text, assistant_max_w);
                for b_line in wrapped_body {
                    lines.push(ViewportLine {
                        text: format!("  {b_line}"),
                        fg: Rgb::WHITE,
                        bold: false,
                        dim: false,
                    });
                }
            }
            // Empty blank line between turns for clean visual rhythm
            lines.push(ViewportLine {
                text: String::new(),
                fg: Rgb::WHITE,
                bold: false,
                dim: false,
            });
        }

        // 3. Render visible viewport lines above the bottom input card with scroll offset
        let top_y = 2u16;
        let card_h = 5u16;
        let card_y = if h >= 12 { h.saturating_sub(7) } else { (h.saturating_sub(card_h + 1)).max(top_y) };
        let viewport_h = card_y.saturating_sub(top_y) as usize;

        let total_lines = lines.len();
        let max_scroll = total_lines.saturating_sub(viewport_h);
        self.scroll_offset = self.scroll_offset.min(max_scroll);

        let end_idx = total_lines.saturating_sub(self.scroll_offset);
        let start_idx = end_idx.saturating_sub(viewport_h);

        for (i, line) in lines[start_idx..end_idx].iter().enumerate() {
            let y = top_y + (i as u16);
            if y >= card_y {
                break;
            }
            self.canvas.print(3, y, &line.text, line.fg, Rgb::BG_DARK, line.bold, line.dim);
        }

        // Render scroll indicator if scrolled up into history
        if self.scroll_offset > 0 {
            let scroll_indicator = format!("▲ Scrolled +{} lines · Mouse wheel or PgDn to return to bottom", self.scroll_offset);
            let sx = w.saturating_sub(scroll_indicator.len() as u16 + 4);
            self.canvas.print(sx, 1, &scroll_indicator, Rgb::ICE_CYAN, Rgb::BG_DARK, true, false);
        }

        // 4. Floating Input Card docked at the bottom
        let card_w = (w.saturating_sub(6)).clamp(40, 84);
        let card_x = (w / 2).saturating_sub(card_w / 2);

        self.render_card_content(card_x, card_y, card_w, card_h, false);

        // 5. Docked Global Footer
        self.render_docked_footer(w, h);

        let _ = self.canvas.flush();
    }

    fn render_docked_footer(&mut self, w: u16, h: u16) {
        let footer_y = h.saturating_sub(1);
        let raw_cwd = std::env::current_dir()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|_| ".".to_string());
        let short_cwd = if let Ok(home) = std::env::var("USERPROFILE") {
            if raw_cwd.starts_with(&home) {
                format!("~{}", &raw_cwd[home.len()..])
            } else {
                raw_cwd
            }
        } else {
            raw_cwd
        };
        let max_left_w = (w.saturating_sub(20)) as usize;
        let footer_left = format!("{}:{}", short_cwd, self.git_branch);
        let clipped_left: String = footer_left.chars().take(max_left_w).collect();
        self.canvas.print(2, footer_y, &clipped_left, Rgb::MUTED, Rgb::BG_DARK, false, false);

        let footer_right = "spindle v0.1.0";
        let right_x = w.saturating_sub(footer_right.len() as u16 + 2);
        self.canvas.print(right_x, footer_y, footer_right, Rgb::MUTED, Rgb::BG_DARK, false, false);
    }

    /// Appends a completed message turn to conversation history.
    pub fn push_message(&mut self, role: &str, text: &str, speed_badge: Option<&str>) {
        self.history.push(ChatMessage {
            role: role.to_string(),
            text: text.to_string(),
            speed_badge: speed_badge.map(str::to_string),
        });
    }

    /// Runs an idle prompt turn through the TUI loop, rendering frames and collecting keyboard input.
    pub fn poll_input(&mut self) -> Option<String> {
        loop {
            if self.history.is_empty() {
                self.render_welcome();
            } else {
                self.render_chat();
            }

            let mut action = self.poll_key();
            while action == KeyAction::None {
                std::thread::sleep(std::time::Duration::from_millis(16)); // ~60 FPS idle
                action = self.poll_key();
            }

            match action {
                KeyAction::Char(c) => self.input.insert(c),
                KeyAction::Backspace => self.input.backspace(),
                KeyAction::Delete => self.input.delete(),
                KeyAction::Left => self.input.move_left(),
                KeyAction::Right => self.input.move_right(),
                KeyAction::Home => self.input.move_home(),
                KeyAction::End => self.input.move_end(),
                KeyAction::ScrollUp(n) => {
                    self.scroll_offset = self.scroll_offset.saturating_add(n);
                }
                KeyAction::ScrollDown(n) => {
                    self.scroll_offset = self.scroll_offset.saturating_sub(n);
                }
                KeyAction::PageUp => {
                    self.scroll_offset = self.scroll_offset.saturating_add(6);
                }
                KeyAction::PageDown => {
                    self.scroll_offset = self.scroll_offset.saturating_sub(6);
                }
                KeyAction::Up => {
                    if self.scroll_offset > 0 {
                        self.scroll_offset = self.scroll_offset.saturating_add(2);
                    } else if !self.history.is_empty() && self.input.text.is_empty() {
                        // Empty input in chat view: Up arrow scrolls conversation history!
                        self.scroll_offset = self.scroll_offset.saturating_add(2);
                    } else if self.input.text.is_empty() && !self.prompt_history.is_empty() && self.history_idx.is_none() {
                        let idx = self.prompt_history.len() - 1;
                        self.history_idx = Some(idx);
                        self.input.set_text(&self.prompt_history[idx]);
                    } else if let Some(idx) = self.history_idx {
                        if idx > 0 {
                            let new_idx = idx - 1;
                            self.history_idx = Some(new_idx);
                            self.input.set_text(&self.prompt_history[new_idx]);
                        } else {
                            self.history_idx = None;
                            self.input.clear();
                        }
                    }
                }
                KeyAction::Down => {
                    if self.scroll_offset > 0 {
                        self.scroll_offset = self.scroll_offset.saturating_sub(2);
                    } else if let Some(idx) = self.history_idx {
                        if idx + 1 < self.prompt_history.len() {
                            let new_idx = idx + 1;
                            self.history_idx = Some(new_idx);
                            self.input.set_text(&self.prompt_history[new_idx]);
                        } else {
                            self.history_idx = None;
                            self.input.clear();
                        }
                    }
                }
                KeyAction::CtrlP => {
                    if !self.prompt_history.is_empty() {
                        let idx = match self.history_idx {
                            None => self.prompt_history.len() - 1,
                            Some(0) => 0,
                            Some(i) => i - 1,
                        };
                        self.history_idx = Some(idx);
                        self.input.set_text(&self.prompt_history[idx]);
                    }
                }
                KeyAction::Enter => {
                    let text = self.input.text.trim().to_string();
                    if !text.is_empty() {
                        self.prompt_history.push(text.clone());
                        self.history_idx = None;
                        self.scroll_offset = 0;
                        self.input.clear();
                        return Some(text);
                    }
                }
                KeyAction::Escape => {
                    self.running = false;
                    return None;
                }
                _ => {}
            }
        }
    }
}

/// Message events transmitted from background generation worker to UI thread.
enum WorkerEvent {
    Token(String),
    ToolStart(String),
    Done {
        reply: String,
        speed_badge: String,
    },
    Stopped,
}

/// Launches the full-screen OpenCode TUI interactive session.
pub fn run_tui(
    engine: &mut crate::smollm_chat::SmolLm2Engine,
    governor: Option<&crate::memory_guard::Governor>,
    opts: crate::chat_cli::ChatOptions,
) {
    let guard = match TerminalGuard::enter() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("[FAIL] Could not initialize full-screen TUI terminal: {e}");
            return;
        }
    };

    let peer_str = engine.cluster_client.as_ref().and_then(|c| c.stream.peer_addr().ok().map(|a| a.to_string()));
    let layers_str = format!("layers {}..{}", engine.layer_range.start, engine.layer_range.end);
    let mut tui = SpindleTui::new(
        &engine.idx.model_name,
        peer_str.as_deref(),
        "fp8",
        0.03,
        &layers_str,
    );
    if let Some(draft) = &engine.draft_engine {
        tui.draft_name = Some(draft.idx.model_name.clone());
    }

    let mut allow_think = opts.allow_think;
    let mut max_tokens = if opts.max_tokens > 0 { opts.max_tokens.min(engine.max_seq) } else { 2048.min(engine.max_seq) };

    if engine.n_keep == 0 {
        tui.render_warming("Prefilling system prompt and tools across cluster mesh...");
        unsafe {
            engine.prefill_system();
        }
    }

    // If an initial prompt was provided, execute it immediately
    if let Some(first_p) = opts.first_prompt {
        execute_tui_turn(&mut tui, engine, governor, &first_p, allow_think, max_tokens);
    }

    // Main TUI interactive event loop
    while tui.running {
        if let Some(user_prompt) = tui.poll_input() {
            if user_prompt == "/exit" || user_prompt == "/quit" || user_prompt == "exit" || user_prompt == "quit" {
                break;
            }
            if handle_slash_command(&mut tui, engine, &user_prompt, &mut allow_think, &mut max_tokens) {
                continue;
            }
            execute_tui_turn(&mut tui, engine, governor, &user_prompt, allow_think, max_tokens);
        }
    }

    drop(guard);
}

/// Dispatches interactive slash commands within the TUI without sending them as LLM prompts.
fn handle_slash_command(
    tui: &mut SpindleTui,
    engine: &mut crate::smollm_chat::SmolLm2Engine,
    cmd_line: &str,
    allow_think: &mut bool,
    max_tokens: &mut usize,
) -> bool {
    let trimmed = cmd_line.trim();
    if !trimmed.starts_with('/') {
        return false;
    }

    tui.push_message("user", trimmed, None);
    tui.scroll_offset = 0;

    let lower = trimmed.to_ascii_lowercase();
    match lower.as_str() {
        "/help" | "/?" => {
            let help_text = "\
╭─ Available Slash Commands ─────────────────────────────────────────────╮
│  /help                       show this command reference               │
│  /clear                      reset KV-cache & start fresh conversation │
│  /cluster                    view distributed mesh status & ping RTT   │
│  /model                      inspect model tensors, layers & RAM       │
│  /tools [on|off]             toggle agentic tools (files, search, cmd) │
│  /moe-prune [threshold]      inspect or dynamically tune expert cutoff │
│  /think [on|off]             toggle reasoning mode                     │
│  /system [prompt|reset]      inspect or dynamically set system prompt  │
│  /review                     semantic AI review of uncommitted git diff│
│  /diff                       view uncommitted git diff of current edits│
│  /undo                       revert the last file change made by agent │
│  /bench [n]                  run on-demand speed benchmark (default 32)│
│  /stats                      view memory residency & KV cache stats    │
│  /exit                       quit interactive session                  │
╰────────────────────────────────────────────────────────────────────────╯";
            tui.push_message("system", help_text, None);
        }
        "/clear" | "/reset" => {
            engine.reset_kv();
            tui.history.clear();
            tui.scroll_offset = 0;
            unsafe {
                engine.prefill_system();
            }
            return true;
        }
        "/model" | "/info" => {
            let mib = engine.raw_bytes.len() >> 20;
            let c = engine.cfg;
            let (ws, private, _) = crate::memory_guard::process_counters();
            let free_ram = crate::mmap_file::available_ram_bytes() >> 20;
            let info = format!(
                "Model: {}\nArchitecture: {} (dim: {}, layers: {}/{}, heads: {}/{})\nMemory: {} MiB file · {} MiB resident · {} MiB private · {} MiB free RAM\nKV-Cache: {}/{} tokens ({:.1} MiB allocated)",
                engine.idx.model_name,
                engine.idx.architecture, c.dim,
                engine.layer_range.len(), c.n_layers,
                c.n_heads, c.n_kv_heads,
                mib, ws >> 20, private >> 20, free_ram,
                engine.cur_pos, engine.max_seq,
                ((engine.k_cache.len() + engine.v_cache.len()) * 2) as f64 / (1024.0 * 1024.0)
            );
            tui.push_message("system", &info, None);
        }
        "/cluster" | "/nodes" => {
            if let Some(client) = engine.cluster_client.as_mut() {
                let rtt = client.ping().unwrap_or(std::time::Duration::from_millis(1));
                let info = format!(
                    "Cluster Mesh: ONLINE\nPartition: Worker (layers {}..{}) + Local (layers {}..{}) = {} total layers\nTransport: Wire {:?} · Network Ping RTT: {:.2} ms ({:.1} µs)",
                    client.worker_range.0, client.worker_range.1,
                    engine.layer_range.start, engine.layer_range.end,
                    engine.cfg.n_layers,
                    client.wire_format,
                    rtt.as_secs_f64() * 1e3,
                    rtt.as_secs_f64() * 1e6
                );
                tui.push_message("system", &info, None);
            } else {
                tui.push_message("system", "Cluster Mesh: standalone (no peer connected)", None);
            }
        }
        "/tools" | "/tools toggle" => {
            let now = !engine.enable_tools;
            engine.set_tools_enabled(now);
            let msg = if now { "Agentic tools: ON (files, web search, shell enabled)" } else { "Agentic tools: OFF (pure chat mode)" };
            tui.push_message("system", msg, None);
        }
        "/tools on" => {
            engine.set_tools_enabled(true);
            tui.push_message("system", "Agentic tools: ON (files, web search, shell enabled)", None);
        }
        "/tools off" => {
            engine.set_tools_enabled(false);
            tui.push_message("system", "Agentic tools: OFF (pure chat mode)", None);
        }
        "/think" | "/think toggle" => {
            *allow_think = !*allow_think;
            let msg = if *allow_think { "Reasoning mode (<think>): ON" } else { "Reasoning mode (<think>): OFF (fast direct reply)" };
            tui.push_message("system", msg, None);
        }
        "/think on" => {
            *allow_think = true;
            tui.push_message("system", "Reasoning mode (<think>): ON", None);
        }
        "/think off" => {
            *allow_think = false;
            tui.push_message("system", "Reasoning mode (<think>): OFF (fast direct reply)", None);
        }
        "/moe-prune" => {
            let p = engine.moe_prune();
            let status = if p > 0.0 { format!("active (threshold {p:.3})") } else { "disabled (all 8 experts active)".to_string() };
            let msg = format!("MoE Pruning Cutoff: {status}\nUse '/moe-prune <threshold>' (e.g. 0.05) to tune");
            tui.push_message("system", &msg, None);
        }
        cmd if cmd.starts_with("/moe-prune ") => {
            if let Ok(p) = cmd[11..].trim().parse::<f32>() {
                engine.set_moe_prune(p);
                tui.prune_val = p;
                let msg = if p > 0.0 {
                    format!("Dynamic MoE pruning set to {:.3} ({:.1}% energy cutoff)", p, p * 100.0)
                } else {
                    "Dynamic MoE pruning disabled (all 8 experts active)".to_string()
                };
                tui.push_message("system", &msg, None);
            } else {
                tui.push_message("system", "Usage: /moe-prune <threshold, e.g. 0.05>", None);
            }
        }
        "/review" => {
            let output = std::process::Command::new("git").args(["diff", "HEAD"]).output();
            match output {
                Ok(out) => {
                    let diff_str = String::from_utf8_lossy(&out.stdout);
                    let trimmed = diff_str.trim();
                    if trimmed.is_empty() {
                        tui.push_message("system", "No uncommitted git changes found to review.", None);
                    } else {
                        let max_diff = if trimmed.len() > 8000 { &trimmed[..8000] } else { trimmed };
                        let prompt = format!(
                            "Please perform a thorough code review of the following uncommitted git diff:\n\n\
                            ```diff\n{max_diff}\n```\n\n\
                            Identify any potential bugs, regressions, performance issues, or architectural improvements."
                        );
                        execute_tui_turn(tui, engine, None, &prompt, *allow_think, *max_tokens);
                        return true;
                    }
                }
                Err(e) => {
                    tui.push_message("system", &format!("Could not execute git diff: {e}"), None);
                }
            }
        }
        "/diff" => {
            let output = std::process::Command::new("git").args(["diff", "--stat"]).output();
            match output {
                Ok(out) => {
                    let stat_str = String::from_utf8_lossy(&out.stdout);
                    if stat_str.trim().is_empty() {
                        tui.push_message("system", "Working tree clean. No uncommitted modifications.", None);
                    } else {
                        tui.push_message("system", stat_str.trim(), None);
                    }
                }
                Err(e) => {
                    tui.push_message("system", &format!("Could not run git diff: {e}"), None);
                }
            }
        }
        "/undo" => {
            match crate::tools::pop_undo() {
                Some(Ok(msg)) => tui.push_message("system", &msg, None),
                Some(Err(e)) => tui.push_message("system", &format!("Undo failed: {e}"), None),
                None => tui.push_message("system", "No recent file modifications to undo.", None),
            }
        }
        cmd if cmd.starts_with("/bench") => {
            let n_tokens = if cmd.starts_with("/bench ") {
                cmd[7..].trim().parse::<usize>().unwrap_or(32).clamp(1, 256)
            } else {
                32
            };
            tui.push_message("system", &format!("Running on-demand benchmark generating {n_tokens} tokens..."), None);
            tui.render_chat();
            let prompt = "Explain in one sentence why systems programming in Rust is reliable and performant.";
            let (reply, secs, tokens) = unsafe { engine.generate_reply(prompt, n_tokens, false, &mut |_| {}) };
            let speed = tokens as f64 / secs.max(0.001);
            let result_msg = format!("Benchmark complete: {:.1} tok/s ({tokens} tokens in {:.2}s)\nSample: \"{}\"", speed, secs, reply.trim());
            tui.push_message("system", &result_msg, None);
        }
        "/stats" => {
            let (ws, private, _) = crate::memory_guard::process_counters();
            let free_ram = crate::mmap_file::available_ram_bytes() >> 20;
            let stats_msg = format!(
                "Engine Performance & Memory:\n  Resident: {} MiB · Private: {} MiB · Available RAM: {} MiB\n  KV-Cache: {}/{} tokens\n  MoE Tokens Evaluated: {}",
                ws >> 20, private >> 20, free_ram,
                engine.cur_pos, engine.max_seq,
                engine.moe_stats.tokens
            );
            tui.push_message("system", &stats_msg, None);
        }
        other => {
            tui.push_message("system", &format!("Unknown command '{other}'. Type /help for available commands."), None);
        }
    }

    tui.render_chat();
    true
}

/// Executes a conversation turn ASYNCHRONOUSLY on a background worker thread.
/// The UI thread remains fully interactive at 60 FPS: live tokens update continuously without
/// needing user input, while the user can scroll with mouse wheel, review history with arrow keys,
/// type their next draft into the input card, or cancel with Escape!
fn execute_tui_turn(
    tui: &mut SpindleTui,
    engine: &mut crate::smollm_chat::SmolLm2Engine,
    _governor: Option<&crate::memory_guard::Governor>,
    prompt: &str,
    allow_think: bool,
    max_tokens: usize,
) {
    tui.push_message("user", prompt, None);
    tui.scroll_offset = 0;
    tui.push_message("assistant", "Thinking...", None);
    tui.render_chat();

    let abort_flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    engine.abort_signal = Some(abort_flag.clone());

    let (tx, rx) = std::sync::mpsc::channel();

    std::thread::scope(|s| {
        let abort_worker = abort_flag.clone();
        let engine_ref = &mut *engine;

        s.spawn(move || {
            let mut current_input = prompt.to_string();
            let mut steps = 0;
            let mut last_call_sig = None;
            const MAX_STEPS: usize = 5;

            loop {
                if abort_worker.load(std::sync::atomic::Ordering::Relaxed) {
                    let _ = tx.send(WorkerEvent::Stopped);
                    break;
                }

                let tx_token = tx.clone();
                let (out_text, secs, tokens) = unsafe {
                    engine_ref.generate_reply(&current_input, max_tokens, allow_think, &mut |token_piece| {
                        let _ = tx_token.send(WorkerEvent::Token(token_piece.to_string()));
                    })
                };

                if abort_worker.load(std::sync::atomic::Ordering::Relaxed) {
                    let _ = tx.send(WorkerEvent::Stopped);
                    break;
                }

                let tok_per_sec = tokens as f64 / secs.max(0.001);
                let (acc, tot, _) = engine_ref.spec_stats;
                let speed_badge = if tot > 0 {
                    let pct = (acc as f64 / tot as f64) * 100.0;
                    format!("{tok_per_sec:.1} tok/s · {tokens} tokens · {secs:.1} s · spec {acc}/{tot} ({pct:.0}%)")
                } else {
                    format!("{tok_per_sec:.1} tok/s · {tokens} tokens · {secs:.1} s")
                };

                // Check if model emitted an agentic tool call
                if engine_ref.enable_tools && steps < MAX_STEPS {
                    if let Some(call) = crate::tools::parse_tool_call(&out_text) {
                        let call_sig = format!("{}:{:?}:{:?}", call.name, call.path, call.command);
                        if last_call_sig.as_ref() == Some(&call_sig) {
                            let _ = tx.send(WorkerEvent::ToolStart(format!("Action: {} (duplicate skipped)", call.name)));
                            current_input = "<tool_response>\nAction already completed. Please provide your final answer to the user.\n</tool_response>".to_string();
                            steps += 1;
                            continue;
                        }
                        last_call_sig = Some(call_sig);
                        steps += 1;

                        let action_summary = match (&call.path, &call.command) {
                            (Some(p), _) => format!("Action: {} (path: {p})", call.name),
                            (_, Some(c)) => format!("Action: {} (cmd: {c})", call.name),
                            _ => format!("Action: {}", call.name),
                        };
                        let _ = tx.send(WorkerEvent::ToolStart(action_summary));

                        let tool_result = match call.name.as_str() {
                            "read_file" => {
                                let p = call.path.as_deref().unwrap_or("");
                                crate::tools::tool_read_file(p, call.start_line, call.limit)
                            }
                            "write_file" => {
                                let p = call.path.as_deref().unwrap_or("");
                                let c = call.content.as_deref().unwrap_or("");
                                crate::tools::tool_write_file(p, c)
                            }
                            "edit_file" => {
                                let p = call.path.as_deref().unwrap_or("");
                                let t = call.target.as_deref().unwrap_or("");
                                let r = call.replacement.as_deref().unwrap_or("");
                                crate::tools::tool_edit_file(p, t, r)
                            }
                            "delete_file" => {
                                let p = call.path.as_deref().unwrap_or("");
                                crate::tools::tool_delete_file(p)
                            }
                            "list_dir" => {
                                let p = call.path.as_deref().unwrap_or(".");
                                crate::tools::tool_list_dir(p)
                            }
                            "run_command" => {
                                let cmd = call.command.as_deref().unwrap_or("");
                                crate::tools::tool_run_command(cmd, |_| true)
                            }
                            "search_web" => {
                                let q = call.query.as_deref().unwrap_or("");
                                crate::tools::tool_search_web(q)
                            }
                            "search_symbols" => {
                                let pat = call.pattern.as_deref().unwrap_or(call.query.as_deref().unwrap_or(""));
                                let p = call.path.as_deref();
                                let st = call.symbol_type.as_deref();
                                crate::tools::tool_search_symbols(pat, p, st)
                            }
                            other => Err(format!("Unknown tool: '{other}'")),
                        };

                        match tool_result {
                            Ok(res) => {
                                current_input = format!("<tool_response>\n{}\n</tool_response>", res);
                                continue;
                            }
                            Err(err) => {
                                current_input = format!("<tool_response>\nError: {}\n</tool_response>", err);
                                continue;
                            }
                        }
                    }
                }

                let display_reply = out_text.trim().to_string();
                let _ = tx.send(WorkerEvent::Done {
                    reply: display_reply,
                    speed_badge,
                });
                break;
            }
        });

        // Main UI Thread: 60 FPS non-blocking Event Loop during live streaming
        let mut stream_buffer = String::new();
        let mut is_generating = true;
        let mut last_render = std::time::Instant::now();

        while is_generating {
            let mut updated = false;

            // 1. Drain all pending incoming worker events (live streaming tokens)
            while let Ok(event) = rx.try_recv() {
                match event {
                    WorkerEvent::Token(piece) => {
                        stream_buffer.push_str(&piece);
                        if let Some(last_msg) = tui.history.last_mut() {
                            if last_msg.role == "assistant" {
                                last_msg.text = stream_buffer.clone();
                            }
                        }
                        updated = true;
                    }
                    WorkerEvent::ToolStart(summary) => {
                        if let Some(last_msg) = tui.history.last_mut() {
                            if last_msg.role == "assistant" && stream_buffer.is_empty() {
                                tui.history.pop();
                            }
                        }
                        tui.push_message("tool", &summary, None);
                        tui.push_message("assistant", "Thinking...", None);
                        stream_buffer.clear();
                        updated = true;
                    }
                    WorkerEvent::Done { reply, speed_badge } => {
                        if let Some(last_msg) = tui.history.last_mut() {
                            if last_msg.role == "assistant" {
                                last_msg.text = if reply.is_empty() { stream_buffer.clone() } else { reply };
                                last_msg.speed_badge = Some(speed_badge);
                            } else {
                                tui.push_message("assistant", &reply, Some(&speed_badge));
                            }
                        }
                        is_generating = false;
                        updated = true;
                    }
                    WorkerEvent::Stopped => {
                        if let Some(last_msg) = tui.history.last_mut() {
                            if last_msg.role == "assistant" {
                                if stream_buffer.is_empty() {
                                    last_msg.text = "[Generation stopped by user]".to_string();
                                } else {
                                    last_msg.text = format!("{} [stopped]", stream_buffer.trim());
                                }
                                last_msg.speed_badge = Some("stopped".to_string());
                            }
                        }
                        is_generating = false;
                        updated = true;
                    }
                }
            }

            // 2. Non-blocking user keyboard & mouse input handling while streaming!
            match tui.poll_key() {
                KeyAction::Char(c) => {
                    tui.input.insert(c);
                    updated = true;
                }
                KeyAction::Backspace => {
                    tui.input.backspace();
                    updated = true;
                }
                KeyAction::Delete => {
                    tui.input.delete();
                    updated = true;
                }
                KeyAction::Left => {
                    tui.input.move_left();
                    updated = true;
                }
                KeyAction::Right => {
                    tui.input.move_right();
                    updated = true;
                }
                KeyAction::Home => {
                    tui.input.move_home();
                    updated = true;
                }
                KeyAction::End => {
                    tui.input.move_end();
                    updated = true;
                }
                KeyAction::ScrollUp(n) => {
                    tui.scroll_offset = tui.scroll_offset.saturating_add(n);
                    updated = true;
                }
                KeyAction::ScrollDown(n) => {
                    tui.scroll_offset = tui.scroll_offset.saturating_sub(n);
                    updated = true;
                }
                KeyAction::PageUp => {
                    tui.scroll_offset = tui.scroll_offset.saturating_add(6);
                    updated = true;
                }
                KeyAction::PageDown => {
                    tui.scroll_offset = tui.scroll_offset.saturating_sub(6);
                    updated = true;
                }
                KeyAction::Up => {
                    tui.scroll_offset = tui.scroll_offset.saturating_add(2);
                    updated = true;
                }
                KeyAction::Down => {
                    tui.scroll_offset = tui.scroll_offset.saturating_sub(2);
                    updated = true;
                }
                KeyAction::Escape => {
                    abort_flag.store(true, std::sync::atomic::Ordering::SeqCst);
                }
                KeyAction::Enter => {
                    // If user pressed Enter with a typed draft, abort current generation to send new prompt!
                    if !tui.input.text.trim().is_empty() {
                        abort_flag.store(true, std::sync::atomic::Ordering::SeqCst);
                    }
                }
                _ => {}
            }

            // 3. Render frame if updated or throttled to ~28-30 FPS
            if updated || last_render.elapsed().as_millis() >= 35 {
                tui.render_chat();
                last_render = std::time::Instant::now();
            }

            std::thread::sleep(std::time::Duration::from_millis(10));
        }
    });

    engine.abort_signal = None;
    tui.render_chat();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_canvas_draw_and_clear() {
        let mut canvas = Canvas::new(80, 24);
        assert_eq!(canvas.width, 80);
        assert_eq!(canvas.height, 24);

        canvas.set(5, 5, 'A', Rgb::SKY_BLUE, Rgb::CARD_BG, true, false);
        let idx = 5 * 80 + 5;
        assert_eq!(canvas.grid[idx].ch, 'A');
        assert_eq!(canvas.grid[idx].fg, Rgb::SKY_BLUE);

        canvas.clear(Rgb::BG_DARK);
        assert_eq!(canvas.grid[idx].ch, ' ');
        assert_eq!(canvas.grid[idx].bg, Rgb::BG_DARK);
    }

    #[test]
    fn test_canvas_draw_card() {
        let mut canvas = Canvas::new(80, 24);
        canvas.draw_card(10, 5, 40, 6, Rgb::CARD_BG, Rgb::BRIGHT_BLUE);

        let bar_idx = 5 * 80 + 10;
        assert_eq!(canvas.grid[bar_idx].ch, '▌');
        assert_eq!(canvas.grid[bar_idx].fg, Rgb::BRIGHT_BLUE);

        let tr_idx = 5 * 80 + 49;
        assert_eq!(canvas.grid[tr_idx].ch, '╮');
    }

    #[test]
    fn test_input_buffer_operations() {
        let mut buf = InputBuffer::default();
        buf.insert('h');
        buf.insert('i');
        assert_eq!(buf.text, "hi");
        assert_eq!(buf.cursor, 2);

        buf.backspace();
        assert_eq!(buf.text, "h");
        assert_eq!(buf.cursor, 1);

        buf.insert('e');
        buf.insert('y');
        assert_eq!(buf.text, "hey");

        buf.move_home();
        assert_eq!(buf.cursor, 0);

        buf.move_right();
        assert_eq!(buf.cursor, 1);

        buf.delete();
        assert_eq!(buf.text, "hy");

        // Multi-byte UTF-8 test (ñ is 2 bytes: 0xC3 0xB1)
        buf.clear();
        buf.insert('k');
        buf.insert('u');
        buf.insert('m');
        buf.insert('u');
        buf.insert('s');
        buf.insert('t');
        buf.insert('a');
        assert_eq!(buf.text, "kumusta");
        assert_eq!(buf.cursor, 7);

        buf.insert('ñ');
        assert_eq!(buf.text, "kumustañ");
        assert_eq!(buf.cursor, 8);

        buf.backspace();
        assert_eq!(buf.text, "kumusta");
        assert_eq!(buf.cursor, 7);
    }

    #[test]
    fn test_spindle_tui_welcome_and_chat_render() {
        let mut tui = SpindleTui::new("Qwen3-30B-A3B", Some("192.168.1.10:42424"), "fp8", 0.03, "layers 35..48");
        tui.canvas.resize(80, 24);
        tui.render_welcome();
        assert!(tui.running);

        // Test chronological chat rendering
        tui.push_message("user", "Hello Spindle", None);
        tui.push_message("assistant", "I am ready to assist you.", Some("8.5 tok/s"));
        tui.render_chat();
        assert_eq!(tui.history.len(), 2);
    }

    #[test]
    fn test_wrap_text() {
        let long_text = "I am Qwen, a large language model created by Alibaba. I can help you with various tasks such as answering questions, generating code, reviewing files, and automating your developer workflows.";
        let wrapped = wrap_text(long_text, 40);
        assert!(wrapped.len() >= 3);
        for line in &wrapped {
            assert!(line.chars().count() <= 40, "Line exceeds max width 40: '{}'", line);
        }
    }

    #[test]
    fn test_engine_send() {
        fn assert_send<T: Send>() {}
        assert_send::<crate::smollm_chat::SmolLm2Engine>();
    }

    #[test]
    fn test_thread_scope() {
        let mut val = 42;
        std::thread::scope(|s| {
            s.spawn(|| {
                val += 1;
            });
        });
        assert_eq!(val, 43);
    }

    #[test]
    fn test_scope_reborrow() {
        let mut num = 100;
        let num_ref = &mut num;
        std::thread::scope(|s| {
            let r = &mut *num_ref;
            s.spawn(move || {
                *r += 50;
            });
        });
        assert_eq!(*num_ref, 150);
    }

    #[test]
    fn test_scroll_navigation() {
        let mut tui = SpindleTui::new("Qwen3-30B-A3B", None, "fp8", 0.03, "layers 0..48");
        assert_eq!(tui.scroll_offset, 0);

        // Simulate ScrollUp (mouse wheel up)
        tui.scroll_offset = tui.scroll_offset.saturating_add(3);
        assert_eq!(tui.scroll_offset, 3);

        // Simulate ScrollDown (mouse wheel down)
        tui.scroll_offset = tui.scroll_offset.saturating_sub(3);
        assert_eq!(tui.scroll_offset, 0);

        // Simulate PageUp
        tui.scroll_offset = tui.scroll_offset.saturating_add(6);
        assert_eq!(tui.scroll_offset, 6);

        // Simulate PageDown
        tui.scroll_offset = tui.scroll_offset.saturating_sub(6);
        assert_eq!(tui.scroll_offset, 0);
    }

    #[test]
    fn test_parse_input_bytes_swallows_mouse_clicks_and_parses_wheel() {
        // 1. Mouse click press & release sequences: button 0 press (\x1b[<0;77;27M) and release (\x1b[<0;77;27m)
        let click_bytes = b"\x1b[<0;77;27M\x1b[<0;77;27m";
        let (actions, consumed) = parse_input_bytes(click_bytes);
        assert_eq!(consumed, click_bytes.len());
        // All click actions are None
        for a in &actions {
            assert_eq!(*a, KeyAction::None);
        }

        // 2. Mouse wheel up: button 64 (\x1b[<64;77;27M)
        let wheel_up = b"\x1b[<64;77;27M";
        let (actions, consumed) = parse_input_bytes(wheel_up);
        assert_eq!(consumed, wheel_up.len());
        assert_eq!(actions, vec![KeyAction::ScrollUp(3)]);

        // 3. Mouse wheel down: button 65 (\x1b[<65;77;27M)
        let wheel_down = b"\x1b[<65;77;27M";
        let (actions, consumed) = parse_input_bytes(wheel_down);
        assert_eq!(consumed, wheel_down.len());
        assert_eq!(actions, vec![KeyAction::ScrollDown(3)]);

        // 4. Orphaned CSI fragments (like [<0;77;27M) without leading ESC
        let orphan = b"[<0;77;27M";
        let (actions, consumed) = parse_input_bytes(orphan);
        assert_eq!(consumed, orphan.len());
        assert_eq!(actions.len(), 0);

        // 5. Orphaned SGR semicolon fragments (like ;77;27M)
        let orphan_semi = b";77;27M";
        let (actions, consumed) = parse_input_bytes(orphan_semi);
        assert_eq!(consumed, orphan_semi.len());
        assert_eq!(actions.len(), 0);

        // 6. Normal text mixed with swallowed mouse clicks
        let mixed = b"hello\x1b[<0;10;10M world";
        let (actions, consumed) = parse_input_bytes(mixed);
        assert_eq!(consumed, mixed.len());
        let text: String = actions.into_iter().filter_map(|a| match a {
            KeyAction::Char(c) => Some(c),
            _ => None,
        }).collect();
        assert_eq!(text, "hello world");
    }
}
