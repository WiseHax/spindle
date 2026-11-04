//! OpenCode-grade Full-Screen Terminal User Interface (TUI) for Spindle.
//!
//! Provides an alternate screen canvas, 24-bit TrueColor differential double-buffering,
//! floating elevated card layout, real-time keyboard event processing, and live token streaming.

use std::io::{self, Read, Write};
use windows::Win32::System::Console::{
    GetConsoleMode, GetConsoleScreenBufferInfo, GetStdHandle, SetConsoleMode,
    CONSOLE_MODE, CONSOLE_SCREEN_BUFFER_INFO, STD_INPUT_HANDLE, STD_OUTPUT_HANDLE,
    ENABLE_ECHO_INPUT, ENABLE_LINE_INPUT, ENABLE_PROCESSED_INPUT,
    ENABLE_VIRTUAL_TERMINAL_INPUT, ENABLE_VIRTUAL_TERMINAL_PROCESSING,
};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Rgb(pub u8, pub u8, pub u8);

impl Rgb {
    pub const BLACK: Rgb = Rgb(12, 13, 15);
    pub const BG_DARK: Rgb = Rgb(15, 16, 20);
    pub const CARD_BG: Rgb = Rgb(26, 28, 35);
    pub const CARD_BORDER: Rgb = Rgb(48, 52, 64);
    pub const WHITE: Rgb = Rgb(248, 250, 252);
    pub const CYAN: Rgb = Rgb(56, 189, 248);
    pub const ELECTRIC_BLUE: Rgb = Rgb(37, 99, 235);
    pub const GREEN: Rgb = Rgb(52, 211, 153);
    pub const ORANGE: Rgb = Rgb(249, 115, 22);
    pub const MUTED: Rgb = Rgb(148, 163, 184);
    pub const DIM: Rgb = Rgb(80, 90, 105);
    pub const GOLD: Rgb = Rgb(251, 191, 36);
    pub const MAGENTA: Rgb = Rgb(217, 70, 239);
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
            // Enter alternate screen buffer and hide cursor
            let _ = write!(stdout, "\x1b[?1049h\x1b[?25l");
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
            // Show cursor and exit alternate screen buffer
            let _ = write!(stdout, "\x1b[?25h\x1b[?1049l\x1b[0m");
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

/// High-performance 2D differential frame buffer with zero flicker.
pub struct Canvas {
    pub width: u16,
    pub height: u16,
    front: Vec<Cell>,
    back: Vec<Cell>,
}

impl Canvas {
    pub fn new(width: u16, height: u16) -> Self {
        let size = (width as usize) * (height as usize);
        Self {
            width,
            height,
            front: vec![Cell::default(); size],
            back: vec![Cell::default(); size],
        }
    }

    pub fn resize(&mut self, width: u16, height: u16) {
        if self.width == width && self.height == height {
            return;
        }
        self.width = width;
        self.height = height;
        let size = (width as usize) * (height as usize);
        self.front = vec![Cell::default(); size];
        self.back = vec![Cell::default(); size];
    }

    pub fn clear(&mut self, bg: Rgb) {
        for c in self.back.iter_mut() {
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
            self.back[idx] = Cell { ch, fg, bg, bold, dim };
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

    /// Renders an OpenCode-style elevated card with an electric cyan accent bar on the left.
    pub fn draw_card(&mut self, x: u16, y: u16, w: u16, h: u16, bg: Rgb, accent_bar: Rgb) {
        // Fill card background
        self.fill_rect(x, y, w, h, ' ', Rgb::WHITE, bg);

        // Draw subtle border around card
        let border_fg = Rgb::CARD_BORDER;
        for col in x..x + w {
            self.set(col, y, '─', border_fg, bg, false, true);
            self.set(col, y + h - 1, '─', border_fg, bg, false, true);
        }
        for row in y..y + h {
            self.set(x, row, '│', border_fg, bg, false, true);
            self.set(x + w - 1, row, '│', border_fg, bg, false, true);
        }
        self.set(x, y, '┌', border_fg, bg, false, true);
        self.set(x + w - 1, y, '┐', border_fg, bg, false, true);
        self.set(x, y + h - 1, '└', border_fg, bg, false, true);
        self.set(x + w - 1, y + h - 1, '┘', border_fg, bg, false, true);

        // Left prominent accent bar (OpenCode signature `▌`)
        for row in (y + 1)..(y + h - 1) {
            self.set(x, row, '▌', accent_bar, bg, true, false);
        }
    }

    /// Flushes only changed cells to stdout using 24-bit TrueColor ANSI escape sequences.
    pub fn flush(&mut self) -> io::Result<()> {
        let mut out = io::stdout();
        let mut current_fg = Rgb(0, 0, 0);
        let mut current_bg = Rgb(0, 0, 0);
        let mut current_bold = false;
        let mut current_dim = false;
        let mut in_escape = false;

        for y in 0..self.height {
            let mut x = 0;
            while x < self.width {
                let idx = (y as usize) * (self.width as usize) + (x as usize);
                if self.back[idx] != self.front[idx] {
                    // Position cursor to (row, col) - 1-based in ANSI
                    write!(out, "\x1b[{};{}H", y + 1, x + 1)?;

                    // Stream contiguous changed cells
                    while x < self.width {
                        let c_idx = (y as usize) * (self.width as usize) + (x as usize);
                        if self.back[c_idx] == self.front[c_idx] && (x + 1 < self.width && self.back[c_idx + 1] == self.front[c_idx + 1]) {
                            break;
                        }
                        let cell = self.back[c_idx];

                        // Update attributes if changed
                        if !in_escape || cell.fg != current_fg || cell.bg != current_bg || cell.bold != current_bold || cell.dim != current_dim {
                            write!(out, "\x1b[0;38;2;{};{};{};48;2;{};{};{}m",
                                cell.fg.0, cell.fg.1, cell.fg.2,
                                cell.bg.0, cell.bg.1, cell.bg.2
                            )?;
                            if cell.bold {
                                write!(out, "\x1b[1m")?;
                            }
                            if cell.dim {
                                write!(out, "\x1b[2m")?;
                            }
                            current_fg = cell.fg;
                            current_bg = cell.bg;
                            current_bold = cell.bold;
                            current_dim = cell.dim;
                            in_escape = true;
                        }

                        write!(out, "{}", cell.ch)?;
                        self.front[c_idx] = cell;
                        x += 1;
                    }
                } else {
                    x += 1;
                }
            }
        }

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

/// Keys captured during the event loop.
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
    Tab,
    CtrlP,
    Escape,
    None,
}

/// Non-blocking raw keyboard reader using ANSI/VT escape parser.
pub fn read_key_nonblocking() -> KeyAction {
    let mut stdin = io::stdin();
    let mut byte = [0u8; 1];

    // Read single byte
    if stdin.read(&mut byte).unwrap_or(0) == 0 {
        return KeyAction::None;
    }

    match byte[0] {
        13 | 10 => KeyAction::Enter,
        8 | 127 => KeyAction::Backspace,
        9 => KeyAction::Tab,
        16 => KeyAction::CtrlP, // Ctrl+P
        27 => {
            // Escape sequence or Alt
            let mut seq = [0u8; 2];
            if stdin.read(&mut seq[0..1]).unwrap_or(0) == 1 {
                if seq[0] == b'[' {
                    if stdin.read(&mut seq[1..2]).unwrap_or(0) == 1 {
                        match seq[1] {
                            b'A' => KeyAction::Up,
                            b'B' => KeyAction::Down,
                            b'C' => KeyAction::Right,
                            b'D' => KeyAction::Left,
                            b'H' => KeyAction::Home,
                            b'F' => KeyAction::End,
                            b'3' => {
                                let mut tilde = [0u8; 1];
                                let _ = stdin.read(&mut tilde);
                                KeyAction::Delete
                            }
                            _ => KeyAction::None,
                        }
                    } else {
                        KeyAction::Escape
                    }
                } else {
                    KeyAction::Escape
                }
            } else {
                KeyAction::Escape
            }
        }
        ch if ch >= 32 => {
            if let Some(c) = char::from_u32(ch as u32) {
                KeyAction::Char(c)
            } else {
                KeyAction::None
            }
        }
        _ => KeyAction::None,
    }
}

/// Text input line editor state.
#[derive(Default)]
pub struct InputBuffer {
    pub text: String,
    pub cursor: usize,
}

impl InputBuffer {
    pub fn insert(&mut self, ch: char) {
        self.text.insert(self.cursor, ch);
        self.cursor += 1;
    }

    pub fn backspace(&mut self) {
        if self.cursor > 0 {
            self.cursor -= 1;
            self.text.remove(self.cursor);
        }
    }

    pub fn delete(&mut self) {
        if self.cursor < self.text.len() {
            self.text.remove(self.cursor);
        }
    }

    pub fn move_left(&mut self) {
        if self.cursor > 0 {
            self.cursor -= 1;
        }
    }

    pub fn move_right(&mut self) {
        if self.cursor < self.text.len() {
            self.cursor += 1;
        }
    }

    pub fn move_home(&mut self) {
        self.cursor = 0;
    }

    pub fn move_end(&mut self) {
        self.cursor = self.text.len();
    }

    pub fn clear(&mut self) {
        self.text.clear();
        self.cursor = 0;
    }
}

/// Message turn stored for rendering conversation history.
pub struct ChatMessage {
    pub role: String,
    pub text: String,
    pub speed_badge: Option<String>,
}

/// The Full-Screen TUI Orchestrator.
pub struct SpindleTui {
    canvas: Canvas,
    input: InputBuffer,
    history: Vec<ChatMessage>,
    pub running: bool,
    pub model_name: String,
    pub cluster_peer: Option<String>,
    pub wire_format: String,
    pub prune_val: f32,
    pub layers_info: String,
    pub git_branch: String,
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
            running: true,
            model_name: model_name.to_string(),
            cluster_peer: cluster_peer.map(str::to_string),
            wire_format: wire_format.to_string(),
            prune_val,
            layers_info: layers_info.to_string(),
            git_branch: branch,
        }
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

    /// Renders the signature OpenCode-style hero screen when no active turn is in progress.
    pub fn render_welcome(&mut self) {
        let (w, h) = terminal_size();
        self.canvas.resize(w, h);
        self.canvas.clear(Rgb::BG_DARK);

        let center_x = w / 2;
        let logo_y = (h / 6).max(2);

        // 1. Centered Hero Block Logo with Electric Cyan/Blue Gradient
        for (i, line) in SPINDLE_LOGO.iter().enumerate() {
            let row = logo_y + (i as u16);
            let len = line.chars().count() as u16;
            let start_x = center_x.saturating_sub(len / 2);
            let color = if i < 2 { Rgb::WHITE } else { Rgb::CYAN };
            self.canvas.print(start_x, row, line, color, Rgb::BG_DARK, true, false);
        }

        // 2. Subtitle Badge
        let sub = "/help for slash commands · /heal autonomous loop";
        let sub_x = center_x.saturating_sub((sub.len() as u16) / 2);
        self.canvas.print(sub_x, logo_y + 5, sub, Rgb::MUTED, Rgb::BG_DARK, false, false);

        // 3. Floating Input Card
        let card_w = 82.min(w.saturating_sub(6));
        let card_h = 6;
        let card_x = center_x.saturating_sub(card_w / 2);
        let card_y = logo_y + 7;

        self.canvas.draw_card(card_x, card_y, card_w, card_h, Rgb::CARD_BG, Rgb::CYAN);

        // Content Row: Text or Ghost Placeholder
        let prompt_x = card_x + 3;
        let prompt_y = card_y + 1;
        if self.input.text.is_empty() {
            let ghost = "Ask anything... \"delete hello.py\", \"/review\", \"/stats\"";
            self.canvas.print(prompt_x, prompt_y, ghost, Rgb::DIM, Rgb::CARD_BG, false, true);
            // Blinking cursor indicator
            self.canvas.print(prompt_x, prompt_y, "█", Rgb::CYAN, Rgb::CARD_BG, true, false);
        } else {
            // Render user input up to visible card width
            let visible_w = (card_w.saturating_sub(6)) as usize;
            let display_text: String = self.input.text.chars().take(visible_w).collect();
            self.canvas.print(prompt_x, prompt_y, &display_text, Rgb::WHITE, Rgb::CARD_BG, false, false);
            // Render block cursor at insertion point
            let cursor_x = prompt_x + (self.input.cursor as u16).min(visible_w as u16);
            self.canvas.set(cursor_x, prompt_y, '█', Rgb::CYAN, Rgb::CARD_BG, true, false);
        }

        // Bottom Meta Badges inside the card
        let meta_y = card_y + 3;
        let mut mx = prompt_x;

        // Build Badge (Cyan pill)
        self.canvas.print(mx, meta_y, "Build", Rgb::CYAN, Rgb::CARD_BG, true, false);
        mx += 6;

        // Model Badge
        let model_badge = format!("· {} ({})", self.model_name, self.layers_info);
        self.canvas.print(mx, meta_y, &model_badge, Rgb::WHITE, Rgb::CARD_BG, false, false);
        mx += model_badge.len() as u16 + 1;

        // Prune Badge
        let prune_badge = format!("· {:.2} prune", self.prune_val);
        self.canvas.print(mx, meta_y, &prune_badge, Rgb::ORANGE, Rgb::CARD_BG, false, false);
        mx += prune_badge.len() as u16 + 1;

        // Cluster Mesh Badge
        if let Some(peer) = &self.cluster_peer {
            let mesh_badge = format!("· mesh: {peer}");
            self.canvas.print(mx, meta_y, &mesh_badge, Rgb::GREEN, Rgb::CARD_BG, false, false);
        }

        // 4. Hotkey legends below the card
        let hotkeys = "tab tools    ctrl+p commands    /help slash commands    esc quit";
        let hk_x = center_x.saturating_sub((hotkeys.len() as u16) / 2);
        self.canvas.print(hk_x, card_y + card_h + 1, hotkeys, Rgb::MUTED, Rgb::BG_DARK, false, false);

        // 5. Docked Global Footer (Pinned at row = h - 1)
        self.render_docked_footer(w, h);

        let _ = self.canvas.flush();
    }

    /// Renders the conversation view when messages exist.
    pub fn render_chat(&mut self) {
        let (w, h) = terminal_size();
        self.canvas.resize(w, h);
        self.canvas.clear(Rgb::BG_DARK);

        // 1. Persistent Top Header (Row 0)
        let header_bg = Rgb::CARD_BG;
        self.canvas.fill_rect(0, 0, w, 1, ' ', Rgb::WHITE, header_bg);
        self.canvas.print(2, 0, "SPINDLE v0.1.0", Rgb::CYAN, header_bg, true, false);
        let status = format!("· {} · {} · peak 9.0 tok/s", self.model_name, self.layers_info);
        self.canvas.print(18, 0, &status, Rgb::MUTED, header_bg, false, false);

        if let Some(p) = &self.cluster_peer {
            let mesh_str = format!("mesh: {p}");
            let mx = w.saturating_sub(mesh_str.len() as u16 + 4);
            self.canvas.print(mx, 0, &mesh_str, Rgb::GREEN, header_bg, true, false);
        }

        // 2. Chat history rendering (Rows 2 to h - 8)
        let mut current_y = 2;
        let max_chat_y = h.saturating_sub(8);

        for msg in self.history.iter().rev() {
            if current_y >= max_chat_y {
                break;
            }
            if msg.role == "user" {
                let badge = "> User";
                self.canvas.print(2, current_y, badge, Rgb::CYAN, Rgb::BG_DARK, true, false);
                self.canvas.print(9, current_y, &msg.text, Rgb::WHITE, Rgb::BG_DARK, false, false);
                current_y += 2;
            } else if msg.role == "tool" {
                self.canvas.print(2, current_y, "┌ Tool Call", Rgb::MUTED, Rgb::BG_DARK, false, true);
                self.canvas.print(14, current_y, &msg.text, Rgb::GOLD, Rgb::BG_DARK, false, false);
                current_y += 2;
            } else {
                let badge = "Spindle";
                self.canvas.print(2, current_y, badge, Rgb::ELECTRIC_BLUE, Rgb::BG_DARK, true, false);
                if let Some(speed) = &msg.speed_badge {
                    self.canvas.print(11, current_y, speed, Rgb::MUTED, Rgb::BG_DARK, false, true);
                }
                current_y += 1;

                // Render assistant lines
                for line in msg.text.lines().take(4) {
                    if current_y >= max_chat_y {
                        break;
                    }
                    self.canvas.print(4, current_y, line, Rgb::WHITE, Rgb::BG_DARK, false, false);
                    current_y += 1;
                }
                current_y += 1;
            }
        }

        // 3. Bottom Docked Input Card (Rows h - 7 to h - 2)
        let card_w = 84.min(w.saturating_sub(4));
        let card_h = 5;
        let card_x = (w / 2).saturating_sub(card_w / 2);
        let card_y = h.saturating_sub(7);

        self.canvas.draw_card(card_x, card_y, card_w, card_h, Rgb::CARD_BG, Rgb::CYAN);

        let prompt_x = card_x + 3;
        let prompt_y = card_y + 1;
        if self.input.text.is_empty() {
            self.canvas.print(prompt_x, prompt_y, "Ask anything...", Rgb::DIM, Rgb::CARD_BG, false, true);
            self.canvas.print(prompt_x, prompt_y, "█", Rgb::CYAN, Rgb::CARD_BG, true, false);
        } else {
            let visible_w = (card_w.saturating_sub(6)) as usize;
            let display: String = self.input.text.chars().take(visible_w).collect();
            self.canvas.print(prompt_x, prompt_y, &display, Rgb::WHITE, Rgb::CARD_BG, false, false);
            let cursor_x = prompt_x + (self.input.cursor as u16).min(visible_w as u16);
            self.canvas.set(cursor_x, prompt_y, '█', Rgb::CYAN, Rgb::CARD_BG, true, false);
        }

        // Card meta line
        self.canvas.print(prompt_x, card_y + 3, "Build · Qwen3-30B-A3B · 0.03 prune · low-mem", Rgb::MUTED, Rgb::CARD_BG, false, false);

        // 4. Docked Global Footer
        self.render_docked_footer(w, h);

        let _ = self.canvas.flush();
    }

    fn render_docked_footer(&mut self, w: u16, h: u16) {
        let footer_y = h.saturating_sub(1);
        let cwd = std::env::current_dir()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|_| ".".to_string());
        let footer_left = format!("{}:{}", cwd, self.git_branch);
        self.canvas.print(2, footer_y, &footer_left, Rgb::MUTED, Rgb::BG_DARK, false, true);

        let footer_right = "spindle v0.1.0";
        let right_x = w.saturating_sub(footer_right.len() as u16 + 2);
        self.canvas.print(right_x, footer_y, footer_right, Rgb::MUTED, Rgb::BG_DARK, false, true);
    }

    /// Appends a completed message turn to conversation history.
    pub fn push_message(&mut self, role: &str, text: &str, speed_badge: Option<&str>) {
        self.history.push(ChatMessage {
            role: role.to_string(),
            text: text.to_string(),
            speed_badge: speed_badge.map(str::to_string),
        });
    }

    /// Runs a prompt turn through the TUI loop, rendering frames and collecting keyboard input.
    pub fn poll_input(&mut self) -> Option<String> {
        loop {
            if self.history.is_empty() {
                self.render_welcome();
            } else {
                self.render_chat();
            }

            match read_key_nonblocking() {
                KeyAction::Char(c) => self.input.insert(c),
                KeyAction::Backspace => self.input.backspace(),
                KeyAction::Delete => self.input.delete(),
                KeyAction::Left => self.input.move_left(),
                KeyAction::Right => self.input.move_right(),
                KeyAction::Home => self.input.move_home(),
                KeyAction::End => self.input.move_end(),
                KeyAction::Enter => {
                    let text = self.input.text.trim().to_string();
                    if !text.is_empty() {
                        self.input.clear();
                        return Some(text);
                    }
                }
                KeyAction::Escape => {
                    self.running = false;
                    return None;
                }
                KeyAction::None => {
                    std::thread::sleep(std::time::Duration::from_millis(16)); // ~60 FPS idle
                }
                _ => {}
            }
        }
    }
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

    let max_tokens = if opts.max_tokens > 0 { opts.max_tokens.min(engine.max_seq) } else { 2048.min(engine.max_seq) };

    // If an initial prompt was provided, execute it immediately
    if let Some(first_p) = opts.first_prompt {
        execute_tui_turn(&mut tui, engine, governor, &first_p, opts.allow_think, max_tokens);
    }

    // Main TUI interactive event loop
    while tui.running {
        if let Some(user_prompt) = tui.poll_input() {
            if user_prompt == "/exit" || user_prompt == "/quit" || user_prompt == "exit" || user_prompt == "quit" {
                break;
            }
            if user_prompt == "/clear" {
                tui.history.clear();
                continue;
            }
            execute_tui_turn(&mut tui, engine, governor, &user_prompt, opts.allow_think, max_tokens);
        }
    }

    drop(guard);
}

fn execute_tui_turn(
    tui: &mut SpindleTui,
    engine: &mut crate::smollm_chat::SmolLm2Engine,
    _governor: Option<&crate::memory_guard::Governor>,
    prompt: &str,
    allow_think: bool,
    max_tokens: usize,
) {
    tui.push_message("user", prompt, None);
    tui.render_chat();

    let mut current_input = prompt.to_string();
    let mut steps = 0;
    let mut last_call_sig = None;
    const MAX_STEPS: usize = 5;

    loop {
        let (out_text, secs, tokens) = unsafe {
            engine.generate_reply(&current_input, max_tokens, allow_think, &mut |_token_piece| {
                // Future enhancement: incremental token stream update
            })
        };

        let tok_per_sec = tokens as f64 / secs.max(0.001);
        let speed_badge = format!("{tok_per_sec:.1} tok/s · {tokens} tokens · {secs:.1} s");

        // Check if model emitted a tool call
        if engine.enable_tools && steps < MAX_STEPS {
            if let Some(call) = crate::tools::parse_tool_call(&out_text) {
                let call_sig = format!("{}:{:?}:{:?}", call.name, call.path, call.command);
                if last_call_sig.as_ref() == Some(&call_sig) {
                    tui.push_message("tool", &format!("Action: {} (duplicate skipped)", call.name), None);
                    tui.render_chat();
                    current_input = "<tool_response>\nAction already completed. Please provide your final answer to the user.\n</tool_response>".to_string();
                    steps += 1;
                    continue;
                }
                last_call_sig = Some(call_sig);
                steps += 1;

                tui.push_message("tool", &format!("Action: {}", call.name), None);
                tui.render_chat();

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

        let display_reply = out_text.trim();
        tui.push_message("assistant", display_reply, Some(&speed_badge));
        tui.render_chat();
        break;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_canvas_draw_and_clear() {
        let mut canvas = Canvas::new(80, 24);
        assert_eq!(canvas.width, 80);
        assert_eq!(canvas.height, 24);

        canvas.set(5, 5, 'A', Rgb::CYAN, Rgb::CARD_BG, true, false);
        let idx = 5 * 80 + 5;
        assert_eq!(canvas.back[idx].ch, 'A');
        assert_eq!(canvas.back[idx].fg, Rgb::CYAN);

        canvas.clear(Rgb::BLACK);
        assert_eq!(canvas.back[idx].ch, ' ');
        assert_eq!(canvas.back[idx].bg, Rgb::BLACK);
    }

    #[test]
    fn test_canvas_draw_card() {
        let mut canvas = Canvas::new(80, 24);
        canvas.draw_card(10, 5, 40, 6, Rgb::CARD_BG, Rgb::CYAN);

        // Check top-left corner
        let tl_idx = 5 * 80 + 10;
        assert_eq!(canvas.back[tl_idx].ch, '┌');

        // Check left accent bar at row 6
        let bar_idx = 6 * 80 + 10;
        assert_eq!(canvas.back[bar_idx].ch, '▌');
        assert_eq!(canvas.back[bar_idx].fg, Rgb::CYAN);
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

        buf.delete(); // deletes 'e'
        assert_eq!(buf.text, "hy");
    }

    #[test]
    fn test_spindle_tui_welcome_render_no_panic() {
        let mut tui = SpindleTui::new("Qwen3-30B-A3B", Some("192.168.1.10:42424"), "fp8", 0.03, "layers 35..48");
        // Ensure rendering welcome does not panic even on smaller virtual canvases
        tui.canvas.resize(80, 24);
        tui.render_welcome();
        assert!(tui.running);
    }
}
