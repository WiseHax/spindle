//! Agent tools: file read/write/edit, search (grep/glob/symbols), shell, web.
//!
//! Tool calls arrive in Qwen's native format (the one its chat template was trained on):
//! `<tool_call>{"name": ..., "arguments": {...}}</tool_call>`, parsed with a real JSON parser
//! (`crate::json`). Every tool returns a [`ToolOutcome`]: text for the model, plus a short
//! title and optional preview (diff / command output) for the UI.

use std::{
    fs,
    path::{Path, PathBuf},
    sync::Mutex,
    time::Duration,
};

use crate::{
    guard::{self, GuardDecision},
    json::{self, Json},
    vault,
};

// ------------------------------------------------------------------------------- undo

static UNDO_STACK: Mutex<Vec<(String, Option<String>)>> = Mutex::new(Vec::new());

/// Saves a shadow snapshot of the file before modifying, enabling `/undo`.
pub fn push_undo_snapshot(path_str: &str) {
    let p = Path::new(path_str);
    let prev = if p.exists() { fs::read_to_string(p).ok() } else { None };
    if let Ok(mut stack) = UNDO_STACK.lock() {
        stack.push((path_str.to_string(), prev));
    }
}

fn restore(path: String, prev: Option<String>) -> Result<String, String> {
    match prev {
        Some(old) => fs::write(&path, old)
            .map(|_| format!("Restored '{path}' to its previous state."))
            .map_err(|e| format!("Failed to restore '{path}': {e}")),
        None => fs::remove_file(&path)
            .map(|_| format!("Removed newly created file '{path}'."))
            .map_err(|e| format!("Failed to remove '{path}': {e}")),
    }
}

/// Reverts the most recent file change made by the agent.
pub fn pop_undo() -> Option<Result<String, String>> {
    let (path, prev) = UNDO_STACK.lock().ok()?.pop()?;
    Some(restore(path, prev))
}

/// Reverts the most recent file change for a given path.
pub fn pop_undo_path(target_path: &str) -> Option<Result<String, String>> {
    let (path, prev) = {
        let mut stack = UNDO_STACK.lock().ok()?;
        let idx = stack.iter().rposition(|(p, _)| p == target_path)?;
        stack.remove(idx)
    };
    Some(restore(path, prev))
}

/// Number of file changes recorded so far (a checkpoint for `undo_to`).
pub fn undo_len() -> usize {
    UNDO_STACK.lock().map_or(0, |s| s.len())
}

/// Reverts every file change made after checkpoint `len`, newest first.
pub fn undo_to(len: usize) -> Vec<Result<String, String>> {
    let mut out = Vec::new();
    while undo_len() > len {
        match pop_undo() {
            Some(r) => out.push(r),
            None => break,
        }
    }
    out
}

/// Distinct files the agent changed this session (most recent first), for the UI sidebar.
pub fn modified_files() -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    if let Ok(stack) = UNDO_STACK.lock() {
        for (p, _) in stack.iter().rev() {
            if !out.contains(p) {
                out.push(p.clone());
            }
        }
    }
    out
}

/// Project instructions fed into the system prompt: `SPINDLE.md`, or the `AGENTS.md`
/// convention shared with other coding agents.
pub fn load_spindle_md() -> Option<String> {
    for candidate in ["SPINDLE.md", "spindle.md", ".spindle.md", "AGENTS.md"] {
        if let Ok(text) = fs::read_to_string(candidate) {
            let trimmed = text.trim();
            if !trimmed.is_empty() {
                return Some(trimmed.to_string());
            }
        }
    }
    None
}

// ------------------------------------------------------------------------------ tasks

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TodoState {
    Pending,
    Doing,
    Done,
}

static TODOS: Mutex<Vec<(String, TodoState)>> = Mutex::new(Vec::new());

/// The model's current task list (for the sidebar).
pub fn todos() -> Vec<(String, TodoState)> {
    TODOS.lock().map(|t| t.clone()).unwrap_or_default()
}

pub fn clear_todos() {
    if let Ok(mut t) = TODOS.lock() {
        t.clear();
    }
}

// ------------------------------------------------------------------------ definitions

/// `(name, description, JSON-schema parameters)` for every tool the model is told about.
const TOOL_DEFS: &[(&str, &str, &str)] = &[
    ("read_file", "Read a text file. Returns numbered lines. Use offset/limit for long files.",
     r#"{"type":"object","properties":{"path":{"type":"string"},"offset":{"type":"integer","description":"1-based first line"},"limit":{"type":"integer","description":"max lines (default 250)"}},"required":["path"]}"#),
    ("write_file", "Create or overwrite a file with the given content.",
     r#"{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}"#),
    ("edit_file", "Replace old_string with new_string in a file. old_string must match exactly (including indentation) and be unique unless replace_all is true. Read the file first.",
     r#"{"type":"object","properties":{"path":{"type":"string"},"old_string":{"type":"string"},"new_string":{"type":"string"},"replace_all":{"type":"boolean"}},"required":["path","old_string","new_string"]}"#),
    ("delete_file", "Delete a file (the user is asked to confirm).",
     r#"{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}"#),
    ("list_dir", "List files and folders in a directory.",
     r#"{"type":"object","properties":{"path":{"type":"string"}}}"#),
    ("glob", "Find files by name pattern, e.g. \"**/*.rs\" or \"src/*.py\".",
     r#"{"type":"object","properties":{"pattern":{"type":"string"},"path":{"type":"string"}},"required":["pattern"]}"#),
    ("grep", "Search file contents for a text. Returns path:line: text. Optional include filters file names, e.g. \"*.rs\".",
     r#"{"type":"object","properties":{"pattern":{"type":"string"},"path":{"type":"string"},"include":{"type":"string"}},"required":["pattern"]}"#),
    ("run_command", "Run a shell command (PowerShell on Windows) in the project directory and return its output.",
     r#"{"type":"object","properties":{"command":{"type":"string"},"timeout":{"type":"integer","description":"seconds (default 60, max 600)"}},"required":["command"]}"#),
    ("code_map", "Outline of the project's source code: each file with its functions, types and line numbers, most relevant first. Use it to find where things are before reading files. Optional focus: words to rank files by, e.g. \"tokenizer cache\".",
     r#"{"type":"object","properties":{"focus":{"type":"string"},"path":{"type":"string"}}}"#),
    ("recall", "Show more of a long tool result that was shortened. Its id (like v3) is in the note under the result. Give lines (\"120-180\") or find (a text to show with surrounding lines).",
     r#"{"type":"object","properties":{"id":{"type":"string"},"lines":{"type":"string"},"find":{"type":"string"}},"required":["id"]}"#),
    ("todo", "Keep a short task list for multi-step work, shown to the user. Send the whole list each time, with each task's status: pending, doing or done. Use it for any request with 3 or more steps, and mark a task done as soon as it is.",
     r#"{"type":"object","properties":{"items":{"type":"array","items":{"type":"object","properties":{"task":{"type":"string"},"status":{"type":"string","enum":["pending","doing","done"]}},"required":["task","status"]}}},"required":["items"]}"#),
    ("web_search", "Search the web and return result titles and snippets.",
     r#"{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}"#),
    ("fetch_url", "Download a web page and return its text.",
     r#"{"type":"object","properties":{"url":{"type":"string"}},"required":["url"]}"#),
];

/// The tools section of the system prompt, in Qwen2.5/Qwen3's native function-calling format.
pub fn tools_system_prompt() -> String {
    let mut s = String::from(
        "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n\
         You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n",
    );
    for (name, desc, params) in TOOL_DEFS {
        s.push_str(&format!(
            "{{\"type\": \"function\", \"function\": {{\"name\": \"{name}\", \"description\": {}, \"parameters\": {params}}}}}\n",
            Json::Str(desc.to_string()).to_string_compact()
        ));
    }
    s.push_str(
        "</tools>\n\nFor each function call, return a json object with function name and arguments \
         within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>\n\n\
         How to work:\n\
         - Find code with code_map or grep, then read only the lines you need (offset/limit).\n\
         - Read a file before editing it; copy old_string exactly from what you read.\n\
         - When a call needs the result of another, make one call and wait for its result.\n\
         - For a request with 3 or more steps, keep a todo list and update it as you go.\n\
         - A result that ends with \"saved as v…\" was shortened; use recall for the rest.\n\
         - Never guess or invent a tool's output. After a result, call the next tool or give a short final answer.\n\
         Paths are relative to the project directory.",
    );
    s
}

// ---------------------------------------------------------------------------- parsing

#[derive(Clone, Debug, PartialEq)]
pub struct ToolCall {
    pub name: String,
    pub args: Json,
}

impl ToolCall {
    /// First present string argument among `keys` (aliases tolerate older/other prompts).
    pub fn str_arg(&self, keys: &[&str]) -> Option<&str> {
        keys.iter().find_map(|k| self.args.get(k).and_then(Json::as_str))
    }

    pub fn num_arg(&self, keys: &[&str]) -> Option<usize> {
        keys.iter().find_map(|k| self.args.get(k).and_then(Json::as_usize))
    }

    pub fn bool_arg(&self, keys: &[&str]) -> Option<bool> {
        keys.iter().find_map(|k| self.args.get(k).and_then(Json::as_bool))
    }

    /// Identity used to spot a model looping on the exact same call.
    pub fn signature(&self) -> String {
        format!("{}:{}", self.name, self.args.to_string_compact())
    }
}

fn find_ci(hay: &str, needle: &str, from: usize) -> Option<usize> {
    let h = hay.as_bytes();
    let n = needle.as_bytes();
    if n.is_empty() || from >= h.len() {
        return None;
    }
    (from..=h.len().saturating_sub(n.len())).find(|&i| h[i..i + n.len()].eq_ignore_ascii_case(n))
}

/// Turns one `{...}` payload into a call. Accepts `{"name", "arguments"|"parameters"}` (args
/// as an object or a JSON-encoded string) and the flat `{"name", "path": ...}` form.
fn call_from_json(v: Json) -> Option<ToolCall> {
    let name = v.get("name")?.as_str()?.trim().to_string();
    if name.is_empty() {
        return None;
    }
    let args = match v.get("arguments").or_else(|| v.get("parameters")).or_else(|| v.get("args")) {
        Some(Json::Obj(kv)) => Json::Obj(kv.clone()),
        Some(Json::Str(s)) => json::parse(s).unwrap_or(Json::Obj(Vec::new())),
        _ => match v {
            Json::Obj(kv) => Json::Obj(kv.into_iter().filter(|(k, _)| k != "name").collect()),
            _ => Json::Obj(Vec::new()),
        },
    };
    Some(ToolCall { name, args })
}

/// Qwen3-Coder's own call format, inside `<tool_call>`:
///
/// ```text
/// <function=run_command>
/// <parameter=command>
/// cargo test
/// </parameter>
/// </function>
/// ```
///
/// Values are text; one that is a JSON array, object, number or boolean is read as that, so
/// `todo`'s list and `read_file`'s numbers come through typed.
fn call_from_xml(body: &str) -> Option<ToolCall> {
    let start = body.find("<function=")? + "<function=".len();
    let gt = body[start..].find('>')? + start;
    let name = body[start..gt].trim().trim_matches(['"', '\'']).to_string();
    if name.is_empty() {
        return None;
    }
    let end = body[gt..].find("</function>").map_or(body.len(), |e| gt + e);
    let inner = &body[gt + 1..end];
    let mut args = Vec::new();
    let mut pos = 0;
    while let Some(p) = inner[pos..].find("<parameter=") {
        let ks = pos + p + "<parameter=".len();
        let Some(kg) = inner[ks..].find('>') else { break };
        let key = inner[ks..ks + kg].trim().trim_matches(['"', '\'']).to_string();
        let vs = ks + kg + 1;
        // The value runs to its close tag, or to the next parameter if the model left it open.
        let close = inner[vs..].find("</parameter>");
        let next = inner[vs..].find("<parameter=");
        let ve = match (close, next) {
            (Some(c), Some(n)) => vs + c.min(n),
            (Some(c), None) => vs + c,
            (None, Some(n)) => vs + n,
            (None, None) => inner.len(),
        };
        let mut v = &inner[vs..ve];
        // One line break after the open tag and before the close tag is formatting.
        v = v.strip_prefix("\r\n").or_else(|| v.strip_prefix('\n')).unwrap_or(v);
        v = v.strip_suffix("\r\n").or_else(|| v.strip_suffix('\n')).unwrap_or(v);
        let t = v.trim();
        let typed = if t.starts_with('[') || t.starts_with('{') || t == "true" || t == "false" || (!t.is_empty() && t.parse::<f64>().is_ok()) {
            json::parse(t).ok()
        } else {
            None
        };
        if !key.is_empty() {
            args.push((key, typed.unwrap_or_else(|| Json::Str(v.to_string()))));
        }
        pos = if close.is_some_and(|c| next.map_or(true, |n| c < n)) { ve + "</parameter>".len() } else { ve };
    }
    Some(ToolCall { name, args: Json::Obj(args) })
}

/// Other tag names models use for the same thing. Qwen is trained on `<tool_call>`, but it
/// sometimes writes `<function_call>` (or `<tool_use>`) instead.
const CALL_TAG_ALIASES: &[&str] = &["function_call", "tool_use", "functioncall"];

/// Rewrites alias call tags (any case) to `<tool_call>` / `</tool_call>`.
pub fn normalize_call_tags(text: &str) -> std::borrow::Cow<'_, str> {
    let aliased = normalize_alias_tags(text);
    // Qwen3-Coder sometimes writes its `<function=…>` block without the `<tool_call>` wrapper.
    if aliased.contains("<function=") && !aliased.contains("<tool_call") {
        return std::borrow::Cow::Owned(aliased.replace("<function=", "<tool_call><function=").replace("</function>", "</function></tool_call>"));
    }
    aliased
}

fn normalize_alias_tags(text: &str) -> std::borrow::Cow<'_, str> {
    let lower = text.to_ascii_lowercase();
    if !CALL_TAG_ALIASES.iter().any(|a| lower.contains(&format!("<{a}")) || lower.contains(&format!("</{a}"))) {
        return std::borrow::Cow::Borrowed(text);
    }
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    'scan: while i < text.len() {
        if text.as_bytes()[i] == b'<' {
            for a in CALL_TAG_ALIASES {
                for (close, pre) in [(false, "<"), (true, "</")] {
                    let tag = format!("{pre}{a}>");
                    if lower[i..].starts_with(&tag) {
                        out.push_str(if close { "</tool_call>" } else { "<tool_call>" });
                        i += tag.len();
                        continue 'scan;
                    }
                }
            }
        }
        let ch = text[i..].chars().next().unwrap();
        out.push(ch);
        i += ch.len_utf8();
    }
    std::borrow::Cow::Owned(out)
}

/// Extracts every tool call from a model reply, plus the reply with the call blocks removed
/// (what the user should see).
pub fn parse_tool_calls(text: &str) -> (Vec<ToolCall>, String) {
    let normalized = normalize_call_tags(text);
    let text: &str = &normalized;
    let mut calls = Vec::new();
    let mut visible = String::new();
    let mut pos = 0;
    while let Some(open) = find_ci(text, "<tool_call", pos) {
        visible.push_str(&text[pos..open]);
        let Some(gt) = text[open..].find('>') else {
            pos = text.len();
            break;
        };
        let body_start = open + gt + 1;
        let close = find_ci(text, "</tool_call", body_start);
        let body_end = close.unwrap_or(text.len());
        let body = &text[body_start..body_end];
        if body.contains("<function=") {
            if let Some(c) = call_from_xml(body) {
                calls.push(c);
            }
        } else if let Some(brace) = body.find('{') {
            // The JSON object may be wrapped in a ``` fence; parse from its first brace.
            if let Ok((v, _)) = json::parse_prefix(&body[brace..]) {
                if let Some(c) = call_from_json(v) {
                    calls.push(c);
                }
            }
        }
        pos = match close {
            Some(c) => text[c..].find('>').map_or(text.len(), |g| c + g + 1),
            None => text.len(),
        };
    }
    visible.push_str(&text[pos.min(text.len())..]);

    // No tags at all, but the reply is (or ends with) a bare call object: accept it if it
    // names one of our tools, so a model that forgets the tags still works.
    if calls.is_empty() {
        if let Some(brace) = text.find("{\"name\"").or_else(|| text.find("{'name'")) {
            if let Ok((v, used)) = json::parse_prefix(&text[brace..]) {
                if let Some(c) = call_from_json(v) {
                    if TOOL_DEFS.iter().any(|(n, _, _)| *n == c.name) || is_alias(&c.name) {
                        visible = format!("{}{}", &text[..brace], &text[brace + used..]);
                        calls.push(c);
                    }
                }
            }
        }
    }
    (calls, visible.trim().to_string())
}

/// Removes `<tool_call>…</tool_call>` from streamed text as it arrives. Small models often
/// spell the tag out as several plain tokens (`<`, `tool`, `_call`, `>`), so this works on
/// the accumulated text and holds back a trailing partial tag instead of matching per token.
#[derive(Default)]
pub struct ToolMarkupFilter {
    buf: String,
    /// The close tag of the block being hidden, while inside one.
    close: Option<&'static str>,
}

/// Longest trailing partial opening call tag (`<tool_c`, `<function_ca`, ...) in `s`.
pub fn partial_call_tag(s: &str) -> usize {
    let mut n = partial_tag_suffix(s, "<tool_call>").max(partial_tag_suffix(s, "<function="));
    for a in CALL_TAG_ALIASES {
        n = n.max(partial_tag_suffix(s, &format!("<{a}>")));
    }
    n
}

/// Length of the longest suffix of `s` that is a proper prefix of `tag`.
pub fn partial_tag_suffix(s: &str, tag: &str) -> usize {
    (1..tag.len().min(s.len() + 1))
        .rev()
        .find(|&n| s.is_char_boundary(s.len() - n) && tag.as_bytes().starts_with(&s.as_bytes()[s.len() - n..]))
        .unwrap_or(0)
}

impl ToolMarkupFilter {
    /// Feeds a piece; returns the text that is now safe to display.
    pub fn push(&mut self, piece: &str) -> String {
        self.buf.push_str(piece);
        if let std::borrow::Cow::Owned(n) = normalize_alias_tags(&self.buf) {
            self.buf = n;
        }
        let mut out = String::new();
        loop {
            if let Some(close) = self.close {
                match self.buf.find(close) {
                    Some(i) => {
                        self.buf.drain(..i + close.len());
                        self.close = None;
                    }
                    None => {
                        let keep = partial_tag_suffix(&self.buf, close);
                        let cut = self.buf.len() - keep;
                        self.buf.drain(..cut);
                        return out;
                    }
                }
            } else {
                // A block opens with `<tool_call>`, or with Qwen3-Coder's bare `<function=`.
                let tc = self.buf.find("<tool_call>");
                let fc = self.buf.find("<function=");
                let open = match (tc, fc) {
                    (Some(t), Some(f)) if f < t => Some((f, "<function=".len(), "</function>")),
                    (Some(t), _) => Some((t, "<tool_call>".len(), "</tool_call>")),
                    (None, Some(f)) => Some((f, "<function=".len(), "</function>")),
                    (None, None) => None,
                };
                match open {
                    Some((i, len, close)) => {
                        out.push_str(&self.buf[..i]);
                        self.buf.drain(..i + len);
                        self.close = Some(close);
                    }
                    None => {
                        let keep = partial_call_tag(&self.buf);
                        let cut = self.buf.len() - keep;
                        out.push_str(&self.buf[..cut]);
                        self.buf.drain(..cut);
                        return out;
                    }
                }
            }
        }
    }
}

fn is_alias(name: &str) -> bool {
    matches!(name, "search_web" | "search_symbols" | "read" | "write" | "edit" | "bash" | "shell" | "ls")
}

/// First tool call in `text` (kept for callers that only need one).
pub fn parse_tool_call(text: &str) -> Option<ToolCall> {
    parse_tool_calls(text).0.into_iter().next()
}

// -------------------------------------------------------------------------- execution

/// What the UI needs to show for a tool call, and what goes back to the model.
#[derive(Clone, Debug)]
pub struct ToolOutcome {
    pub ok: bool,
    /// Row glyph: `▸` read/list, `✎` write/edit/delete, `⌕` search, `❯` shell, `◎` web, `⇣` fetch.
    pub icon: &'static str,
    /// One-line summary, e.g. `Read src/main.rs:1-250`.
    pub title: String,
    /// Full text returned to the model inside `<tool_response>`.
    pub output: String,
    /// Optional preview for the UI: a diff (`+`/`-` lines) or the first lines of output.
    pub preview: Option<String>,
}

impl ToolOutcome {
    fn ok(icon: &'static str, title: String, output: String) -> Self {
        Self { ok: true, icon, title, output, preview: None }
    }
    fn err(icon: &'static str, title: String, error: String) -> Self {
        Self { ok: false, icon, title, output: format!("Error: {error}"), preview: None }
    }
}

/// Per-call execution context.
pub struct ToolCtx<'a> {
    /// Asks the user to approve an action; `false` aborts it.
    pub confirm: &'a mut dyn FnMut(&str) -> bool,
    /// Budget for text returned to the model (keeps one tool result from flooding the context).
    pub max_chars: usize,
}

/// Keeps the head and tail of `s` within `max` bytes (char-boundary safe).
pub fn truncate_middle(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_string();
    }
    let head = max * 2 / 3;
    let tail = max - head;
    let mut h = head;
    while !s.is_char_boundary(h) {
        h -= 1;
    }
    let mut t = s.len() - tail;
    while !s.is_char_boundary(t) {
        t += 1;
    }
    format!("{}\n\n... [{} bytes omitted] ...\n\n{}", &s[..h], t - h, &s[t..])
}

/// Guard reasons are lowercase phrases; permission prompts start with a capital.
fn capitalize(s: &str) -> String {
    let mut c = s.chars();
    c.next().map_or(String::new(), |f| f.to_uppercase().chain(c).collect())
}

/// Is `p` inside the current working directory? (Writes elsewhere need confirmation.)
fn inside_workspace(p: &Path) -> bool {
    let Ok(cwd) = std::env::current_dir() else { return true };
    let abs = if p.is_absolute() { p.to_path_buf() } else { cwd.join(p) };
    // Resolve `..` lexically (the target may not exist yet).
    let mut norm = PathBuf::new();
    for c in abs.components() {
        match c {
            std::path::Component::ParentDir => {
                norm.pop();
            }
            std::path::Component::CurDir => {}
            other => norm.push(other),
        }
    }
    let lower = |p: &Path| p.to_string_lossy().to_lowercase().replace('/', "\\");
    lower(&norm).starts_with(&lower(&cwd))
}

fn check_write(path: &str, ctx: &mut ToolCtx) -> Result<(), String> {
    match guard::check_path(path, true) {
        GuardDecision::Block(r) => return Err(format!("refused: {r}")),
        GuardDecision::NeedConfirmation(r) => {
            if !(ctx.confirm)(&format!("{}: {path}", capitalize(&r))) {
                return Err("the user declined this action".into());
            }
        }
        GuardDecision::Allow => {}
    }
    if !inside_workspace(Path::new(path)) && !(ctx.confirm)(&format!("Write outside the project directory: {path}")) {
        return Err("the user declined writing outside the project directory".into());
    }
    Ok(())
}

/// Runs one tool call.
pub fn execute(call: &ToolCall, ctx: &mut ToolCtx) -> ToolOutcome {
    let path = call.str_arg(&["path", "file_path", "file", "filename"]).unwrap_or("").trim().to_string();
    match call.name.as_str() {
        "read_file" | "read" => {
            let offset = call.num_arg(&["offset", "start_line", "line"]);
            let limit = call.num_arg(&["limit", "lines"]);
            read_file(&path, offset, limit, ctx.max_chars)
        }
        "write_file" | "write" => {
            let content = call.str_arg(&["content", "text", "contents"]).unwrap_or("");
            write_file(&path, content, ctx)
        }
        "edit_file" | "edit" => {
            let old = call.str_arg(&["old_string", "target", "old", "search"]).unwrap_or("");
            let new = call.str_arg(&["new_string", "replacement", "new", "replace"]).unwrap_or("");
            let all = call.bool_arg(&["replace_all"]).unwrap_or(false);
            edit_file(&path, old, new, all, ctx)
        }
        "delete_file" => delete_file(&path, ctx),
        "list_dir" | "ls" => list_dir(if path.is_empty() { "." } else { &path }, ctx.max_chars),
        "glob" => glob_tool(call.str_arg(&["pattern", "glob"]).unwrap_or(""), &path, ctx.max_chars),
        "grep" => grep_tool(
            call.str_arg(&["pattern", "query", "text"]).unwrap_or(""),
            &path,
            call.str_arg(&["include", "glob"]),
            ctx.max_chars,
        ),
        "search_symbols" => {
            let pat = call.str_arg(&["pattern", "query"]).unwrap_or("");
            let title = format!("Symbols \"{pat}\"");
            match tool_search_symbols(pat, Some(if path.is_empty() { "." } else { &path }), call.str_arg(&["symbol_type"])) {
                Ok(o) => ToolOutcome::ok("⌕", title, truncate_middle(&o, ctx.max_chars)),
                Err(e) => ToolOutcome::err("⌕", title, e),
            }
        }
        "run_command" | "bash" | "shell" => {
            let cmd = call.str_arg(&["command", "cmd"]).unwrap_or("");
            let timeout = call.num_arg(&["timeout"]).unwrap_or(60).clamp(1, 600);
            run_command(cmd, timeout, ctx)
        }
        "web_search" | "search_web" => {
            let q = call.str_arg(&["query", "q"]).unwrap_or("");
            let title = format!("Search \"{q}\"");
            match tool_search_web(q) {
                Ok(o) => ToolOutcome::ok("◎", title.clone(), vault::fit(&title, &o, ctx.max_chars, vault::Kind::Text)),
                Err(e) => ToolOutcome::err("◎", title, e),
            }
        }
        "fetch_url" => {
            let url = call.str_arg(&["url"]).unwrap_or("");
            let title = format!("Fetch {url}");
            match fetch_url(url) {
                Ok(o) => ToolOutcome::ok("⇣", title.clone(), vault::fit(&title, &o, ctx.max_chars, vault::Kind::Text)),
                Err(e) => ToolOutcome::err("⇣", title, e),
            }
        }
        "code_map" | "repo_map" | "outline" => {
            let focus = call.str_arg(&["focus", "query", "pattern"]);
            let title = match focus {
                Some(f) if !f.trim().is_empty() => format!("Code map \"{}\"", f.trim()),
                _ => "Code map".to_string(),
            };
            ToolOutcome::ok("⌕", title, code_map(if path.is_empty() { "." } else { &path }, focus, ctx.max_chars))
        }
        "todo" | "todo_write" | "tasks" => {
            let items = match call.args.get("items").or_else(|| call.args.get("todos")).or_else(|| call.args.get("tasks")) {
                Some(Json::Arr(a)) => a.clone(),
                _ => return ToolOutcome::err("☐", "Tasks".into(), "expected 'items': a list of {\"task\", \"status\"}".into()),
            };
            let list: Vec<(String, TodoState)> = items
                .iter()
                .filter_map(|it| {
                    let task = it.get("task").or_else(|| it.get("content")).or_else(|| it.get("text")).and_then(Json::as_str)?.trim().to_string();
                    let st = match it.get("status").and_then(Json::as_str).unwrap_or("pending").to_ascii_lowercase().as_str() {
                        "done" | "completed" | "complete" => TodoState::Done,
                        "doing" | "in_progress" | "in progress" | "active" => TodoState::Doing,
                        _ => TodoState::Pending,
                    };
                    (!task.is_empty()).then_some((task, st))
                })
                .take(20)
                .collect();
            let (d, g) = (list.iter().filter(|t| t.1 == TodoState::Done).count(), list.iter().filter(|t| t.1 == TodoState::Doing).count());
            let n = list.len();
            if let Ok(mut t) = TODOS.lock() {
                *t = list;
            }
            ToolOutcome::ok("☐", format!("Tasks {d}/{n} done"), format!("Task list updated: {d} done, {g} in progress, {} pending.", n - d - g))
        }
        "recall" => {
            let id = call.str_arg(&["id", "ref"]).unwrap_or("");
            let lines = call.str_arg(&["lines", "range"]);
            let find = call.str_arg(&["find", "query", "pattern"]);
            match vault::recall(id, lines, find, ctx.max_chars) {
                Ok((t, body)) => ToolOutcome::ok("▸", format!("Recall {id} ({t})"), body),
                Err(e) => ToolOutcome::err("▸", format!("Recall {id}"), e),
            }
        }
        other => ToolOutcome::err(
            "▪",
            other.to_string(),
            format!(
                "unknown tool '{other}'. Available: {}",
                TOOL_DEFS.iter().map(|t| t.0).collect::<Vec<_>>().join(", ")
            ),
        ),
    }
}

fn read_file(path: &str, offset: Option<usize>, limit: Option<usize>, max_chars: usize) -> ToolOutcome {
    let first = offset.unwrap_or(1).max(1);
    let title = |end: usize| {
        if offset.is_some() || limit.is_some() {
            format!("Read {path}:{first}-{end}")
        } else {
            format!("Read {path}")
        }
    };
    if path.is_empty() {
        return ToolOutcome::err("▸", "Read".into(), "missing 'path'".into());
    }
    if let GuardDecision::Block(r) = guard::check_path(path, false) {
        return ToolOutcome::err("▸", title(first), format!("refused: {r}"));
    }
    let p = Path::new(path);
    if p.is_dir() {
        return ToolOutcome::err("▸", title(first), format!("'{path}' is a directory; use list_dir"));
    }
    let bytes = match fs::read(p) {
        Ok(b) => b,
        Err(e) => return ToolOutcome::err("▸", title(first), format!("cannot read '{path}': {e}")),
    };
    if bytes.iter().take(8192).any(|&b| b == 0) {
        return ToolOutcome::err("▸", title(first), format!("'{path}' looks like a binary file ({} bytes)", bytes.len()));
    }
    let raw = String::from_utf8_lossy(&bytes);
    let lines: Vec<&str> = raw.lines().collect();
    let total = lines.len();
    let start = (first - 1).min(total);
    let count = limit.unwrap_or(250).min(total - start);
    let mut out = String::new();
    // A long file opened from the top: lead with its outline, so the next read can jump
    // straight to the right place instead of paging through the whole file.
    if offset.is_none() && limit.is_none() && total > count {
        let ext = p.extension().and_then(|e| e.to_str()).unwrap_or("");
        let outline = file_outline(&lines, ext, max_chars / 3);
        if !outline.is_empty() {
            out.push_str(&format!("Outline of {path} ({total} lines):\n{outline}\nFirst lines:\n"));
        }
    }
    let mut shown = 0;
    for (i, line) in lines[start..start + count].iter().enumerate() {
        let l: String = if line.chars().count() > 400 { line.chars().take(400).chain("…".chars()).collect() } else { line.to_string() };
        let row = format!("{:>5}\t{}\n", start + i + 1, l);
        if out.len() + row.len() > max_chars {
            break;
        }
        out.push_str(&row);
        shown += 1;
    }
    let end = start + shown;
    if end < total {
        out.push_str(&format!("\n(lines {}-{end} of {total}; call read_file with offset={} to continue)\n", start + 1, end + 1));
    } else if total == 0 {
        out.push_str("(empty file)\n");
    }
    ToolOutcome::ok("▸", title(end), out)
}

fn write_file(path: &str, content: &str, ctx: &mut ToolCtx) -> ToolOutcome {
    let title = format!("Write {path}");
    if path.is_empty() {
        return ToolOutcome::err("✎", "Write".into(), "missing 'path'".into());
    }
    if let Err(e) = check_write(path, ctx) {
        return ToolOutcome::err("✎", title, e);
    }
    let p = Path::new(path);
    if let Some(parent) = p.parent() {
        if !parent.as_os_str().is_empty() && !parent.exists() {
            if let Err(e) = fs::create_dir_all(parent) {
                return ToolOutcome::err("✎", title, format!("cannot create '{}': {e}", parent.display()));
            }
        }
    }
    let existed = p.exists();
    push_undo_snapshot(path);
    match fs::write(p, content) {
        Ok(()) => {
            let n = content.lines().count();
            let mut o = ToolOutcome::ok(
                "✎",
                title,
                format!("{} '{path}' ({n} lines, {} bytes).", if existed { "Overwrote" } else { "Created" }, content.len()),
            );
            o.preview = Some(content.lines().take(12).map(|l| format!("+{l}")).collect::<Vec<_>>().join("\n"));
            o
        }
        Err(e) => ToolOutcome::err("✎", title, format!("cannot write '{path}': {e}")),
    }
}

fn edit_file(path: &str, old: &str, new: &str, replace_all: bool, ctx: &mut ToolCtx) -> ToolOutcome {
    let title = format!("Edit {path}");
    if path.is_empty() {
        return ToolOutcome::err("✎", "Edit".into(), "missing 'path'".into());
    }
    if old.is_empty() {
        return ToolOutcome::err("✎", title, "old_string is empty; use write_file to create a file".into());
    }
    if old == new {
        return ToolOutcome::err("✎", title, "old_string and new_string are identical".into());
    }
    if let Err(e) = check_write(path, ctx) {
        return ToolOutcome::err("✎", title, e);
    }
    let src = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => return ToolOutcome::err("✎", title, format!("cannot read '{path}': {e}")),
    };
    // Windows files use CRLF, but models write `\n`: match in the file's own line endings.
    let crlf = src.contains("\r\n");
    let (old_f, new_f) = if crlf {
        (old.replace("\r\n", "\n").replace('\n', "\r\n"), new.replace("\r\n", "\n").replace('\n', "\r\n"))
    } else {
        (old.to_string(), new.to_string())
    };
    let n = src.matches(&old_f).count();
    if n == 0 {
        // Help the model recover: point at the closest line to its first old_string line.
        let first = old.lines().find(|l| !l.trim().is_empty()).unwrap_or("").trim();
        let hint = src
            .lines()
            .enumerate()
            .find(|(_, l)| !first.is_empty() && l.contains(first))
            .map(|(i, l)| format!(" A similar line is at {}: {:?} - check whitespace/indentation and read the file again.", i + 1, l.trim_end()))
            .unwrap_or_else(|| " Read the file again and copy old_string exactly.".into());
        return ToolOutcome::err("✎", title, format!("old_string not found in '{path}'.{hint}"));
    }
    if n > 1 && !replace_all {
        return ToolOutcome::err(
            "✎",
            title,
            format!("old_string occurs {n} times in '{path}'; add surrounding lines to make it unique, or set replace_all=true"),
        );
    }
    let updated = if replace_all { src.replace(&old_f, &new_f) } else { src.replacen(&old_f, &new_f, 1) };
    push_undo_snapshot(path);
    if let Err(e) = fs::write(path, &updated) {
        return ToolOutcome::err("✎", title, format!("cannot write '{path}': {e}"));
    }
    let line = src[..src.find(&old_f).unwrap_or(0)].matches('\n').count() + 1;
    let mut o = ToolOutcome::ok("✎", title, format!("Edited '{path}' ({} replacement(s), starting at line {line}).", if replace_all { n } else { 1 }));
    let mut diff: Vec<String> = old.lines().map(|l| format!("-{l}")).collect();
    diff.extend(new.lines().map(|l| format!("+{l}")));
    o.preview = Some(diff.join("\n"));
    o
}

fn delete_file(path: &str, ctx: &mut ToolCtx) -> ToolOutcome {
    let title = format!("Delete {path}");
    let p = Path::new(path);
    if path.is_empty() || !p.is_file() {
        return ToolOutcome::err("✎", title, format!("'{path}' is not an existing file"));
    }
    if let Err(e) = check_write(path, ctx) {
        return ToolOutcome::err("✎", title, e);
    }
    if !(ctx.confirm)(&format!("Delete file {path}")) {
        return ToolOutcome::err("✎", title, "the user declined deleting this file".into());
    }
    push_undo_snapshot(path);
    match fs::remove_file(p) {
        Ok(()) => ToolOutcome::ok("✎", title, format!("Deleted '{path}' (the user can /undo it).")),
        Err(e) => ToolOutcome::err("✎", title, format!("cannot delete '{path}': {e}")),
    }
}

fn list_dir(path: &str, max_chars: usize) -> ToolOutcome {
    let title = format!("List {path}");
    let entries = match fs::read_dir(path) {
        Ok(e) => e,
        Err(e) => return ToolOutcome::err("▸", title, format!("cannot list '{path}': {e}")),
    };
    let (mut dirs, mut files) = (Vec::new(), Vec::new());
    for e in entries.flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        match e.metadata() {
            Ok(m) if m.is_dir() => dirs.push(format!("{name}/")),
            Ok(m) => files.push(format!("{name}  ({})", human_size(m.len()))),
            Err(_) => files.push(name),
        }
    }
    dirs.sort();
    files.sort();
    let mut out = String::new();
    for l in dirs.iter().chain(files.iter()) {
        out.push_str(l);
        out.push('\n');
    }
    if out.is_empty() {
        out.push_str("(empty directory)\n");
    }
    ToolOutcome::ok("▸", title, truncate_middle(&out, max_chars))
}

fn human_size(n: u64) -> String {
    if n >= 1 << 30 {
        format!("{:.1} GiB", n as f64 / (1u64 << 30) as f64)
    } else if n >= 1 << 20 {
        format!("{:.1} MiB", n as f64 / (1u64 << 20) as f64)
    } else if n >= 1 << 10 {
        format!("{:.1} KiB", n as f64 / 1024.0)
    } else {
        format!("{n} B")
    }
}

// ------------------------------------------------------------------- search helpers

const SKIP_DIRS: &[&str] = &[".git", "target", "node_modules", ".venv", "venv", "__pycache__", "dist", "build", ".next", ".cargo", "vendor", ".idea", ".vscode"];

/// Recursively collects files under `root` (skipping build/vendor dirs), at most `cap`.
fn walk_files(root: &Path, cap: usize) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    if root.is_file() {
        return vec![root.to_path_buf()];
    }
    while let Some(dir) = stack.pop() {
        let Ok(rd) = fs::read_dir(&dir) else { continue };
        let mut entries: Vec<_> = rd.flatten().collect();
        entries.sort_by_key(|e| e.file_name());
        for e in entries {
            let p = e.path();
            let name = e.file_name().to_string_lossy().into_owned();
            match e.file_type() {
                Ok(t) if t.is_dir() => {
                    if !SKIP_DIRS.contains(&name.as_str()) {
                        stack.push(p);
                    }
                }
                Ok(t) if t.is_file() => {
                    out.push(p);
                    if out.len() >= cap {
                        return out;
                    }
                }
                _ => {}
            }
        }
    }
    out
}

fn display_path(p: &Path) -> String {
    let s = p.to_string_lossy().replace('\\', "/");
    s.strip_prefix("./").unwrap_or(&s).to_string()
}

/// Glob match on `/`-separated paths: `*` (within a segment), `?`, and `**` (any depth).
pub fn glob_match(pattern: &str, path: &str) -> bool {
    fn seg(p: &[u8], s: &[u8]) -> bool {
        match (p.first(), s.first()) {
            (None, None) => true,
            (Some(b'*'), _) => seg(&p[1..], s) || (!s.is_empty() && seg(p, &s[1..])),
            (Some(b'?'), Some(_)) => seg(&p[1..], &s[1..]),
            (Some(a), Some(b)) if a.eq_ignore_ascii_case(b) => seg(&p[1..], &s[1..]),
            _ => false,
        }
    }
    fn parts(p: &[&str], s: &[&str]) -> bool {
        match (p.first(), s.first()) {
            (None, None) => true,
            (Some(&"**"), _) => parts(&p[1..], s) || (!s.is_empty() && parts(p, &s[1..])),
            (Some(a), Some(b)) if seg(a.as_bytes(), b.as_bytes()) => parts(&p[1..], &s[1..]),
            _ => false,
        }
    }
    let pat = pattern.replace('\\', "/");
    let path = path.replace('\\', "/");
    let s: Vec<&str> = path.split('/').filter(|x| !x.is_empty() && *x != ".").collect();
    if !pat.contains('/') {
        // A bare pattern ("*.rs") matches the file name at any depth.
        return s.last().is_some_and(|name| seg(pat.as_bytes(), name.as_bytes()));
    }
    let p: Vec<&str> = pat.split('/').filter(|x| !x.is_empty() && *x != ".").collect();
    parts(&p, &s)
}

fn glob_tool(pattern: &str, path: &str, max_chars: usize) -> ToolOutcome {
    let title = format!("Glob \"{pattern}\"");
    if pattern.trim().is_empty() {
        return ToolOutcome::err("⌕", title, "missing 'pattern'".into());
    }
    let root = Path::new(if path.is_empty() { "." } else { path });
    let mut hits: Vec<(std::time::SystemTime, String)> = walk_files(root, 50_000)
        .into_iter()
        .filter_map(|p| {
            let rel = p.strip_prefix(root).unwrap_or(&p);
            if glob_match(pattern, &rel.to_string_lossy()) {
                let t = fs::metadata(&p).and_then(|m| m.modified()).unwrap_or(std::time::UNIX_EPOCH);
                Some((t, display_path(&p)))
            } else {
                None
            }
        })
        .collect();
    // Most recently modified first.
    hits.sort_by(|a, b| b.0.cmp(&a.0));
    let n = hits.len();
    let mut out: String = hits.iter().take(200).map(|(_, p)| format!("{p}\n")).collect();
    if n > 200 {
        out.push_str(&format!("({n} files; showing 200, narrow the pattern)\n"));
    }
    if n == 0 {
        out = format!("No files match '{pattern}'.\n");
    }
    ToolOutcome::ok("⌕", format!("{title} ({n} {})", if n == 1 { "file" } else { "files" }), truncate_middle(&out, max_chars))
}

fn grep_tool(pattern: &str, path: &str, include: Option<&str>, max_chars: usize) -> ToolOutcome {
    let title = format!("Grep \"{pattern}\"");
    if pattern.is_empty() {
        return ToolOutcome::err("⌕", title, "missing 'pattern'".into());
    }
    // Smart case: all-lowercase patterns match case-insensitively.
    let ci = !pattern.chars().any(char::is_uppercase);
    let needle = if ci { pattern.to_lowercase() } else { pattern.to_string() };
    let root = Path::new(if path.is_empty() { "." } else { path });
    let mut out = String::new();
    let mut n = 0usize;
    'files: for p in walk_files(root, 20_000) {
        if let Some(inc) = include {
            let name = p.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
            if !glob_match(inc, &name) {
                continue;
            }
        }
        let Ok(meta) = fs::metadata(&p) else { continue };
        if meta.len() > 2 << 20 {
            continue;
        }
        let Ok(bytes) = fs::read(&p) else { continue };
        if bytes.iter().take(4096).any(|&b| b == 0) {
            continue;
        }
        let text = String::from_utf8_lossy(&bytes);
        for (i, line) in text.lines().enumerate() {
            let hit = if ci { line.to_lowercase().contains(&needle) } else { line.contains(&needle) };
            if hit {
                n += 1;
                let l: String = line.trim().chars().take(240).collect();
                out.push_str(&format!("{}:{}: {l}\n", display_path(&p), i + 1));
                if n >= 150 || out.len() > max_chars {
                    out.push_str("(more matches omitted; narrow the pattern or path)\n");
                    break 'files;
                }
            }
        }
    }
    if n == 0 {
        out = format!("No matches for '{pattern}'.\n");
    }
    let body = vault::fit(&title, &out, max_chars, vault::Kind::Text);
    ToolOutcome::ok("⌕", format!("{title} ({n} {})", if n == 1 { "match" } else { "matches" }), body)
}

// ------------------------------------------------------------------- shell and web

fn run_command(cmd: &str, timeout_s: usize, ctx: &mut ToolCtx) -> ToolOutcome {
    let title = cmd.to_string();
    if cmd.trim().is_empty() {
        return ToolOutcome::err("❯", title, "missing 'command'".into());
    }
    match guard::check_command(cmd) {
        GuardDecision::Block(r) => return ToolOutcome::err("❯", title, format!("refused: {r}")),
        GuardDecision::NeedConfirmation(r) => {
            if !(ctx.confirm)(&format!("{}: {cmd}", capitalize(&r))) {
                return ToolOutcome::err("❯", title, "the user declined running this command".into());
            }
        }
        GuardDecision::Allow => {}
    }
    match guard::run_sandboxed(cmd, Duration::from_secs(timeout_s as u64)) {
        Ok((out, code)) => {
            let body = vault::fit(cmd, out.trim_end(), ctx.max_chars.saturating_sub(64).max(512), vault::Kind::Log);
            let mut o = ToolOutcome {
                ok: code == 0,
                icon: "❯",
                title,
                output: format!("Exit code: {code}\n{body}"),
                preview: None,
            };
            let lines: Vec<&str> = out.trim_end().lines().collect();
            let mut prev = lines.iter().take(10).copied().collect::<Vec<_>>().join("\n");
            if lines.len() > 10 {
                prev.push_str(&format!("\n… {} more lines", lines.len() - 10));
            }
            if code != 0 {
                prev.push_str(&format!("\n(exit code {code})"));
            }
            o.preview = Some(prev);
            o
        }
        Err(e) => ToolOutcome::err("❯", title, format!("failed to start: {e}")),
    }
}

fn url_encode(q: &str) -> String {
    let mut out = String::new();
    for b in q.bytes() {
        if b.is_ascii_alphanumeric() || b"-_.~".contains(&b) {
            out.push(b as char);
        } else if b == b' ' {
            out.push('+');
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

fn curl(url: &str, secs: u32) -> Result<String, String> {
    let out = std::process::Command::new(if cfg!(windows) { "curl.exe" } else { "curl" })
        .args(["-s", "-L", "--max-time", &secs.to_string(), "-A", "Mozilla/5.0 (Windows NT 10.0; Win64; x64)", url])
        .output()
        .map_err(|e| format!("cannot run curl: {e}"))?;
    if !out.status.success() {
        return Err(format!("curl failed (exit {:?})", out.status.code()));
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

/// DuckDuckGo HTML search via the system `curl` (no extra dependencies).
pub fn tool_search_web(query: &str) -> Result<String, String> {
    let q = query.trim();
    if q.is_empty() {
        return Err("missing 'query'".into());
    }
    let html = curl(&format!("https://html.duckduckgo.com/html/?q={}", url_encode(q)), 10)?;
    let mut results = Vec::new();
    for part in html.split("class=\"result__a\"").skip(1).take(6) {
        let title = part.find('>').and_then(|s| part[s..].find("</a>").map(|e| strip_html(&part[s + 1..s + e])));
        let href = part.find("href=\"").and_then(|s| part[s + 6..].find('"').map(|e| part[s + 6..s + 6 + e].to_string()));
        let snippet = part
            .find("class=\"result__snippet\"")
            .and_then(|i| {
                let sp = &part[i..];
                sp.find('>').and_then(|s| sp[s..].find("</a>").map(|e| strip_html(&sp[s + 1..s + e])))
            })
            .unwrap_or_default();
        if let Some(t) = title.filter(|t| !t.trim().is_empty()) {
            let link = href.map(|h| decode_ddg_link(&h)).unwrap_or_default();
            results.push(format!("- {}\n  {}\n  {}", t.trim(), link, snippet.trim()));
        }
    }
    if results.is_empty() {
        Ok(format!("No results parsed for '{q}'."))
    } else {
        Ok(format!("Results for '{q}':\n{}", results.join("\n")))
    }
}

/// DuckDuckGo wraps result links as `//duckduckgo.com/l/?uddg=<encoded url>&...`.
fn decode_ddg_link(h: &str) -> String {
    let Some(i) = h.find("uddg=") else { return h.to_string() };
    let enc = &h[i + 5..];
    let enc = &enc[..enc.find('&').unwrap_or(enc.len())];
    let mut out = Vec::new();
    let b = enc.as_bytes();
    let mut j = 0;
    while j < b.len() {
        if b[j] == b'%' && j + 2 < b.len() {
            if let Ok(v) = u8::from_str_radix(&enc[j + 1..j + 3], 16) {
                out.push(v);
                j += 3;
                continue;
            }
        }
        out.push(b[j]);
        j += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn fetch_url(url: &str) -> Result<String, String> {
    let u = url.trim();
    if !(u.starts_with("http://") || u.starts_with("https://")) {
        return Err("url must start with http:// or https://".into());
    }
    let html = curl(u, 15)?;
    let text = html_to_text(&html);
    if text.trim().is_empty() {
        return Err("the page had no readable text".into());
    }
    Ok(text)
}

/// Visible text of an HTML page: drops script/style/head, tags and blank runs.
fn html_to_text(html: &str) -> String {
    let lower = html.to_lowercase();
    let mut keep = String::with_capacity(html.len() / 2);
    let mut i = 0;
    while i < html.len() {
        let mut skipped = false;
        for tag in ["script", "style", "head", "noscript", "svg"] {
            if lower[i..].starts_with(&format!("<{tag}")) {
                if let Some(end) = lower[i..].find(&format!("</{tag}>")) {
                    i += end + tag.len() + 3;
                    skipped = true;
                    break;
                }
            }
        }
        if skipped {
            continue;
        }
        let ch = html[i..].chars().next().unwrap();
        if matches!(lower[i..].get(..4), Some("<br>") | Some("<br/") | Some("</p>") | Some("<li>")) || lower[i..].starts_with("</div") || lower[i..].starts_with("</h") {
            keep.push('\n');
        }
        keep.push(ch);
        i += ch.len_utf8();
    }
    let text = strip_html(&keep);
    let mut out = String::new();
    let mut blank = 0;
    for line in text.lines() {
        let l = line.split_whitespace().collect::<Vec<_>>().join(" ");
        if l.is_empty() {
            blank += 1;
            if blank > 1 {
                continue;
            }
        } else {
            blank = 0;
        }
        out.push_str(&l);
        out.push('\n');
    }
    out
}

fn strip_html(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut in_tag = false;
    for ch in input.chars() {
        match ch {
            '<' => in_tag = true,
            '>' => in_tag = false,
            c if !in_tag => out.push(c),
            _ => {}
        }
    }
    out.replace("&amp;", "&")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&#x27;", "'")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&nbsp;", " ")
}

// ------------------------------------------------------------------- symbol search

/// Scans source files for declarations (fn/struct/class/...) whose line contains `pattern`.
const SOURCE_EXTS: &[&str] = &["rs", "py", "js", "ts", "jsx", "tsx", "go", "c", "cpp", "h", "hpp"];

/// Top-level declarations of one file as `line: text` rows, within `max` bytes.
fn file_outline(lines: &[&str], ext: &str, max: usize) -> String {
    if !SOURCE_EXTS.contains(&ext) {
        return String::new();
    }
    let mut rows = Vec::new();
    for (i, line) in lines.iter().enumerate() {
        let indent = line.len() - line.trim_start().len();
        let t = line.trim();
        // Top level and one level in (methods inside impl / class blocks).
        if indent <= 4 && is_decl_line(t, ext) {
            let t = t.trim_end_matches('{').trim_end();
            let t: String = t.chars().take(110).collect();
            rows.push(format!("{:>5}: {}{t}", i + 1, if indent > 0 { "  " } else { "" }));
        }
    }
    let mut out = String::new();
    for (k, r) in rows.iter().enumerate() {
        if out.len() + r.len() + 1 > max {
            out.push_str(&format!("  … {} more\n", rows.len() - k));
            break;
        }
        out.push_str(r);
        out.push('\n');
    }
    out
}

/// The `code_map` tool: source files with their outlines, ranked by the focus words (or by
/// most recently changed), within `max` bytes. Files that don't fit are listed by name.
pub fn code_map(path: &str, focus: Option<&str>, max: usize) -> String {
    let root = Path::new(path);
    if !root.exists() {
        return format!("Error: path '{path}' does not exist");
    }
    let words: Vec<String> = focus
        .unwrap_or("")
        .split(|c: char| !c.is_alphanumeric() && c != '_')
        .filter(|w| w.len() >= 2)
        .map(|w| w.to_lowercase())
        .collect();
    struct F {
        path: String,
        lines: usize,
        outline: Vec<(usize, String)>,
        score: f64,
        mtime: std::time::SystemTime,
    }
    let mut files = Vec::new();
    for f in walk_files(root, 20_000) {
        let ext = f.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();
        if !SOURCE_EXTS.contains(&ext.as_str()) {
            continue;
        }
        let Ok(meta) = fs::metadata(&f) else { continue };
        if meta.len() > 4 << 20 {
            continue;
        }
        let Ok(text) = fs::read_to_string(&f) else { continue };
        let lines: Vec<&str> = text.lines().collect();
        let mut outline = Vec::new();
        for (i, line) in lines.iter().enumerate() {
            let indent = line.len() - line.trim_start().len();
            let t = line.trim();
            if indent <= 4 && is_decl_line(t, &ext) {
                let t = t.trim_end_matches('{').trim_end();
                outline.push((i + 1, format!("{}{}", if indent > 0 { "  " } else { "" }, t.chars().take(100).collect::<String>())));
            }
        }
        let dp = display_path(&f);
        let mut score = 0.0;
        if !words.is_empty() {
            let lp = dp.to_lowercase();
            let lt = text.to_lowercase();
            for w in &words {
                if lp.contains(w.as_str()) {
                    score += 6.0;
                }
                score += 2.0 * outline.iter().filter(|(_, d)| d.to_lowercase().contains(w.as_str())).count().min(10) as f64;
                score += (lt.matches(w.as_str()).count().min(20)) as f64 * 0.25;
            }
        }
        files.push(F { path: dp, lines: lines.len(), outline, score, mtime: meta.modified().unwrap_or(std::time::UNIX_EPOCH) });
    }
    if files.is_empty() {
        return format!("No source files under '{path}'.");
    }
    if words.is_empty() {
        files.sort_by(|a, b| b.mtime.cmp(&a.mtime));
    } else {
        files.sort_by(|a, b| b.score.total_cmp(&a.score).then(b.mtime.cmp(&a.mtime)));
    }
    let mut out = format!("{} source files under {path}{}:\n", files.len(), if words.is_empty() { ", most recently changed first" } else { ", most relevant first" });
    let mut rest = Vec::new();
    for f in &files {
        if !words.is_empty() && f.score == 0.0 {
            rest.push(f.path.clone());
            continue;
        }
        let mut block = format!("\n{} ({} lines)\n", f.path, f.lines);
        // With a focus, the matching declarations first; at most 25 rows per file.
        let mut rows: Vec<&(usize, String)> = f.outline.iter().collect();
        if !words.is_empty() {
            rows.sort_by_key(|(l, d)| (!words.iter().any(|w| d.to_lowercase().contains(w.as_str())), *l));
        }
        for (l, d) in rows.iter().take(25) {
            block.push_str(&format!("{l:>6}: {d}\n"));
        }
        if f.outline.len() > 25 {
            block.push_str(&format!("        … {} more (read_file shows the full outline)\n", f.outline.len() - 25));
        }
        if out.len() + block.len() + 64 > max {
            rest.push(f.path.clone());
        } else {
            out.push_str(&block);
        }
    }
    if !rest.is_empty() {
        let mut names = String::new();
        for (k, r) in rest.iter().enumerate() {
            if out.len() + names.len() + r.len() + 48 > max {
                names.push_str(&format!(" and {} more", rest.len() - k));
                break;
            }
            if !names.is_empty() {
                names.push_str(", ");
            }
            names.push_str(r);
        }
        out.push_str(&format!("\nOther files: {names}\n"));
    }
    out
}

pub fn tool_search_symbols(pattern: &str, path_opt: Option<&str>, symbol_type: Option<&str>) -> Result<String, String> {
    let p = path_opt.unwrap_or(".").trim();
    let root = Path::new(if p.is_empty() { "." } else { p });
    if !root.exists() {
        return Err(format!("path '{}' does not exist", root.display()));
    }
    let pat = pattern.to_lowercase();
    let tf = symbol_type.map(|t| t.trim().to_lowercase());
    let mut matches = Vec::new();
    let mut scanned = 0;
    for f in walk_files(root, 20_000) {
        let ext = f.extension().and_then(|e| e.to_str()).unwrap_or("");
        if !matches!(ext, "rs" | "py" | "js" | "ts" | "jsx" | "tsx" | "go" | "c" | "cpp" | "h" | "hpp") {
            continue;
        }
        scanned += 1;
        let Ok(content) = fs::read_to_string(&f) else { continue };
        for (i, line) in content.lines().enumerate() {
            let t = line.trim();
            if is_decl_line(t, ext) {
                let low = t.to_lowercase();
                if low.contains(&pat) && tf.as_ref().is_none_or(|tf| low.contains(tf.as_str())) {
                    matches.push(format!("{}:{}: {t}", display_path(&f), i + 1));
                    if matches.len() >= 50 {
                        break;
                    }
                }
            }
        }
    }
    if matches.is_empty() {
        Ok(format!("No symbols matching '{pattern}' in {scanned} source file(s)."))
    } else {
        Ok(format!("{} match(es) in {scanned} file(s):\n{}", matches.len(), matches.join("\n")))
    }
}

fn is_decl_line(line: &str, ext: &str) -> bool {
    let starts = |ps: &[&str]| ps.iter().any(|p| line.starts_with(p));
    match ext {
        "rs" => starts(&["pub fn ", "fn ", "pub async fn ", "async fn ", "pub(crate) fn ", "pub struct ", "struct ", "pub enum ", "enum ", "pub trait ", "trait ", "impl ", "pub type ", "type ", "pub const ", "const ", "pub(crate) struct ", "pub unsafe fn ", "unsafe fn "]),
        "py" => starts(&["def ", "async def ", "class "]),
        "js" | "ts" | "jsx" | "tsx" => starts(&["function ", "export function ", "class ", "export class ", "interface ", "export interface ", "type ", "export type ", "const ", "export const ", "export default "]),
        "go" => starts(&["func ", "type "]),
        _ => line.contains('(') && starts(&["void ", "int ", "bool ", "class ", "struct ", "static ", "char ", "float ", "double "]),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx_yes() -> impl FnMut(&str) -> bool {
        |_| true
    }

    #[test]
    fn code_map_ranks_by_focus_and_fits() {
        let dir = std::env::temp_dir().join(format!("spindle_codemap_{}", std::process::id()));
        let _ = fs::create_dir_all(dir.join("src"));
        fs::write(dir.join("src/tokenizer.rs"), "pub struct Tokenizer {}\nimpl Tokenizer {\n    pub fn encode(&self) {}\n}\nfn merge_pairs() {}\n").unwrap();
        fs::write(dir.join("src/net.rs"), "pub fn connect() {}\npub fn send() {}\n").unwrap();
        let root = dir.to_string_lossy().to_string();
        let m = code_map(&root, Some("tokenizer encode"), 4000);
        let t = m.find("tokenizer.rs").expect("tokenizer listed");
        assert!(m.contains("pub fn encode"));
        assert!(m[t..].contains("3:"), "line numbers missing: {m}");
        // net.rs doesn't match the focus: named only, after the matching file.
        let n = m.find("net.rs").expect("net listed");
        assert!(n > t && m.contains("Other files"));
        // Tiny budget: still valid, still names the files.
        let small = code_map(&root, None, 120);
        assert!(small.contains("source files"));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn long_file_read_starts_with_outline() {
        let dir = std::env::temp_dir().join(format!("spindle_outline_{}", std::process::id()));
        let _ = fs::create_dir_all(&dir);
        let mut text = String::new();
        for i in 0..400 {
            if i % 100 == 0 {
                text.push_str(&format!("pub fn part_{i}() {{\n"));
            } else {
                text.push_str("    let x = 1;\n");
            }
        }
        let f = dir.join("long.rs");
        fs::write(&f, &text).unwrap();
        let out = read_file(&f.to_string_lossy(), None, None, 20_000);
        assert!(out.output.starts_with("Outline of"), "{}", &out.output[..80]);
        assert!(out.output.contains("301: pub fn part_300()"));
        // With an offset, no outline.
        let out2 = read_file(&f.to_string_lossy(), Some(10), Some(5), 20_000);
        assert!(!out2.output.contains("Outline"));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn todo_tool_keeps_the_list() {
        let c = parse_tool_call("<tool_call>{\"name\": \"todo\", \"arguments\": {\"items\": [{\"task\": \"read main.rs\", \"status\": \"done\"}, {\"task\": \"fix bug\", \"status\": \"in_progress\"}, {\"task\": \"run tests\", \"status\": \"pending\"}]}}</tool_call>").unwrap();
        let mut yes = |_: &str| true;
        let mut ctx = ToolCtx { confirm: &mut yes, max_chars: 4000 };
        let out = execute(&c, &mut ctx);
        assert!(out.ok, "{}", out.output);
        assert_eq!(out.title, "Tasks 1/3 done");
        let t = todos();
        assert_eq!(t.len(), 3);
        assert_eq!(t[1], ("fix bug".to_string(), TodoState::Doing));
    }

    #[test]
    fn qwen3_coder_xml_calls() {
        let text = "I'll run the tests.\n<tool_call>\n<function=run_command>\n<parameter=command>\ncargo test --release\n</parameter>\n<parameter=timeout>\n300\n</parameter>\n</function>\n</tool_call>";
        let (calls, visible) = parse_tool_calls(text);
        assert_eq!(visible, "I'll run the tests.");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "run_command");
        assert_eq!(calls[0].str_arg(&["command"]), Some("cargo test --release"));
        assert_eq!(calls[0].num_arg(&["timeout"]), Some(300));
        // Multi-line value, a JSON list value, and no <tool_call> wrapper.
        let text = "<function=edit_file>\n<parameter=path>\nsrc/a.rs\n</parameter>\n<parameter=old_string>\n    let x = 1;\n    let y = 2;\n</parameter>\n<parameter=new_string>\n    let x = 3;\n</parameter>\n</function>\n<function=todo>\n<parameter=items>\n[{\"task\": \"a\", \"status\": \"done\"}]\n</parameter>\n</function>";
        let (calls, visible) = parse_tool_calls(text);
        assert_eq!(visible, "");
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].str_arg(&["old_string"]), Some("    let x = 1;\n    let y = 2;"));
        assert!(matches!(calls[1].args.get("items"), Some(Json::Arr(a)) if a.len() == 1));
        // Streaming: the block never shows.
        let mut f = ToolMarkupFilter::default();
        let mut shown = String::new();
        for p in ["Sure. ", "<func", "tion=read_file>\n<parameter=path>x</parameter>\n", "</function>", " ok"] {
            shown.push_str(&f.push(p));
        }
        assert_eq!(shown, "Sure.  ok");
    }

    #[test]
    fn alias_call_tags_are_parsed_and_hidden() {
        let text = "Running it.\n<function_call>\n{\"name\": \"run_command\", \"arguments\": {\"command\": \"python popup.py\"}}\n</function_call>";
        let (calls, visible) = parse_tool_calls(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "run_command");
        assert_eq!(visible, "Running it.");
        let mut f = ToolMarkupFilter::default();
        let mut shown = String::new();
        for p in ["Ok ", "<func", "tion_call>{\"name\":1}", "</function_call>", " done"] {
            shown.push_str(&f.push(p));
        }
        assert_eq!(shown, "Ok  done");
        assert_eq!(normalize_call_tags("<TOOL_USE>x</Tool_Use>"), "<tool_call>x</tool_call>");
    }

    #[test]
    fn parses_qwen_tool_calls_and_strips_them() {
        let text = "I'll look first.\n<tool_call>\n{\"name\": \"read_file\", \"arguments\": {\"path\": \"src/main.rs\", \"offset\": 10, \"limit\": 20}}\n</tool_call>\n<tool_call>\n{\"name\": \"grep\", \"arguments\": {\"pattern\": \"fn main\"}}\n</tool_call>";
        let (calls, visible) = parse_tool_calls(text);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "read_file");
        assert_eq!(calls[0].str_arg(&["path"]), Some("src/main.rs"));
        assert_eq!(calls[0].num_arg(&["offset"]), Some(10));
        assert_eq!(calls[1].str_arg(&["pattern"]), Some("fn main"));
        assert_eq!(visible, "I'll look first.");
    }

    #[test]
    fn tolerates_model_slips() {
        // Missing closing tag, single quotes, args as a JSON string, flat form, code fence.
        let c = parse_tool_call("<tool_call>{'name': 'delete_file', 'arguments': {'path': 'a.py'}}").unwrap();
        assert_eq!((c.name.as_str(), c.str_arg(&["path"])), ("delete_file", Some("a.py")));
        let c = parse_tool_call("<tool_call>{\"name\":\"list_dir\",\"arguments\":\"{\\\"path\\\": \\\"src\\\"}\"}</tool_call>").unwrap();
        assert_eq!(c.str_arg(&["path"]), Some("src"));
        let c = parse_tool_call("```json\n<tool_call>\n{\"name\": \"list_dir\", \"path\": \".\"}\n</tool_call>\n```").unwrap();
        assert_eq!(c.str_arg(&["path"]), Some("."));
        // Content containing ``` and quotes must survive intact (the old parser cut it).
        let c = parse_tool_call("<tool_call>{\"name\":\"write_file\",\"arguments\":{\"path\":\"R.md\",\"content\":\"# T\\n```sh\\necho \\\"hi\\\"\\n```\\n\"}}</tool_call>").unwrap();
        assert_eq!(c.str_arg(&["content"]), Some("# T\n```sh\necho \"hi\"\n```\n"));
        // Bare call without tags is accepted only for known tools.
        assert!(parse_tool_call("{\"name\": \"grep\", \"arguments\": {\"pattern\": \"x\"}}").is_some());
        assert!(parse_tool_call("{\"name\": \"Alice\", \"age\": 3}").is_none());
    }

    #[test]
    fn file_tools_roundtrip_with_unique_edit_and_crlf() {
        let p = "target/test_tools_roundtrip.txt";
        let mut yes = ctx_yes();
        let mut ctx = ToolCtx { confirm: &mut yes, max_chars: 8000 };
        let w = execute(&ToolCall { name: "write_file".into(), args: json::parse(&format!("{{\"path\":\"{p}\",\"content\":\"a\\r\\nb\\r\\nb\\r\\n\"}}")).unwrap() }, &mut ctx);
        assert!(w.ok, "{}", w.output);
        // `\n` from the model matches the file's CRLF; ambiguous match is refused.
        let e = execute(&ToolCall { name: "edit_file".into(), args: json::parse(&format!("{{\"path\":\"{p}\",\"old_string\":\"b\",\"new_string\":\"c\"}}")).unwrap() }, &mut ctx);
        assert!(!e.ok && e.output.contains("2 times"), "{}", e.output);
        let e = execute(&ToolCall { name: "edit_file".into(), args: json::parse(&format!("{{\"path\":\"{p}\",\"old_string\":\"a\\nb\",\"new_string\":\"x\\ny\"}}")).unwrap() }, &mut ctx);
        assert!(e.ok, "{}", e.output);
        assert_eq!(fs::read_to_string(p).unwrap(), "x\r\ny\r\nb\r\n");
        let r = execute(&ToolCall { name: "read_file".into(), args: json::parse(&format!("{{\"path\":\"{p}\"}}")).unwrap() }, &mut ctx);
        assert!(r.output.contains("    1\tx") && r.output.contains("    3\tb"), "{}", r.output);
        // Undo both edits back to nothing.
        pop_undo_path(p).unwrap().unwrap();
        assert_eq!(fs::read_to_string(p).unwrap(), "a\r\nb\r\nb\r\n");
        pop_undo_path(p).unwrap().unwrap();
        assert!(!Path::new(p).exists());
    }

    #[test]
    fn delete_requires_confirmation() {
        let p = "target/test_tools_delete.txt";
        fs::write(p, "keep me").unwrap();
        let mut no = |_: &str| false;
        let mut ctx = ToolCtx { confirm: &mut no, max_chars: 8000 };
        let d = execute(&ToolCall { name: "delete_file".into(), args: json::parse(&format!("{{\"path\":\"{p}\"}}")).unwrap() }, &mut ctx);
        assert!(!d.ok && Path::new(p).exists());
        let mut yes = ctx_yes();
        let mut ctx = ToolCtx { confirm: &mut yes, max_chars: 8000 };
        let d = execute(&ToolCall { name: "delete_file".into(), args: json::parse(&format!("{{\"path\":\"{p}\"}}")).unwrap() }, &mut ctx);
        assert!(d.ok && !Path::new(p).exists());
        pop_undo_path(p).unwrap().unwrap();
        assert_eq!(fs::read_to_string(p).unwrap(), "keep me");
        let _ = fs::remove_file(p);
    }

    #[test]
    fn glob_and_grep_search_the_tree() {
        assert!(glob_match("*.rs", "src/tools.rs"));
        assert!(glob_match("src/**/*.rs", "src/a/b/c.rs"));
        assert!(glob_match("src/**/*.rs", "src/c.rs"));
        assert!(!glob_match("src/*.rs", "src/a/c.rs"));
        let mut yes = ctx_yes();
        let mut ctx = ToolCtx { confirm: &mut yes, max_chars: 8000 };
        let g = execute(&ToolCall { name: "glob".into(), args: json::parse(r#"{"pattern":"src/*.rs"}"#).unwrap() }, &mut ctx);
        assert!(g.output.contains("src/tools.rs"), "{}", g.output);
        let g = execute(&ToolCall { name: "grep".into(), args: json::parse(r#"{"pattern":"fn glob_match","path":"src","include":"*.rs"}"#).unwrap() }, &mut ctx);
        assert!(g.ok && g.output.contains("src/tools.rs:"), "{}", g.output);
        assert!(g.title.contains("match"));
    }

    #[test]
    fn symbol_search_finds_declarations() {
        let res = tool_search_symbols("SmolLm2Engine", Some("src"), Some("struct")).unwrap();
        assert!(res.contains("pub struct SmolLm2Engine"), "{res}");
    }

    #[test]
    fn streaming_filter_hides_spelled_out_tags() {
        let mut f = ToolMarkupFilter::default();
        let pieces = ["Reading", " now.\n", "<", "tool", "_call", ">\n{\"name\": ", "\"read_file\"}", "\n</", "tool_call", ">", " Done", " <b>"];
        let shown: String = pieces.iter().map(|p| f.push(p)).collect();
        assert_eq!(shown, "Reading now.\n Done <b>");
        assert_eq!(partial_tag_suffix("abc <tool", "<tool_call>"), 5);
        assert_eq!(partial_tag_suffix("x <", "<tool_call>"), 1);
        assert_eq!(partial_tag_suffix("x", "<tool_call>"), 0);
    }

    #[test]
    fn truncation_is_char_safe() {
        let s = "ñ".repeat(5000);
        let t = truncate_middle(&s, 1001);
        assert!(t.len() < 1200 && t.contains("omitted"));
    }

    #[test]
    fn tools_prompt_is_valid_json_per_line() {
        let p = tools_system_prompt();
        let body = &p[p.find("<tools>\n").unwrap() + 8..p.rfind("</tools>").unwrap()];
        for line in body.lines() {
            let v = json::parse(line).unwrap_or_else(|e| panic!("{e}: {line}"));
            assert!(v.get("function").and_then(|f| f.get("name")).is_some());
        }
    }
}
