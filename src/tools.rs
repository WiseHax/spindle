//! Spindle Agent Tools: Claude Code & Antigravity-style Tool Implementations.
//!
//! Provides:
//! - `read_file`: Line-numbered source inspection
//! - `write_file`: File creation & overwrite with parent directory synthesis
//! - `edit_file`: Exact search-and-replace surgical editing
//! - `list_dir`: Directory listing with file sizes
//! - `search_web`: DuckDuckGo online web search via native curl.exe
//! - `run_command`: Sandboxed shell execution with AntiSabotage Guard

use std::{
    fs,
    path::Path,
    sync::Mutex,
    time::Duration,
};

use crate::guard::{self, GuardDecision};

static UNDO_STACK: Mutex<Vec<(String, Option<String>)>> = Mutex::new(Vec::new());

/// Saves a shadow snapshot of the file before modifying, enabling `/undo`.
pub fn push_undo_snapshot(path_str: &str) {
    let p = Path::new(path_str);
    let prev = if p.exists() {
        fs::read_to_string(p).ok()
    } else {
        None
    };
    if let Ok(mut stack) = UNDO_STACK.lock() {
        stack.push((path_str.to_string(), prev));
    }
}

/// Reverts the most recent file change made by the agent.
pub fn pop_undo() -> Option<Result<String, String>> {
    let (path, prev) = {
        let mut stack = UNDO_STACK.lock().ok()?;
        stack.pop()?
    };
    match prev {
        Some(old_content) => {
            let res = fs::write(&path, old_content)
                .map(|_| format!("Restored '{}' to previous state.", path))
                .map_err(|e| format!("Failed to restore '{}': {}", path, e));
            Some(res)
        }
        None => {
            let res = fs::remove_file(&path)
                .map(|_| format!("Removed newly created file '{}'.", path))
                .map_err(|e| format!("Failed to remove '{}': {}", path, e));
            Some(res)
        }
    }
}

/// Reverts the most recent file change specifically for a given path.
pub fn pop_undo_path(target_path: &str) -> Option<Result<String, String>> {
    let (path, prev) = {
        let mut stack = UNDO_STACK.lock().ok()?;
        let idx = stack.iter().rposition(|(p, _)| p == target_path)?;
        stack.remove(idx)
    };
    match prev {
        Some(old_content) => {
            let res = fs::write(&path, old_content)
                .map(|_| format!("Restored '{}' to previous state.", path))
                .map_err(|e| format!("Failed to restore '{}': {}", path, e));
            Some(res)
        }
        None => {
            let res = fs::remove_file(&path)
                .map(|_| format!("Removed newly created file '{}'.", path))
                .map_err(|e| format!("Failed to remove '{}': {}", path, e));
            Some(res)
        }
    }
}

/// Checks if SPINDLE.md exists in current workspace to feed persistent project memory.
pub fn load_spindle_md() -> Option<String> {
    for candidate in ["SPINDLE.md", "spindle.md", ".spindle.md"] {
        if let Ok(text) = fs::read_to_string(candidate) {
            let trimmed = text.trim();
            if !trimmed.is_empty() {
                return Some(trimmed.to_string());
            }
        }
    }
    None
}

/// System prompt injection defining available tools for Qwen / ChatML models.
pub const TOOLS_SYSTEM_PROMPT: &str = r#"
# Tools
You have access to the following tools to assist the user:

## `read_file`
Description: Read the contents of a file with line numbers.
Parameters: {"type": "object", "properties": {"path": {"type": "string"}, "start_line": {"type": "integer"}, "limit": {"type": "integer"}}, "required": ["path"]}

## `write_file`
Description: Create or overwrite a file with given content.
Parameters: {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}

## `edit_file`
Description: Perform exact search-and-replace edit in a file.
Parameters: {"type": "object", "properties": {"path": {"type": "string"}, "target": {"type": "string"}, "replacement": {"type": "string"}}, "required": ["path", "target", "replacement"]}

## `delete_file`
Description: Delete a file from the workspace.
Parameters: {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}

## `list_dir`
Description: List files and subdirectories in a folder.
Parameters: {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}

## `run_command`
Description: Run a shell command in the project directory.
Parameters: {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}

## `search_symbols`
Description: Fast AST/symbol scanner across workspace. Finds function signatures, structs, classes, enums, traits and types without reading full files.
Parameters: {"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}, "symbol_type": {"type": "string"}}, "required": ["pattern"]}

## `search_web`
Description: Search the internet online for live information.
Parameters: {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}

To invoke a tool, output:
<tool_call>
{"name": "<tool_name>", "arguments": {"<param>": "<value>"}}
</tool_call>
Wait for the <tool_response> before continuing.
After receiving a tool response that fulfills the user's request, provide your final direct answer to the user. Do not call the same tool repeatedly.
"#;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolCall {
    pub name: String,
    pub path: Option<String>,
    pub content: Option<String>,
    pub target: Option<String>,
    pub replacement: Option<String>,
    pub command: Option<String>,
    pub query: Option<String>,
    pub pattern: Option<String>,
    pub symbol_type: Option<String>,
    pub start_line: Option<usize>,
    pub limit: Option<usize>,
}

/// Reads a file and returns contents formatted with 1-based line numbers.
pub fn tool_read_file(path_str: &str, start_line: Option<usize>, limit: Option<usize>) -> Result<String, String> {
    if let GuardDecision::Block(r) = guard::check_path(path_str, false) {
        return Err(format!("[Blocked by Security Guard: {r}]"));
    }
    let p = Path::new(path_str);
    if !p.exists() {
        return Err(format!("File not found: '{path_str}'"));
    }
    if p.is_dir() {
        return Err(format!("'{path_str}' is a directory, not a file. Use list_dir instead."));
    }
    let raw = fs::read_to_string(p).map_err(|e| format!("Failed to read '{path_str}': {e}"))?;
    let lines: Vec<&str> = raw.lines().collect();
    let total = lines.len();
    let start = start_line.unwrap_or(1).saturating_sub(1).min(total);
    let count = limit.unwrap_or(400).min(total.saturating_sub(start));

    let mut out = String::new();
    out.push_str(&format!("--- {path_str} (showing lines {}-{} of {total}) ---\n", start + 1, start + count));
    for (i, line) in lines[start..start + count].iter().enumerate() {
        out.push_str(&format!("{:>4} | {}\n", start + i + 1, line));
    }
    Ok(out)
}

/// Writes or overwrites a file on disk, automatically creating parent folders.
pub fn tool_write_file(path_str: &str, content: &str) -> Result<String, String> {
    if let GuardDecision::Block(r) = guard::check_path(path_str, true) {
        return Err(format!("[Blocked by Security Guard: {r}]"));
    }
    let p = Path::new(path_str);
    if let Some(parent) = p.parent() {
        if !parent.as_os_str().is_empty() && !parent.exists() {
            let _ = fs::create_dir_all(parent);
        }
    }
    push_undo_snapshot(path_str);
    fs::write(p, content).map_err(|e| format!("Failed to write '{path_str}': {e}"))?;
    Ok(format!("Successfully wrote {} bytes to '{path_str}'.", content.len()))
}

/// Surgically replaces exact target occurrences in a file.
pub fn tool_edit_file(path_str: &str, target: &str, replacement: &str) -> Result<String, String> {
    if let GuardDecision::Block(r) = guard::check_path(path_str, true) {
        return Err(format!("[Blocked by Security Guard: {r}]"));
    }
    let p = Path::new(path_str);
    if !p.exists() {
        return Err(format!("File '{path_str}' does not exist."));
    }
    let old = fs::read_to_string(p).map_err(|e| format!("Could not read '{path_str}': {e}"))?;
    if !old.contains(target) {
        return Err(format!("Target text not found in '{path_str}'. Verify exact whitespace and lines."));
    }
    let occurrences = old.matches(target).count();
    let updated = old.replace(target, replacement);
    push_undo_snapshot(path_str);
    fs::write(p, updated).map_err(|e| format!("Could not write updated '{path_str}': {e}"))?;
    Ok(format!("Successfully replaced {occurrences} occurrence(s) in '{path_str}'."))
}

/// Deletes a file on disk after saving a shadow undo snapshot.
pub fn tool_delete_file(path_str: &str) -> Result<String, String> {
    if let GuardDecision::Block(r) = guard::check_path(path_str, true) {
        return Err(format!("[Blocked by Security Guard: {r}]"));
    }
    let p = Path::new(path_str);
    if !p.exists() {
        return Err(format!("File '{path_str}' does not exist."));
    }
    if p.is_dir() {
        return Err(format!("'{path_str}' is a directory, not a file. Use run_command with rmdir to delete directories."));
    }
    push_undo_snapshot(path_str);
    fs::remove_file(p).map_err(|e| format!("Failed to delete '{path_str}': {e}"))?;
    Ok(format!("Successfully deleted '{path_str}'."))
}

/// Lists files and subdirectories with sizes.
pub fn tool_list_dir(path_str: &str) -> Result<String, String> {
    let p = if path_str.trim().is_empty() { Path::new(".") } else { Path::new(path_str) };
    if !p.exists() {
        return Err(format!("Directory '{path_str}' does not exist."));
    }
    let entries = fs::read_dir(p).map_err(|e| format!("Could not read directory '{path_str}': {e}"))?;
    let mut dirs = Vec::new();
    let mut files = Vec::new();

    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if let Ok(meta) = entry.metadata() {
            if meta.is_dir() {
                dirs.push(format!("{name}/"));
            } else {
                let sz = meta.len();
                let sz_str = if sz > (1024 * 1024) {
                    format!("{:.1} MiB", sz as f64 / (1024.0 * 1024.0))
                } else if sz > 1024 {
                    format!("{:.1} KiB", sz as f64 / 1024.0)
                } else {
                    format!("{sz} B")
                };
                files.push(format!("{name:<35} ({sz_str})"));
            }
        }
    }
    dirs.sort();
    files.sort();

    let mut out = format!("Directory listing for '{}':\n", p.display());
    for d in dirs {
        out.push_str(&format!("  [DIR]  {d}\n"));
    }
    for f in files {
        out.push_str(&format!("  [FILE] {f}\n"));
    }
    Ok(out)
}

/// Searches the web online using DuckDuckGo via native curl.exe (zero external dependencies).
pub fn tool_search_web(query: &str) -> Result<String, String> {
    let q = query.trim();
    if q.is_empty() {
        return Err("Search query cannot be empty.".into());
    }
    let mut encoded = String::new();
    for b in q.bytes() {
        if b.is_ascii_alphanumeric() || b == b'-' || b == b'_' || b == b'.' || b == b'~' {
            encoded.push(b as char);
        } else if b == b' ' {
            encoded.push('+');
        } else {
            encoded.push_str(&format!("%{:02X}", b));
        }
    }
    let url = format!("https://html.duckduckgo.com/html/?q={encoded}");

    let mut cmd = std::process::Command::new("curl.exe");
    cmd.args(["-s", "-L", "--max-time", "10", "-A", "Mozilla/5.0 (Windows NT 10.0; Win64; x64)", &url]);

    let output = cmd.output().map_err(|e| format!("Failed to invoke curl for web search: {e}"))?;
    if !output.status.success() {
        return Err(format!("curl returned error code {:?}", output.status.code()));
    }
    let html = String::from_utf8_lossy(&output.stdout);
    let mut results = Vec::new();

    // Lightweight extraction of DuckDuckGo HTML results
    let snippet_tag = "class=\"result__snippet\"";
    let title_tag = "class=\"result__a\"";

    for part in html.split(title_tag).skip(1).take(5) {
        let title = if let Some(start) = part.find('>') {
            if let Some(end) = part[start..].find("</a>") {
                strip_html(&part[start + 1..start + end])
            } else {
                continue;
            }
        } else {
            continue;
        };

        let snippet = if let Some(s_idx) = part.find(snippet_tag) {
            let s_part = &part[s_idx..];
            if let Some(start) = s_part.find('>') {
                if let Some(end) = s_part[start..].find("</a>") {
                    strip_html(&s_part[start + 1..start + end])
                } else {
                    String::new()
                }
            } else {
                String::new()
            }
        } else {
            String::new()
        };

        if !title.is_empty() {
            results.push(format!("- **{}**\n  {}", title.trim(), snippet.trim()));
        }
    }

    if results.is_empty() {
        Ok(format!("Search completed for '{q}', but no direct snippets could be parsed from DuckDuckGo HTML."))
    } else {
        Ok(format!("Online search results for '{q}':\n\n{}", results.join("\n\n")))
    }
}

/// Runs a shell command through the AntiSabotage Guard.
pub fn tool_run_command<F>(cmd: &str, confirm_fn: F) -> Result<String, String>
where
    F: FnOnce(&str) -> bool,
{
    match guard::check_command(cmd) {
        GuardDecision::Block(reason) => Err(format!("[Blocked by Security Guard: {reason}]")),
        GuardDecision::NeedConfirmation(reason) => {
            if confirm_fn(&format!("{reason} (command: '{cmd}')")) {
                let (out, code) = guard::run_sandboxed(cmd, Duration::from_secs(30))
                    .map_err(|e| format!("Failed to execute command: {e}"))?;
                Ok(format!("Exit code: {code}\nOutput:\n{out}"))
            } else {
                Err("[Aborted by user confirmation]".into())
            }
        }
        GuardDecision::Allow => {
            let (out, code) = guard::run_sandboxed(cmd, Duration::from_secs(30))
                .map_err(|e| format!("Failed to execute command: {e}"))?;
            Ok(format!("Exit code: {code}\nOutput:\n{out}"))
        }
    }
}

fn strip_html(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut in_tag = false;
    for ch in input.chars() {
        if ch == '<' {
            in_tag = true;
        } else if ch == '>' {
            in_tag = false;
        } else if !in_tag {
            out.push(ch);
        }
    }
    out.replace("&amp;", "&")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
}

/// Scans source files across a directory recursively for symbols matching `pattern`.
pub fn tool_search_symbols(pattern: &str, path_opt: Option<&str>, symbol_type: Option<&str>) -> Result<String, String> {
    let p = path_opt.unwrap_or(".").trim();
    let root = if p.is_empty() { Path::new(".") } else { Path::new(p) };
    if !root.exists() {
        return Err(format!("Path '{}' does not exist.", root.display()));
    }
    if let GuardDecision::Block(r) = guard::check_path(p, false) {
        return Err(format!("[Blocked by Security Guard: {r}]"));
    }

    let pat_lower = pattern.to_lowercase();
    let type_filter = symbol_type.map(|t| t.trim().to_lowercase());

    let mut matches = Vec::new();
    let mut files_scanned = 0;
    scan_symbols_recursive(root, &pat_lower, type_filter.as_deref(), &mut matches, &mut files_scanned, 50)?;

    if matches.is_empty() {
        Ok(format!("No symbols matching '{pattern}' found across {files_scanned} source file(s)."))
    } else {
        let mut out = format!("Found {} symbol match(es) across {} source file(s) scanned:\n\n", matches.len(), files_scanned);
        for m in matches {
            out.push_str(&format!("  {}\n", m));
        }
        Ok(out)
    }
}

fn scan_symbols_recursive(
    dir: &Path,
    pattern_lower: &str,
    type_filter: Option<&str>,
    matches: &mut Vec<String>,
    files_scanned: &mut usize,
    max_matches: usize,
) -> Result<(), String> {
    if matches.len() >= max_matches {
        return Ok(());
    }
    if dir.is_file() {
        scan_single_file_symbols(dir, pattern_lower, type_filter, matches, files_scanned, max_matches)?;
        return Ok(());
    }
    let entries = fs::read_dir(dir).map_err(|e| format!("Could not read directory '{}': {e}", dir.display()))?;
    for entry in entries.flatten() {
        if matches.len() >= max_matches {
            break;
        }
        let p = entry.path();
        let fname = entry.file_name().to_string_lossy().to_string();
        if p.is_dir() {
            if matches!(fname.as_str(), ".git" | "target" | "node_modules" | ".vscode" | "dist" | "build" | "venv" | ".venv" | ".cargo" | "vendor") {
                continue;
            }
            let _ = scan_symbols_recursive(&p, pattern_lower, type_filter, matches, files_scanned, max_matches);
        } else {
            let _ = scan_single_file_symbols(&p, pattern_lower, type_filter, matches, files_scanned, max_matches);
        }
    }
    Ok(())
}

fn scan_single_file_symbols(
    p: &Path,
    pattern_lower: &str,
    type_filter: Option<&str>,
    matches: &mut Vec<String>,
    files_scanned: &mut usize,
    max_matches: usize,
) -> Result<(), String> {
    let ext = p.extension().and_then(|e| e.to_str()).unwrap_or("");
    let is_source = matches!(ext, "rs" | "py" | "js" | "ts" | "jsx" | "tsx" | "go" | "c" | "cpp" | "h" | "hpp" | "toml");
    if !is_source {
        return Ok(());
    }
    *files_scanned += 1;
    let Ok(content) = fs::read_to_string(p) else {
        return Ok(());
    };

    let p_display = p.to_string_lossy().replace('\\', "/");
    for (idx, line) in content.lines().enumerate() {
        if matches.len() >= max_matches {
            break;
        }
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with("//") || trimmed.starts_with('#') || trimmed.starts_with("/*") {
            continue;
        }
        if is_decl_line(trimmed, ext) {
            let line_lower = trimmed.to_lowercase();
            if line_lower.contains(pattern_lower) {
                if let Some(tf) = type_filter {
                    if !line_lower.contains(tf) {
                        continue;
                    }
                }
                matches.push(format!("{}:{}: {}", p_display, idx + 1, trimmed));
            }
        }
    }
    Ok(())
}

fn is_decl_line(line: &str, ext: &str) -> bool {
    match ext {
        "rs" => {
            line.starts_with("pub fn ") || line.starts_with("fn ") ||
            line.starts_with("pub async fn ") || line.starts_with("async fn ") ||
            line.starts_with("pub struct ") || line.starts_with("struct ") ||
            line.starts_with("pub enum ") || line.starts_with("enum ") ||
            line.starts_with("pub trait ") || line.starts_with("trait ") ||
            line.starts_with("impl ") || line.starts_with("pub type ") ||
            line.starts_with("type ") || line.starts_with("pub const ") ||
            line.starts_with("const ")
        }
        "py" => {
            line.starts_with("def ") || line.starts_with("async def ") || line.starts_with("class ")
        }
        "js" | "ts" | "jsx" | "tsx" => {
            line.starts_with("function ") || line.starts_with("export function ") ||
            line.starts_with("class ") || line.starts_with("export class ") ||
            line.starts_with("interface ") || line.starts_with("export interface ") ||
            line.starts_with("type ") || line.starts_with("export type ") ||
            line.starts_with("const ") || line.starts_with("export const ")
        }
        "go" => {
            line.starts_with("func ") || line.starts_with("type ")
        }
        "c" | "cpp" | "h" | "hpp" => {
            line.contains('(') && (line.starts_with("void ") || line.starts_with("int ") || line.starts_with("bool ") || line.starts_with("class ") || line.starts_with("struct "))
        }
        _ => line.contains("fn ") || line.contains("struct ") || line.contains("class "),
    }
}

/// Parses `<tool_call> ... </tool_call>` payload from the model's generated text.
/// Tolerates whitespace variations, single quotes, markdown fences, and missing closing tags.
pub fn parse_tool_call(text: &str) -> Option<ToolCall> {
    let lower = text.to_lowercase();
    let body = if let Some(s) = lower.find("<tool_call").or_else(|| lower.find("<toolcall")).or_else(|| lower.find("<tool-call")) {
        let after_tag_start = &text[s..];
        let tag_close = after_tag_start.find('>')?;
        let payload_start = s + tag_close + 1;
        let payload_slice = &text[payload_start..];
        let lower_slice = &lower[payload_start..];
        if let Some(e) = lower_slice.find("</tool_call").or_else(|| lower_slice.find("</toolcall")).or_else(|| lower_slice.find("</tool-call")).or_else(|| lower_slice.find("```")) {
            &payload_slice[..e]
        } else if let Some(open_brace) = payload_slice.find('{') {
            extract_balanced_json(&payload_slice[open_brace..])?
        } else {
            payload_slice
        }
    } else if let Some(open_brace) = text.find('{') {
        let candidate = extract_balanced_json(&text[open_brace..])?;
        if candidate.contains("\"name\"") || candidate.contains("'name'") {
            candidate
        } else {
            return None;
        }
    } else {
        return None;
    };

    let body = body.trim();
    let name = extract_json_string(body, "name")?;
    let path = extract_json_string(body, "path");
    let content = extract_json_string(body, "content");
    let target = extract_json_string(body, "target");
    let replacement = extract_json_string(body, "replacement");
    let command = extract_json_string(body, "command");
    let query = extract_json_string(body, "query");
    let pattern = extract_json_string(body, "pattern");
    let symbol_type = extract_json_string(body, "symbol_type");
    let start_line = extract_json_number(body, "start_line");
    let limit = extract_json_number(body, "limit");

    Some(ToolCall {
        name,
        path,
        content,
        target,
        replacement,
        command,
        query,
        pattern,
        symbol_type,
        start_line,
        limit,
    })
}

fn extract_balanced_json(s: &str) -> Option<&str> {
    let mut depth = 0;
    let mut in_quote = false;
    let mut quote_char = '"';
    let mut escape = false;
    let mut end_idx = None;

    for (i, c) in s.char_indices() {
        if escape {
            escape = false;
            continue;
        }
        if c == '\\' {
            escape = true;
            continue;
        }
        if c == '"' || c == '\'' {
            if !in_quote {
                in_quote = true;
                quote_char = c;
            } else if c == quote_char {
                in_quote = false;
            }
            continue;
        }
        if !in_quote {
            if c == '{' {
                depth += 1;
            } else if c == '}' {
                depth -= 1;
                if depth == 0 {
                    end_idx = Some(i + 1);
                    break;
                }
            }
        }
    }
    end_idx.map(|idx| &s[..idx])
}

fn extract_json_string(json: &str, key: &str) -> Option<String> {
    let pat_double = format!("\"{key}\"");
    let pat_single = format!("'{key}'");
    let idx = json.find(&pat_double).or_else(|| json.find(&pat_single))?;
    let key_len = if json[idx..].starts_with(&pat_double) { pat_double.len() } else { pat_single.len() };
    let after = &json[idx + key_len..];
    let colon = after.find(':')?;
    let val_part = after[colon + 1..].trim_start();
    let quote = val_part.chars().next()?;
    if quote != '"' && quote != '\'' {
        return None;
    }
    let mut out = String::new();
    let mut chars = val_part[quote.len_utf8()..].chars();
    let mut escape = false;

    while let Some(ch) = chars.next() {
        if escape {
            match ch {
                'n' => out.push('\n'),
                'r' => out.push('\r'),
                't' => out.push('\t'),
                '\\' => out.push('\\'),
                '"' => out.push('"'),
                '\'' => out.push('\''),
                other => out.push(other),
            }
            escape = false;
        } else if ch == '\\' {
            escape = true;
        } else if ch == quote {
            return Some(out);
        } else {
            out.push(ch);
        }
    }
    None
}

fn extract_json_number(json: &str, key: &str) -> Option<usize> {
    let pat_double = format!("\"{key}\"");
    let pat_single = format!("'{key}'");
    let idx = json.find(&pat_double).or_else(|| json.find(&pat_single))?;
    let key_len = if json[idx..].starts_with(&pat_double) { pat_double.len() } else { pat_single.len() };
    let after = &json[idx + key_len..];
    let colon = after.find(':')?;
    let val_part = after[colon + 1..].trim_start();
    let num_str: String = val_part.chars().take_while(|c| c.is_ascii_digit()).collect();
    num_str.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_tool_call() {
        let text = r#"
I will now read the file.
<tool_call>
{"name": "read_file", "arguments": {"path": "src/main.rs", "start_line": 10, "limit": 20}}
</tool_call>
"#;
        let call = parse_tool_call(text).expect("parsed call");
        assert_eq!(call.name, "read_file");
        assert_eq!(call.path.as_deref(), Some("src/main.rs"));
        assert_eq!(call.start_line, Some(10));
        assert_eq!(call.limit, Some(20));
    }

    #[test]
    fn test_file_tools_roundtrip() {
        let test_path = "target/test_dummy_file.txt";
        let content = "Hello Spindle!\nLine 2\nLine 3";
        let res = tool_write_file(test_path, content);
        assert!(res.is_ok(), "write file failed: {:?}", res);

        let read_res = tool_read_file(test_path, None, None);
        assert!(read_res.is_ok());
        assert!(read_res.unwrap().contains("Hello Spindle!"));

        let edit_res = tool_edit_file(test_path, "Line 2", "Modified Line 2");
        assert!(edit_res.is_ok());

        let read_res2 = tool_read_file(test_path, None, None).unwrap();
        assert!(read_res2.contains("Modified Line 2"));

        let _ = fs::remove_file(test_path);
    }

    #[test]
    fn test_undo_functionality() {
        let test_path = "target/test_undo_file.txt";
        let _ = fs::remove_file(test_path);

        // 1. Initial write
        tool_write_file(test_path, "Version 1").unwrap();
        // 2. Edit
        tool_edit_file(test_path, "Version 1", "Version 2").unwrap();
        assert_eq!(fs::read_to_string(test_path).unwrap(), "Version 2");

        // 3. Undo edit
        let undo1 = pop_undo_path(test_path).expect("has undo");
        assert!(undo1.is_ok());
        assert_eq!(fs::read_to_string(test_path).unwrap(), "Version 1");

        // 4. Undo initial creation
        let undo2 = pop_undo_path(test_path).expect("has undo");
        assert!(undo2.is_ok());
        assert!(!Path::new(test_path).exists());
    }

    #[test]
    fn test_search_symbols_tool() {
        let res = tool_search_symbols("SmolLm2Engine", Some("src"), Some("struct")).expect("search succeeds");
        assert!(res.contains("smollm_chat.rs"), "Expected result to contain smollm_chat.rs, got:\n{}", res);
        assert!(res.contains("pub struct SmolLm2Engine"), "Expected result to contain struct SmolLm2Engine, got:\n{}", res);

        let res_call = parse_tool_call(r#"
<tool_call>
{"name": "search_symbols", "arguments": {"pattern": "tool_read_file", "symbol_type": "fn"}}
</tool_call>
"#).expect("parsed search_symbols call");
        assert_eq!(res_call.name, "search_symbols");
        assert_eq!(res_call.pattern.as_deref(), Some("tool_read_file"));
        assert_eq!(res_call.symbol_type.as_deref(), Some("fn"));
    }

    #[test]
    fn test_delete_file_tool() {
        let test_path = "target/test_delete_me.txt";
        tool_write_file(test_path, "temporary content").unwrap();
        assert!(Path::new(test_path).exists());

        let del_res = tool_delete_file(test_path);
        assert!(del_res.is_ok());
        assert!(!Path::new(test_path).exists());

        // Test that undo can restore the deleted file
        let undo = pop_undo_path(test_path).expect("has undo");
        assert!(undo.is_ok());
        assert!(Path::new(test_path).exists());
        assert_eq!(fs::read_to_string(test_path).unwrap(), "temporary content");

        let _ = fs::remove_file(test_path);
    }

    #[test]
    fn test_robust_parse_tool_call_variations() {
        // Variation 1: Extra whitespace in opening tag and closing tag
        let v1 = "<tool_call >\n{\"name\": \"delete_file\", \"arguments\": {\"path\": \"hello.py\"}}\n</tool_call >";
        let c1 = parse_tool_call(v1).expect("v1 parsed");
        assert_eq!(c1.name, "delete_file");
        assert_eq!(c1.path.as_deref(), Some("hello.py"));

        // Variation 2: Missing closing tag (model stopped at EOS)
        let v2 = "<tool_call>\n{\"name\": \"delete_file\", \"arguments\": {\"path\": \"hello.py\"}}";
        let c2 = parse_tool_call(v2).expect("v2 parsed without closing tag");
        assert_eq!(c2.name, "delete_file");
        assert_eq!(c2.path.as_deref(), Some("hello.py"));

        // Variation 3: Single quotes
        let v3 = "<tool_call>{'name': 'delete_file', 'arguments': {'path': 'hello.py'}}</tool_call>";
        let c3 = parse_tool_call(v3).expect("v3 parsed with single quotes");
        assert_eq!(c3.name, "delete_file");
        assert_eq!(c3.path.as_deref(), Some("hello.py"));

        // Variation 4: Markdown codeblock wrapped
        let v4 = "```json\n<tool_call>\n{\"name\": \"list_dir\", \"path\": \".\"}\n</tool_call>\n```";
        let c4 = parse_tool_call(v4).expect("v4 parsed with markdown fences");
        assert_eq!(c4.name, "list_dir");
        assert_eq!(c4.path.as_deref(), Some("."));
    }
}

