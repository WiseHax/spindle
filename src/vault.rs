//! The output vault: long tool results are kept in full here, and the model gets a digest
//! plus a short id. `recall` shows any part of the original later, so shortening a result
//! never loses it.
//!
//! On a CPU, every token of tool output has to be processed before the model can answer:
//! at 25 tok/s, a 6,000-token build log costs four minutes. A digest of the lines that matter
//! (errors, failing tests, `file:line` locations, the end of the log) usually carries the
//! whole message in a few hundred tokens.

use std::sync::Mutex;

struct Entry {
    id: usize,
    title: String,
    text: String,
}

static VAULT: Mutex<(usize, Vec<Entry>)> = Mutex::new((0, Vec::new()));

/// Total bytes kept; the oldest entries go first.
const VAULT_MAX_BYTES: usize = 16 << 20;

/// Stores `text` and returns its id (`v1`, `v2`, …).
pub fn put(title: &str, text: &str) -> String {
    let mut v = VAULT.lock().unwrap_or_else(|e| e.into_inner());
    v.0 += 1;
    let id = v.0;
    v.1.push(Entry { id, title: title.to_string(), text: text.to_string() });
    let mut total: usize = v.1.iter().map(|e| e.text.len()).sum();
    while total > VAULT_MAX_BYTES && v.1.len() > 1 {
        total -= v.1.remove(0).text.len();
    }
    format!("v{id}")
}

pub fn clear() {
    let mut v = VAULT.lock().unwrap_or_else(|e| e.into_inner());
    v.1.clear();
}

fn get(id: &str) -> Option<(String, String)> {
    let n: usize = id.trim().trim_start_matches(['v', 'V', '#']).parse().ok()?;
    let v = VAULT.lock().unwrap_or_else(|e| e.into_inner());
    v.1.iter().find(|e| e.id == n).map(|e| (e.title.clone(), e.text.clone()))
}

/// What kind of text a digest is for.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Kind {
    /// Command output: keep error lines and the end.
    Log,
    /// Anything else (pages, search results): keep the start and the end.
    Text,
}

/// `text` if it fits in `max` bytes; otherwise a digest that fits, with the full text in the
/// vault and a note on how to get the rest.
pub fn fit(title: &str, text: &str, max: usize, kind: Kind) -> String {
    if text.len() <= max {
        return text.to_string();
    }
    let id = put(title, text);
    let n_lines = text.lines().count();
    let note = format!(
        "\n[Shortened: {n_lines} lines, {} bytes in total, saved as {id}. To see more, call recall with {{\"id\": \"{id}\", \"lines\": \"120-180\"}} or {{\"id\": \"{id}\", \"find\": \"some text\"}}.]",
        text.len()
    );
    let room = max.saturating_sub(note.len()).max(256);
    let body = match kind {
        Kind::Log => digest_log(text, room),
        Kind::Text => crate::tools::truncate_middle(text, room),
    };
    format!("{body}{note}")
}

/// Lines worth keeping from a build, test or command log.
fn is_signal(line: &str) -> bool {
    let l = line.to_ascii_lowercase();
    const WORDS: &[&str] = &[
        "error", "warning:", "failed", "failure", "fail:", "panicked", "panic:", "exception", "traceback",
        "assert", "denied", "not found", "cannot ", "can't ", "undefined", "unresolved", "fatal", "segfault",
        "timed out", "expected", "--> ",
    ];
    if WORDS.iter().any(|w| l.contains(w)) {
        // "0 errors", "no warnings" and the like are noise.
        return !(l.contains(" 0 error") || l.contains("0 failed") || l.contains("no error") || l.starts_with("test result: ok"));
    }
    // `path/file.rs:12:5`, `file.py", line 40`
    has_location(line)
}

fn has_location(line: &str) -> bool {
    let b = line.as_bytes();
    for i in 1..b.len().saturating_sub(1) {
        if b[i] == b':' && b[i + 1].is_ascii_digit() {
            // Something like an extension right before the colon.
            let before = &line[..i];
            if let Some(dot) = before.rfind('.') {
                let ext = &before[dot + 1..];
                if (1..=5).contains(&ext.len()) && ext.chars().all(|c| c.is_ascii_alphanumeric()) {
                    return true;
                }
            }
        }
    }
    line.contains("\", line ")
}

/// Keeps the first lines, every signal line with one line of context on each side, and as
/// much of the end as fits, in their original order, with `…` where lines were dropped.
pub fn digest_log(text: &str, max: usize) -> String {
    let lines: Vec<&str> = text.lines().collect();
    let n = lines.len();
    let mut keep = vec![false; n];
    let mut used = 0usize;
    let take = |i: usize, keep: &mut Vec<bool>, used: &mut usize, cap: usize| -> bool {
        if i >= n || keep[i] {
            return true;
        }
        let cost = lines[i].len().min(300) + 1;
        if *used + cost > cap {
            return false;
        }
        keep[i] = true;
        *used += cost;
        true
    };
    // Head: what was run.
    for i in 0..n.min(3) {
        take(i, &mut keep, &mut used, max / 8);
    }
    // Signals, up to two thirds of the room.
    let signal_cap = max * 2 / 3;
    'sig: for i in 0..n {
        if is_signal(lines[i]) {
            for j in [i.saturating_sub(1), i, i + 1] {
                if !take(j, &mut keep, &mut used, signal_cap) {
                    break 'sig;
                }
            }
        }
    }
    // Tail: summaries and the last error usually sit at the end.
    let mut i = n;
    while i > 0 {
        i -= 1;
        if !take(i, &mut keep, &mut used, max) {
            break;
        }
    }
    let mut out = String::new();
    let mut gap = false;
    for i in 0..n {
        if keep[i] {
            if gap {
                out.push_str("…\n");
                gap = false;
            }
            let l = lines[i];
            if l.len() > 300 {
                let mut e = 300;
                while !l.is_char_boundary(e) {
                    e -= 1;
                }
                out.push_str(&l[..e]);
                out.push('…');
            } else {
                out.push_str(l);
            }
            out.push('\n');
        } else {
            gap = true;
        }
    }
    if gap {
        out.push_str("…\n");
    }
    out
}

/// The `recall` tool: lines `a-b` of a saved result, or the lines around matches of `find`.
pub fn recall(id: &str, lines: Option<&str>, find: Option<&str>, max: usize) -> Result<(String, String), String> {
    let Some((title, text)) = get(id) else {
        return Err(format!("no saved output with id '{id}' (ids look like v3 and appear in the note under a shortened result)"));
    };
    let all: Vec<&str> = text.lines().collect();
    let numbered = |a: usize, b: usize| -> String {
        let mut s = String::new();
        for (i, l) in all.iter().enumerate().take(b).skip(a) {
            s.push_str(&format!("{:>5}\t{l}\n", i + 1));
        }
        s
    };
    let body = if let Some(f) = find.filter(|f| !f.trim().is_empty()) {
        let needle = f.to_lowercase();
        let mut s = String::new();
        let mut hits = 0;
        let mut last_end = 0usize;
        for (i, l) in all.iter().enumerate() {
            if l.to_lowercase().contains(&needle) {
                hits += 1;
                let (a, b) = (i.saturating_sub(2).max(last_end), (i + 3).min(all.len()));
                if a > last_end && !s.is_empty() {
                    s.push_str("…\n");
                }
                s.push_str(&numbered(a, b));
                last_end = b;
                if s.len() > max {
                    break;
                }
            }
        }
        if hits == 0 {
            format!("'{f}' does not appear in {id} ({} lines).", all.len())
        } else {
            s
        }
    } else {
        let (a, b) = parse_range(lines.unwrap_or("1-80"), all.len());
        let mut s = numbered(a, b);
        if b < all.len() {
            s.push_str(&format!("(lines {}-{b} of {}; ask for \"{}-{}\" to continue)\n", a + 1, all.len(), b + 1, (b + 80).min(all.len())));
        }
        s
    };
    Ok((title, crate::tools::truncate_middle(&body, max)))
}

/// `"120-180"`, `"120"`, `"120-"` → 0-based `[a, b)`.
fn parse_range(r: &str, n: usize) -> (usize, usize) {
    let r = r.trim();
    let (a, b) = match r.split_once('-') {
        Some((a, b)) => (a.trim().parse().unwrap_or(1usize), b.trim().parse().unwrap_or(n)),
        None => {
            let a = r.parse().unwrap_or(1usize);
            (a, a + 79)
        }
    };
    let a = a.max(1).min(n.max(1)) - 1;
    (a, b.max(a + 1).min(n))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cargo_log() -> String {
        let mut s = String::from("   Compiling spindle v1.1.0\n");
        for i in 0..400 {
            s.push_str(&format!("   Compiling dep{i} v0.1.0\n"));
        }
        s.push_str("error[E0308]: mismatched types\n  --> src/main.rs:42:9\n   |\n42 |     let x: u32 = \"a\";\n   |                  ^^^ expected `u32`, found `&str`\n");
        for i in 0..200 {
            s.push_str(&format!("note line {i}\n"));
        }
        s.push_str("error: could not compile `spindle` due to 1 previous error\n");
        s
    }

    #[test]
    fn digest_keeps_errors_locations_and_the_end() {
        let log = cargo_log();
        let d = digest_log(&log, 1500);
        assert!(d.len() <= 1700, "digest too long: {}", d.len());
        assert!(d.contains("error[E0308]: mismatched types"));
        assert!(d.contains("src/main.rs:42:9"));
        assert!(d.contains("could not compile"));
        assert!(d.contains("Compiling spindle"));
        assert!(!d.contains("dep200"), "noise kept");
    }

    #[test]
    fn fit_saves_and_recall_finds_it() {
        let log = cargo_log();
        let short = fit("cargo build", &log, 1500, Kind::Log);
        let id = short.split("saved as ").nth(1).and_then(|s| s.split('.').next()).unwrap().to_string();
        assert!(id.starts_with('v'));
        let (_, around) = recall(&id, None, Some("mismatched"), 4000).unwrap();
        assert!(around.contains("mismatched types") && around.contains("src/main.rs:42:9"));
        let (_, slice) = recall(&id, Some("2-4"), None, 4000).unwrap();
        assert!(slice.contains("dep0") && slice.contains("dep2") && !slice.contains("dep3 "));
        assert!(recall("v999999", None, None, 100).is_err());
        // Short output is returned as is and not stored.
        assert_eq!(fit("echo", "hello", 100, Kind::Log), "hello");
    }

    #[test]
    fn ranges() {
        assert_eq!(parse_range("120-180", 1000), (119, 180));
        assert_eq!(parse_range("5", 1000), (4, 84));
        assert_eq!(parse_range("990-", 1000), (989, 1000));
        assert_eq!(parse_range("0-0", 3), (0, 1));
    }
}
