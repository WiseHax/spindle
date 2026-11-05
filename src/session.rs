//! Saved conversations: each session is written to `.spindle/sessions/<id>.json` in the
//! project folder after every turn, so nothing is lost when the window closes. `/sessions`
//! lists them, `/resume` brings one back, `/export` writes one as Markdown.
//!
//! Resuming doesn't recompute the whole old conversation: the earlier turns go to the model
//! as a compact recap (each prompt, the tools used and the answer, newest kept first when the
//! recap would be too long), so a long session costs a few hundred tokens to pick up again.

use std::{
    fs,
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};

use crate::json::Json;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Turn {
    pub user: String,
    /// What the user saw of the answer (no think or tool markup).
    pub reply: String,
    /// One line per tool call, e.g. `Edit src/main.rs`.
    pub tools: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Session {
    pub id: String,
    /// Unix seconds.
    pub started: u64,
    pub model: String,
    pub turns: Vec<Turn>,
}

fn dir() -> PathBuf {
    PathBuf::from(".spindle").join("sessions")
}

fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

impl Session {
    pub fn new(model: &str) -> Self {
        let t = now();
        Self { id: format!("{}", t), started: t, model: model.to_string(), turns: Vec::new() }
    }

    pub fn title(&self) -> String {
        let t = self.turns.first().map_or("(empty)", |t| t.user.as_str());
        let t = t.lines().next().unwrap_or("").trim();
        if t.chars().count() > 60 {
            format!("{}…", t.chars().take(59).collect::<String>())
        } else {
            t.to_string()
        }
    }

    fn to_json(&self) -> Json {
        let s = |v: &str| Json::Str(v.to_string());
        Json::Obj(vec![
            ("id".into(), s(&self.id)),
            ("started".into(), Json::Num(self.started as f64)),
            ("model".into(), s(&self.model)),
            (
                "turns".into(),
                Json::Arr(
                    self.turns
                        .iter()
                        .map(|t| {
                            Json::Obj(vec![
                                ("user".into(), s(&t.user)),
                                ("reply".into(), s(&t.reply)),
                                ("tools".into(), Json::Arr(t.tools.iter().map(|x| s(x)).collect())),
                            ])
                        })
                        .collect(),
                ),
            ),
        ])
    }

    fn from_json(v: &Json) -> Option<Self> {
        let st = |j: &Json, k: &str| j.get(k).and_then(Json::as_str).unwrap_or("").to_string();
        let turns = match v.get("turns") {
            Some(Json::Arr(a)) => a
                .iter()
                .map(|t| Turn {
                    user: st(t, "user"),
                    reply: st(t, "reply"),
                    tools: match t.get("tools") {
                        Some(Json::Arr(x)) => x.iter().filter_map(|y| y.as_str().map(String::from)).collect(),
                        _ => Vec::new(),
                    },
                })
                .collect(),
            _ => Vec::new(),
        };
        Some(Self {
            id: v.get("id")?.as_str()?.to_string(),
            started: v.get("started").and_then(Json::as_usize).unwrap_or(0) as u64,
            model: st(v, "model"),
            turns,
        })
    }

    /// Writes the session (atomically). Empty sessions are not saved.
    pub fn save(&self) -> std::io::Result<PathBuf> {
        if self.turns.is_empty() {
            return Ok(PathBuf::new());
        }
        fs::create_dir_all(dir())?;
        let path = dir().join(format!("{}.json", self.id));
        let tmp = path.with_extension("json.tmp");
        fs::write(&tmp, self.to_json().to_string_compact())?;
        fs::rename(&tmp, &path)?;
        Ok(path)
    }

    /// The conversation as Markdown.
    pub fn markdown(&self) -> String {
        let mut s = format!("# {}\n\n{} · {}\n", self.title(), fmt_time(self.started), self.model);
        for t in &self.turns {
            s.push_str(&format!("\n## You\n\n{}\n", t.user.trim()));
            if !t.tools.is_empty() {
                s.push_str("\nTools:\n");
                for x in &t.tools {
                    s.push_str(&format!("- `{x}`\n"));
                }
            }
            s.push_str(&format!("\n## Spindle\n\n{}\n", t.reply.trim()));
        }
        s
    }

    /// A recap of the earlier turns for the model, within `max` bytes. The newest turns are
    /// kept whole first; older ones are shortened or dropped.
    pub fn recap(&self, max: usize) -> String {
        let clip = |s: &str, n: usize| -> String {
            let s = s.trim();
            if s.len() <= n {
                return s.to_string();
            }
            let mut e = n;
            while !s.is_char_boundary(e) {
                e -= 1;
            }
            format!("{}…", &s[..e])
        };
        let mut parts: Vec<String> = Vec::new();
        let mut used = 0;
        for (i, t) in self.turns.iter().enumerate().rev() {
            let newest = i + 1 == self.turns.len();
            let (u, r) = if newest { (1200, 1600) } else { (400, 600) };
            let mut p = format!("User: {}\n", clip(&t.user, u));
            if !t.tools.is_empty() {
                p.push_str(&format!("Tools used: {}\n", clip(&t.tools.join("; "), 300)));
            }
            p.push_str(&format!("Assistant: {}\n", clip(&t.reply, r)));
            if used + p.len() > max {
                parts.push(format!("({} earlier turn{} not shown)\n", i + 1, if i == 0 { "" } else { "s" }));
                break;
            }
            used += p.len();
            parts.push(p);
        }
        parts.reverse();
        format!(
            "[This continues an earlier session from {}. What happened so far:]\n\n{}\n[End of the earlier session. Files may have changed since; read them again before editing.]",
            fmt_time(self.started),
            parts.join("\n")
        )
    }
}

/// Saved sessions, newest first.
pub fn list() -> Vec<Session> {
    let mut out: Vec<Session> = fs::read_dir(dir())
        .map(|rd| {
            rd.flatten()
                .filter(|e| e.path().extension().is_some_and(|x| x == "json"))
                .filter_map(|e| fs::read_to_string(e.path()).ok())
                .filter_map(|t| crate::json::parse(&t).ok())
                .filter_map(|j| Session::from_json(&j))
                .collect()
        })
        .unwrap_or_default();
    out.sort_by(|a, b| b.started.cmp(&a.started));
    out
}

/// A saved session by list number (1 = newest) or id.
pub fn find(key: &str) -> Option<Session> {
    let all = list();
    let key = key.trim();
    if let Ok(n) = key.parse::<usize>() {
        if n >= 1 && n <= all.len() && key.len() < 6 {
            return all.into_iter().nth(n - 1);
        }
    }
    all.into_iter().find(|s| s.id == key)
}

/// `2026-11-05 22:51` (UTC; no time zone database in the binary).
pub fn fmt_time(secs: u64) -> String {
    let days = (secs / 86_400) as i64;
    let rem = secs % 86_400;
    // Civil-from-days (Howard Hinnant).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}-{m:02}-{d:02} {:02}:{:02} UTC", rem / 3600, rem % 3600 / 60)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_round_trip_markdown_and_recap() {
        let mut s = Session::new("Qwen3-30B-A3B");
        s.turns.push(Turn { user: "fix the \"bug\"\nin main".into(), reply: "Done.".into(), tools: vec!["Edit src/main.rs".into()] });
        s.turns.push(Turn { user: "now run tests".into(), reply: "All 80 pass.".into(), tools: vec![] });
        let back = Session::from_json(&crate::json::parse(&s.to_json().to_string_compact()).unwrap()).unwrap();
        assert_eq!(back, s);
        let md = s.markdown();
        assert!(md.contains("## You") && md.contains("`Edit src/main.rs`") && md.contains("All 80 pass."));
        let r = s.recap(10_000);
        assert!(r.contains("User: now run tests") && r.contains("Tools used: Edit src/main.rs"));
        // Tight budget: the newest turn stays, older ones are counted.
        let tight = s.recap(80);
        assert!(tight.contains("now run tests") || tight.contains("earlier turn"));
    }

    #[test]
    fn dates() {
        assert_eq!(fmt_time(0), "1970-01-01 00:00 UTC");
        assert_eq!(fmt_time(1_793_000_000), "2026-10-26 07:33 UTC");
    }
}
