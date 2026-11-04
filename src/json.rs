//! Minimal JSON reader/writer for tool calls (zero dependencies).
//!
//! Strict JSON plus the slips small models make: single-quoted strings, trailing commas,
//! bare `True`/`False`/`None`, and raw newlines inside strings.

use std::fmt::Write as _;

#[derive(Clone, Debug, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    /// Insertion-ordered object.
    Obj(Vec<(String, Json)>),
}

impl Json {
    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(kv) => kv.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Json::Str(s) => Some(s),
            _ => None,
        }
    }

    /// Numbers, or numeric strings (`"10"`), as a non-negative integer.
    pub fn as_usize(&self) -> Option<usize> {
        match self {
            Json::Num(n) if *n >= 0.0 => Some(*n as usize),
            Json::Str(s) => s.trim().parse().ok(),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Json::Bool(b) => Some(*b),
            Json::Str(s) => match s.to_ascii_lowercase().as_str() {
                "true" | "yes" | "1" => Some(true),
                "false" | "no" | "0" => Some(false),
                _ => None,
            },
            _ => None,
        }
    }

    /// Compact JSON text.
    pub fn to_string_compact(&self) -> String {
        let mut out = String::new();
        self.write(&mut out);
        out
    }

    fn write(&self, out: &mut String) {
        match self {
            Json::Null => out.push_str("null"),
            Json::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Json::Num(n) => {
                if n.fract() == 0.0 && n.abs() < 1e15 {
                    let _ = write!(out, "{}", *n as i64);
                } else {
                    let _ = write!(out, "{n}");
                }
            }
            Json::Str(s) => write_str(s, out),
            Json::Arr(a) => {
                out.push('[');
                for (i, v) in a.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    v.write(out);
                }
                out.push(']');
            }
            Json::Obj(kv) => {
                out.push('{');
                for (i, (k, v)) in kv.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    write_str(k, out);
                    out.push(':');
                    v.write(out);
                }
                out.push('}');
            }
        }
    }
}

fn write_str(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Parses the first JSON value in `text` (leading whitespace allowed). Returns the value and
/// the number of bytes consumed.
pub fn parse_prefix(text: &str) -> Result<(Json, usize), String> {
    let mut p = Parser { s: text.as_bytes(), src: text, i: 0 };
    p.ws();
    let v = p.value(0)?;
    Ok((v, p.i))
}

/// Parses `text` as exactly one JSON value (trailing whitespace allowed).
pub fn parse(text: &str) -> Result<Json, String> {
    let (v, used) = parse_prefix(text)?;
    if !text[used..].trim().is_empty() {
        return Err(format!("unexpected trailing text at byte {used}"));
    }
    Ok(v)
}

struct Parser<'a> {
    s: &'a [u8],
    src: &'a str,
    i: usize,
}

impl Parser<'_> {
    fn ws(&mut self) {
        while self.i < self.s.len() && self.s[self.i].is_ascii_whitespace() {
            self.i += 1;
        }
    }

    fn err<T>(&self, what: &str) -> Result<T, String> {
        Err(format!("{what} at byte {}", self.i))
    }

    fn value(&mut self, depth: usize) -> Result<Json, String> {
        if depth > 64 {
            return self.err("nesting too deep");
        }
        self.ws();
        match self.s.get(self.i) {
            None => self.err("unexpected end of input"),
            Some(b'{') => self.object(depth),
            Some(b'[') => self.array(depth),
            Some(b'"') | Some(b'\'') => Ok(Json::Str(self.string()?)),
            Some(b't') | Some(b'T') if self.word("true") => Ok(Json::Bool(true)),
            Some(b'f') | Some(b'F') if self.word("false") => Ok(Json::Bool(false)),
            Some(b'n') | Some(b'N') if self.word("null") || self.word("none") => Ok(Json::Null),
            Some(c) if *c == b'-' || c.is_ascii_digit() => self.number(),
            Some(_) => self.err("unexpected character"),
        }
    }

    fn word(&mut self, w: &str) -> bool {
        let end = self.i + w.len();
        if end <= self.s.len() && self.s[self.i..end].eq_ignore_ascii_case(w.as_bytes()) {
            self.i = end;
            true
        } else {
            false
        }
    }

    fn number(&mut self) -> Result<Json, String> {
        let start = self.i;
        if self.s[self.i] == b'-' {
            self.i += 1;
        }
        while self.i < self.s.len() && matches!(self.s[self.i], b'0'..=b'9' | b'.' | b'e' | b'E' | b'+' | b'-') {
            self.i += 1;
        }
        self.src[start..self.i].parse::<f64>().map(Json::Num).or_else(|_| self.err("bad number"))
    }

    fn string(&mut self) -> Result<String, String> {
        let quote = self.s[self.i];
        self.i += 1;
        let mut out = String::new();
        loop {
            let Some(&b) = self.s.get(self.i) else {
                return self.err("unterminated string");
            };
            if b == quote {
                self.i += 1;
                return Ok(out);
            }
            if b == b'\\' {
                self.i += 1;
                let Some(&e) = self.s.get(self.i) else {
                    return self.err("unterminated escape");
                };
                self.i += 1;
                match e {
                    b'n' => out.push('\n'),
                    b't' => out.push('\t'),
                    b'r' => out.push('\r'),
                    b'b' => out.push('\u{8}'),
                    b'f' => out.push('\u{c}'),
                    b'/' => out.push('/'),
                    b'u' => {
                        let cp = self.hex4()?;
                        if (0xD800..0xDC00).contains(&cp) && self.s.get(self.i) == Some(&b'\\') && self.s.get(self.i + 1) == Some(&b'u') {
                            self.i += 2;
                            let lo = self.hex4()?;
                            let c = 0x10000 + ((cp - 0xD800) << 10) + (lo.wrapping_sub(0xDC00) & 0x3FF);
                            out.push(char::from_u32(c).unwrap_or('\u{FFFD}'));
                        } else {
                            out.push(char::from_u32(cp).unwrap_or('\u{FFFD}'));
                        }
                    }
                    // `\"`, `\\`, `\'` and anything unknown: keep the character itself.
                    _ => {
                        let ch_len = utf8_len(e);
                        out.push_str(&self.src[self.i - 1..(self.i - 1 + ch_len).min(self.s.len())]);
                        self.i += ch_len - 1;
                    }
                }
                continue;
            }
            // Copy one UTF-8 character (raw control characters are tolerated).
            let len = utf8_len(b);
            let end = (self.i + len).min(self.s.len());
            out.push_str(&self.src[self.i..end]);
            self.i = end;
        }
    }

    fn hex4(&mut self) -> Result<u32, String> {
        let end = self.i + 4;
        if end > self.s.len() {
            return self.err("short \\u escape");
        }
        let v = u32::from_str_radix(&self.src[self.i..end], 16).or_else(|_| self.err("bad \\u escape"))?;
        self.i = end;
        Ok(v)
    }

    fn array(&mut self, depth: usize) -> Result<Json, String> {
        self.i += 1;
        let mut out = Vec::new();
        loop {
            self.ws();
            match self.s.get(self.i) {
                Some(b']') => {
                    self.i += 1;
                    return Ok(Json::Arr(out));
                }
                None => return self.err("unterminated array"),
                _ => {}
            }
            out.push(self.value(depth + 1)?);
            self.ws();
            match self.s.get(self.i) {
                Some(b',') => self.i += 1,
                Some(b']') => {}
                _ => return self.err("expected ',' or ']'"),
            }
        }
    }

    fn object(&mut self, depth: usize) -> Result<Json, String> {
        self.i += 1;
        let mut out = Vec::new();
        loop {
            self.ws();
            match self.s.get(self.i) {
                Some(b'}') => {
                    self.i += 1;
                    return Ok(Json::Obj(out));
                }
                Some(b'"') | Some(b'\'') => {}
                None => return self.err("unterminated object"),
                _ => return self.err("expected a key"),
            }
            let key = self.string()?;
            self.ws();
            if self.s.get(self.i) != Some(&b':') {
                return self.err("expected ':'");
            }
            self.i += 1;
            let v = self.value(depth + 1)?;
            out.push((key, v));
            self.ws();
            match self.s.get(self.i) {
                Some(b',') => self.i += 1,
                Some(b'}') => {}
                _ => return self.err("expected ',' or '}'"),
            }
        }
    }
}

fn utf8_len(first: u8) -> usize {
    match first {
        0xC0..=0xDF => 2,
        0xE0..=0xEF => 3,
        0xF0..=0xF7 => 4,
        _ => 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_strict_and_lenient_json() {
        let v = parse(r#"{"name": "edit_file", "arguments": {"path": "a.rs", "n": 3, "ok": true, "x": null, "l": [1, 2.5]}}"#).unwrap();
        let a = v.get("arguments").unwrap();
        assert_eq!(a.get("path").and_then(Json::as_str), Some("a.rs"));
        assert_eq!(a.get("n").and_then(Json::as_usize), Some(3));
        assert_eq!(a.get("ok").and_then(Json::as_bool), Some(true));

        // Escapes: quotes, newlines, unicode, surrogate pairs; code fences inside content.
        let v = parse(r#"{"content": "say \"hi\"\n```rust\nfn main() {}\n```\nñ 😀"}"#).unwrap();
        assert_eq!(v.get("content").and_then(Json::as_str), Some("say \"hi\"\n```rust\nfn main() {}\n```\nñ 😀"));

        // Small-model slips: single quotes, trailing commas, Python literals, raw newline.
        let v = parse("{'path': 'x.py', 'flag': True, 'z': None, 'body': 'a\nb',}").unwrap();
        assert_eq!(v.get("path").and_then(Json::as_str), Some("x.py"));
        assert_eq!(v.get("flag").and_then(Json::as_bool), Some(true));
        assert_eq!(v.get("body").and_then(Json::as_str), Some("a\nb"));
    }

    #[test]
    fn round_trips_through_writer() {
        let src = r#"{"a":"line1\nline2 \"q\" \\","b":[1,2.5,true,null],"c":{}}"#;
        let v = parse(src).unwrap();
        assert_eq!(parse(&v.to_string_compact()).unwrap(), v);
        assert_eq!(v.to_string_compact(), src);
    }
}
