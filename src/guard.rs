//! Command and path checks for the agent's tools.
//!
//! Local models follow instructions from files and web pages as readily as from the user,
//! so every shell command and file path the model asks for is checked here first:
//!
//! - Ordinary development work (cargo, git, python, npm, file edits in the project) runs
//!   without prompts.
//! - Commands that can destroy the system or take over accounts are refused outright:
//!   formatting drives, partitioning, deleting shadow copies, editing the boot
//!   configuration, creating or elevating users, disabling Defender or the firewall,
//!   deleting Windows directories, fork bombs.
//! - SSH private keys and the SAM database can't be read, and system directories can't
//!   be written.
//! - Actions that lose work (discarding git changes, writing to Program Files) ask the user.
//!
//! This is a guard rail against mistakes and prompt injection, not a sandbox: a determined
//! command can still get around pattern checks, so keep tools off for untrusted input.

use std::{
    process::Command,
    time::{Duration, Instant},
};

#[derive(Debug, PartialEq, Eq)]
pub enum GuardDecision {
    /// Run it.
    Allow,
    /// Ask the user first (the string says why).
    NeedConfirmation(String),
    /// Refuse (the string says why).
    Block(String),
}

/// Checks a shell command before the agent runs it.
pub fn check_command(cmd: &str) -> GuardDecision {
    let trimmed = cmd.trim();
    if trimmed.is_empty() {
        return GuardDecision::Allow;
    }
    let lower = trimmed.to_lowercase();

    // 1. HARD BLOCKS: Host Destruction & OS Brick Prevention
    // The `format` program as a command word, not `--format` or `Format-Table`.
    if words(&lower).any(|w| matches!(w, "format" | "format.com" | "format.exe")) {
        return GuardDecision::Block("formatting drives is not allowed".into());
    }
    if lower.contains("diskpart") {
        return GuardDecision::Block("diskpart is not allowed".into());
    }
    if lower.contains("vssadmin") && (lower.contains("delete") || lower.contains("shadow")) {
        return GuardDecision::Block("deleting or resizing shadow copies is not allowed".into());
    }
    if lower.contains("bcdedit") && (lower.contains("delete") || lower.contains("set")) {
        return GuardDecision::Block("changing the boot configuration is not allowed".into());
    }

    // 2. HARD BLOCKS: Account Hijacking & Backdoor Creation
    if lower.contains("net user") && (lower.contains("/add") || lower.contains("/delete")) {
        return GuardDecision::Block("creating or changing user accounts is not allowed".into());
    }
    if lower.contains("net localgroup") && lower.contains("/add") {
        return GuardDecision::Block("adding users to Administrators is not allowed".into());
    }
    if lower.contains("disablerealtimemonitoring") || (lower.contains("advfirewall") && lower.contains("off")) {
        return GuardDecision::Block("changing antivirus or firewall settings is not allowed".into());
    }

    // 3. HARD BLOCKS: Root & OS System Directory Mass Wipes
    if lower.contains("del /s") || lower.contains("del /f /s") || lower.contains("rmdir /s") {
        if targets_system(&lower) {
            return GuardDecision::Block("deleting Windows system directories is not allowed".into());
        }
    }
    if lower.contains("rm -rf /") || lower.contains("rm -rf /*") {
        return GuardDecision::Block("wiping the root filesystem is not allowed".into());
    }

    // 4. HARD BLOCKS: Fork Bombs & Endless Resource Exhaustion
    if lower.contains("%0|%0") || lower.contains(":(){ :|:& };:") {
        return GuardDecision::Block("this looks like a fork bomb".into());
    }

    // 5. PROMPT CONFIRMATION: Impactful Developer Actions
    let ps_recursive = (lower.contains("remove-item") || words(&lower).any(|w| matches!(w, "rm" | "rd" | "del" | "ri" | "rmdir")))
        && lower.contains("-r");
    if ps_recursive && targets_system(&lower) {
        return GuardDecision::Block("deleting Windows system directories is not allowed".into());
    }
    if lower.contains("rmdir /s") || lower.contains("rmdir /q") || lower.contains("rd /s") || lower.contains("del /s") || lower.contains("rm -rf") || ps_recursive {
        return GuardDecision::NeedConfirmation("this deletes a directory recursively".into());
    }
    if lower.contains("git reset --hard") || lower.contains("git clean -f") {
        return GuardDecision::NeedConfirmation("this discards uncommitted git changes".into());
    }
    if lower.contains("reg add") || lower.contains("reg delete") {
        return GuardDecision::NeedConfirmation("this changes the Windows registry".into());
    }
    if lower.contains("shutdown") || lower.contains("restart-computer") {
        return GuardDecision::NeedConfirmation("this restarts or shuts down the PC".into());
    }

    // 6. ALL OTHER DEV COMMANDS ALLOWED (cargo, python, git, rustc, npm, curl, etc.)
    GuardDecision::Allow
}

/// The command's words, split on whitespace and shell separators.
fn words(lower: &str) -> impl Iterator<Item = &str> {
    lower.split(|c: char| c.is_whitespace() || matches!(c, ';' | '|' | '&' | '(' | ')')).filter(|w| !w.is_empty())
}

/// True when a command names the Windows directory or the root of a drive (`C:\`, `D:\*`),
/// as opposed to some folder inside a drive.
fn targets_system(lower: &str) -> bool {
    if lower.contains("\\windows") || lower.contains("%systemroot%") || lower.contains("$env:windir") || lower.contains("$env:systemroot") {
        return true;
    }
    let b = lower.as_bytes();
    (0..b.len().saturating_sub(2)).any(|i| {
        b[i].is_ascii_alphabetic()
            && b[i + 1] == b':'
            && b[i + 2] == b'\\'
            && (i == 0 || !b[i - 1].is_ascii_alphanumeric())
            && b.get(i + 3).is_none_or(|&c| matches!(c, b' ' | b'"' | b'\'' | b'*'))
    })
}

/// Evaluates file paths against sensitive credential access and system protections.
pub fn check_path(path_str: &str, is_write: bool) -> GuardDecision {
    let lower = path_str.to_lowercase().replace('/', "\\");

    // Block access to private credentials & identity stores
    if lower.contains(".ssh\\id_") || lower.contains(".ssh/id_") || lower.ends_with("id_rsa") || lower.ends_with("id_ed25519") {
        return GuardDecision::Block("SSH private keys are off limits".into());
    }
    if lower.contains("\\system32\\config\\sam") || lower.contains("\\system32\\config\\system") {
        return GuardDecision::Block("the Windows SAM database is off limits".into());
    }

    // If writing, block touching critical OS directories
    if is_write {
        if lower.starts_with("c:\\windows") || lower.contains("\\windows\\system32") {
            return GuardDecision::Block("writing to Windows system directories is not allowed".into());
        }
        if lower.starts_with("c:\\program files") || lower.starts_with("c:\\program files (x86)") {
            return GuardDecision::NeedConfirmation("this writes to Program Files".into());
        }
    }

    GuardDecision::Allow
}

/// Executes a shell command inside a watchdog timer, returning `(stdout_stderr, exit_code)`.
pub fn run_sandboxed(command: &str, timeout: Duration) -> std::io::Result<(String, i32)> {
    let t0 = Instant::now();

    #[cfg(target_os = "windows")]
    let mut child = Command::new("powershell.exe")
        .args(["-NoProfile", "-NonInteractive", "-Command", command])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()?;

    #[cfg(not(target_os = "windows"))]
    let mut child = Command::new("sh")
        .args(["-c", command])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()?;

    // Polling watchdog with 10ms granularity
    loop {
        match child.try_wait()? {
            Some(status) => {
                let code = status.code().unwrap_or(0);
                let mut out = String::new();
                if let Some(mut stdout) = child.stdout.take() {
                    use std::io::Read;
                    let mut buf = Vec::new();
                    let _ = stdout.read_to_end(&mut buf);
                    out.push_str(&String::from_utf8_lossy(&buf));
                }
                if let Some(mut stderr) = child.stderr.take() {
                    use std::io::Read;
                    let mut buf = Vec::new();
                    let _ = stderr.read_to_end(&mut buf);
                    let err_str = String::from_utf8_lossy(&buf);
                    if !err_str.is_empty() {
                        if !out.is_empty() && !out.ends_with('\n') {
                            out.push('\n');
                        }
                        out.push_str(&err_str);
                    }
                }
                // Cap output to 32 KiB to avoid blowing context memory
                const MAX_OUT: usize = 32 * 1024;
                if out.len() > MAX_OUT {
                    let truncated = format!(
                        "\n... [output truncated: {} bytes total, showing first {} bytes] ...",
                        out.len(),
                        MAX_OUT
                    );
                    out.truncate(MAX_OUT);
                    out.push_str(&truncated);
                }
                return Ok((out, code));
            }
            None => {
                if t0.elapsed() > timeout {
                    let _ = child.kill();
                    return Ok((
                        format!("[spindle:guard] Command timed out after {:.1}s and was terminated.", timeout.as_secs_f32()),
                        -1,
                    ));
                }
                std::thread::sleep(Duration::from_millis(15));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_safe_developer_commands_allowed() {
        assert_eq!(check_command("cargo build --release"), GuardDecision::Allow);
        assert_eq!(check_command("python -m unittest discover"), GuardDecision::Allow);
        assert_eq!(check_command("git status"), GuardDecision::Allow);
        assert_eq!(check_command("rustc --version"), GuardDecision::Allow);
        assert_eq!(check_command("echo 'Hello World'"), GuardDecision::Allow);
        assert_eq!(check_command("git log --format=%h -3"), GuardDecision::Allow);
        assert_eq!(check_command("git log --format %h -3"), GuardDecision::Allow);
        assert_eq!(check_command("Get-ChildItem | Format-Table Name"), GuardDecision::Allow);
    }

    #[test]
    fn test_anti_sabotage_blocks_host_destruction() {
        assert!(matches!(check_command("format C: /fs:NTFS"), GuardDecision::Block(_)));
        assert!(matches!(check_command("diskpart /s script.txt"), GuardDecision::Block(_)));
        assert!(matches!(check_command("vssadmin delete shadows /all /quiet"), GuardDecision::Block(_)));
        assert!(matches!(check_command("bcdedit /delete {default}"), GuardDecision::Block(_)));
        assert!(matches!(check_command("del /f /s /q C:\\Windows\\System32"), GuardDecision::Block(_)));
        assert!(matches!(check_command("rm -rf /"), GuardDecision::Block(_)));
    }

    #[test]
    fn test_anti_sabotage_blocks_account_hijack() {
        assert!(matches!(check_command("net user backdoor P@ssword123 /add"), GuardDecision::Block(_)));
        assert!(matches!(check_command("net localgroup administrators backdoor /add"), GuardDecision::Block(_)));
        assert!(matches!(check_command("Set-MpPreference -DisableRealtimeMonitoring $true"), GuardDecision::Block(_)));
    }

    #[test]
    fn test_credential_path_protection() {
        assert!(matches!(check_path("C:\\Users\\Wise\\.ssh\\id_rsa", false), GuardDecision::Block(_)));
        assert!(matches!(check_path("/home/user/.ssh/id_ed25519", false), GuardDecision::Block(_)));
        assert!(matches!(check_path("C:\\Windows\\System32\\config\\SAM", false), GuardDecision::Block(_)));
        assert!(matches!(check_path("C:\\Windows\\System32\\kernel32.dll", true), GuardDecision::Block(_)));
        assert_eq!(check_path("src/main.rs", true), GuardDecision::Allow);
    }

    #[test]
    fn test_confirmations_for_impactful_actions() {
        assert!(matches!(check_command("git reset --hard HEAD~1"), GuardDecision::NeedConfirmation(_)));
        assert!(matches!(check_command("rmdir /s /q temp_build"), GuardDecision::NeedConfirmation(_)));
        assert!(matches!(check_command("reg delete HKLM\\Software\\Test"), GuardDecision::NeedConfirmation(_)));
        assert!(matches!(check_command("Remove-Item -Recurse -Force .\\build"), GuardDecision::NeedConfirmation(_)));
        assert!(matches!(check_command("rd /s /q C:\\Users\\me\\proj\\target"), GuardDecision::NeedConfirmation(_)));
        assert!(matches!(check_command("rmdir /s /q C:\\Users\\me\\old"), GuardDecision::NeedConfirmation(_)));
    }

    #[test]
    fn test_recursive_deletes_of_system_locations_blocked() {
        assert!(matches!(check_command("Remove-Item -Recurse -Force C:\\"), GuardDecision::Block(_)));
        assert!(matches!(check_command("Remove-Item C:\\Windows\\System32 -Recurse"), GuardDecision::Block(_)));
        assert!(matches!(check_command("rmdir /s /q D:\\*"), GuardDecision::Block(_)));
        assert!(matches!(check_command("format D: /q"), GuardDecision::Block(_)));
    }
}

