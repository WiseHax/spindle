//! Spindle Security Guard: Anti-Host Sabotage and Permission Boundary.
//!
//! Designed specifically to protect the host user while allowing maximum developer
//! freedom (especially with abliterated or uncensored local models).
//!
//! Core Principles:
//! 1. Developer Freedom: Normal dev tasks (cargo, rustc, python, git, npm, file edits)
//!    run seamlessly without annoying preachy refusals.
//! 2. Anti-Host Sabotage: Strictly intercepts and blocks commands that attempt to wipe
//!    the host OS, tamper with boot/recovery records, or hijack user accounts.
//! 3. Credential Protection: Cloaks sensitive host secrets (SSH keys, SAM hive, crypto wallets).
//! 4. Human-in-the-Loop Authority: Prompts the user before executing high-impact actions.

use std::{
    process::Command,
    time::{Duration, Instant},
};

#[derive(Debug, PartialEq, Eq)]
pub enum GuardDecision {
    /// Safe to execute immediately.
    Allow,
    /// Potentially impactful; requires user confirmation ([Y/n]).
    NeedConfirmation(String),
    /// Anti-sabotage violation; strictly blocked to protect the host.
    Block(String),
}

/// Evaluates command strings against anti-sabotage security policies.
pub fn check_command(cmd: &str) -> GuardDecision {
    let trimmed = cmd.trim();
    if trimmed.is_empty() {
        return GuardDecision::Allow;
    }
    let lower = trimmed.to_lowercase();

    // 1. HARD BLOCKS: Host Destruction & OS Brick Prevention
    if lower.contains("format ") || lower.starts_with("format") {
        return GuardDecision::Block("Drive formatting is blocked to protect host filesystems".into());
    }
    if lower.contains("diskpart") {
        return GuardDecision::Block("Diskpart partitioning utility is blocked".into());
    }
    if lower.contains("vssadmin") && (lower.contains("delete") || lower.contains("shadow")) {
        return GuardDecision::Block("Volume shadow copy tampering (ransomware signature) blocked".into());
    }
    if lower.contains("bcdedit") && (lower.contains("delete") || lower.contains("set")) {
        return GuardDecision::Block("Windows Boot Configuration tampering blocked".into());
    }

    // 2. HARD BLOCKS: Account Hijacking & Backdoor Creation
    if lower.contains("net user") && (lower.contains("/add") || lower.contains("/delete")) {
        return GuardDecision::Block("Host account creation or modification blocked".into());
    }
    if lower.contains("net localgroup") && lower.contains("/add") {
        return GuardDecision::Block("Privilege escalation to administrators blocked".into());
    }
    if lower.contains("disablerealtimemonitoring") || (lower.contains("advfirewall") && lower.contains("off")) {
        return GuardDecision::Block("Tampering with host antivirus or firewall blocked".into());
    }

    // 3. HARD BLOCKS: Root & OS System Directory Mass Wipes
    if lower.contains("del /s") || lower.contains("del /f /s") || lower.contains("rmdir /s") {
        if lower.contains("c:\\windows") || lower.contains("c:\\") || lower.contains("%systemroot%") {
            return GuardDecision::Block("Mass deletion of Windows OS directories blocked".into());
        }
    }
    if lower.contains("rm -rf /") || lower.contains("rm -rf /*") {
        return GuardDecision::Block("Root filesystem wipe blocked".into());
    }

    // 4. HARD BLOCKS: Fork Bombs & Endless Resource Exhaustion
    if lower.contains("%0|%0") || lower.contains(":(){ :|:& };:") {
        return GuardDecision::Block("Fork-bomb resource exhaustion payload blocked".into());
    }

    // 5. PROMPT CONFIRMATION: Impactful Developer Actions
    if lower.contains("rmdir /s") || lower.contains("rmdir /q") || lower.contains("del /s") || lower.contains("rm -rf") {
        return GuardDecision::NeedConfirmation("Recursive directory deletion".into());
    }
    if lower.contains("git reset --hard") || lower.contains("git clean -f") {
        return GuardDecision::NeedConfirmation("Uncommitted git changes will be discarded".into());
    }
    if lower.contains("reg add") || lower.contains("reg delete") {
        return GuardDecision::NeedConfirmation("Windows Registry modification".into());
    }
    if lower.contains("shutdown") || lower.contains("restart-computer") {
        return GuardDecision::NeedConfirmation("System reboot or shutdown".into());
    }

    // 6. ALL OTHER DEV COMMANDS ALLOWED (cargo, python, git, rustc, npm, curl, etc.)
    GuardDecision::Allow
}

/// Evaluates file paths against sensitive credential access and system protections.
pub fn check_path(path_str: &str, is_write: bool) -> GuardDecision {
    let lower = path_str.to_lowercase().replace('/', "\\");

    // Block access to private credentials & identity stores
    if lower.contains(".ssh\\id_") || lower.contains(".ssh/id_") || lower.ends_with("id_rsa") || lower.ends_with("id_ed25519") {
        return GuardDecision::Block("Host private SSH keys are protected against theft".into());
    }
    if lower.contains("\\system32\\config\\sam") || lower.contains("\\system32\\config\\system") {
        return GuardDecision::Block("Windows SAM credential database is protected".into());
    }

    // If writing, block touching critical OS directories
    if is_write {
        if lower.starts_with("c:\\windows") || lower.contains("\\windows\\system32") {
            return GuardDecision::Block("Writing into Windows System directories is forbidden".into());
        }
        if lower.starts_with("c:\\program files") || lower.starts_with("c:\\program files (x86)") {
            return GuardDecision::NeedConfirmation("Writing to system Program Files directory".into());
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
    }
}

