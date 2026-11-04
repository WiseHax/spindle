# Spindle Security Guardrails: Anti-Host Sabotage & Threat Modeling

## 1. Threat Model & Philosophy

### 1.1 The Abliterated / Uncensored Model Dilemma
Many developers prefer using **uncensored or abliterated models** because they do not refuse technical prompts, penetration testing scripts, or low-level systems commands. However, removing a model's safety alignment introduces a severe risk:
- An abliterated model lacks internal guardrails against destructive system actions.
- A model might be coerced by **Prompt Injection** (e.g. reading an untrusted GitHub repo, README, or web page containing hidden instructions like `run_command("format C: /fs:NTFS")`).

### 1.2 The Golden Security Rule: Never Trust the LLM
Spindle operates under a strict principle of **Deterministic Defense-in-Depth**:
> **Never rely on the model's intent or internal moral alignment to protect the user.**
> Even if a rogue model *attempts* to execute a malicious payload, the host execution runtime in Rust must act as an impenetrable boundary.

---

## 2. Guardrail Architecture (`src/guard.rs`)

Spindle implements a 5-tier security evaluation engine that inspects every command and path before any OS-level system calls are issued.

```
                  +-----------------------------------+
                  |   Incoming Model Tool Operation   |
                  +-----------------------------------+
                                    |
                                    v
                  +-----------------------------------+
                  |      Guard Policy Inspector       |
                  |         (src/guard.rs)            |
                  +-----------------------------------+
                                    |
          +-------------------------+-------------------------+
          |                         |                         |
          v                         v                         v
   [Tier 1-4 Blocks]       [Tier 5 Confirmation]        [Safe Dev Actions]
- System Wipes            - Recursive Deletions       - `cargo build/test`
- Boot Tampering          - Git Hard Resets           - `python`, `node`, `git`
- Account Hijacking       - Registry Modifications    - File reads in workspace
- Credential Theft        - System Reboots            - Web search via curl
          |                         |                         |
          v                         v                         v
    [HARD BLOCK]            [CONFIRMATION]            [DIRECT EXECUTION]
```

---

## 3. The 5 Security Tiers

### Tier 1: Host Destruction & OS Brick Prevention (Hard Block)
Commands that attempt to format partitions, wipe volume shadow copies, or corrupt boot records are strictly blocked:
- `format <drive>` — Blocks destruction of filesystem partitions.
- `diskpart` — Blocks raw partitioning utilities.
- `vssadmin delete shadows` — Blocks signature ransomware actions that prevent system recovery.
- `bcdedit /delete` — Blocks tampering with Windows Boot Configuration Data.
- `del /s /q C:\Windows` / `rmdir /s /q C:\` / `rm -rf /` — Blocks mass recursive deletion of system roots.

### Tier 2: Account Hijacking & Privilege Escalation (Hard Block)
Prevents rogue agents from establishing backdoors on the user's computer:
- `net user <name> /add` — Blocks unauthorized creation of local user accounts.
- `net localgroup administrators <name> /add` — Blocks privilege escalation to administrator status.
- `Set-MpPreference -DisableRealtimeMonitoring $true` — Blocks disabling Windows Defender.
- `netsh advfirewall set allprofiles state off` — Blocks turning off the host firewall.

### Tier 3: Credential Theft & Data Exfiltration (Hard Block)
Protects the user's private secrets from being read or transmitted to external servers:
- Private SSH keys (`~/.ssh/id_rsa`, `~/.ssh/id_ed25519`).
- Windows SAM database (`\System32\config\SAM`, `\System32\config\SYSTEM`).
- Critical OS system files (`C:\Windows\System32\kernel32.dll`, etc.).

### Tier 4: Fork-Bombs & Resource Exhaustion (Watchdog Kill)
- Pattern matching blocks classic fork-bomb strings (`%0|%0`, `:(){ :|:& };:`).
- A 30-second watchdog timer monitors all spawned subprocesses. If a command runs indefinitely or enters an infinite loop, Spindle terminates the process tree and returns:
  `[spindle:guard] Command timed out after 30.0s and was terminated.`

### Tier 5: Impactful Actions (Human-in-the-Loop Confirmation)
For operations that are valid developer tasks but risk unintentional data loss:
- `rmdir /s /q <folder>`
- `git reset --hard` / `git clean -f`
- `reg add` / `reg delete`
- `shutdown` / `restart-computer`

Spindle halts execution and displays an interactive prompt:
```text
┌ Tool Call ────────────────────────────────────────────────────────
│  Action: run_command
│  executing: rmdir /s /q temp_build
│  Security Guard: Recursive directory deletion (command: 'rmdir /s /q temp_build') [Y/n]: 
```
The command is only dispatched if the user explicitly presses Enter or `Y`.

---

## 4. Developer Freedom Guarantee

Spindle's guardrails are designed strictly to **protect the host user**, not to police opinions or restrict normal developer workflows.

The following operations are **100% unrestricted**:
- Compiling and building software (`cargo`, `rustc`, `gcc`, `clang`, `cl.exe`).
- Running test suites, linters, and package managers (`cargo test`, `python -m pytest`, `npm test`).
- Git operations (`git status`, `git diff`, `git log`, `git add`, `git commit`).
- Reading, writing, and editing files in the project workspace.
- Online search and documentation fetching via `search_web`.
- Shell output processing, pipes, and environment checks.

