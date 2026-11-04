# What the agent is allowed to do

A model with tools can run commands on your PC, and a local model follows instructions from
a file or web page as readily as from you. A README with hidden text, a poisoned search
result, or a model that simply misreads a request could all end in a destructive command.
Spindle doesn't count on the model to refuse. Every command and path goes through
`src/guard.rs` before it runs.

The guard sorts each action into one of three groups.

## Refused

These are refused outright, and the model is told why:

| Action | Examples |
|---|---|
| formatting or partitioning drives | `format D:`, `diskpart` |
| deleting shadow copies, editing the boot configuration | `vssadmin delete shadows`, `bcdedit /set` |
| recursive deletes of a drive root or the Windows folder | `rmdir /s /q C:\`, `Remove-Item -Recurse C:\Windows` |
| creating or changing user accounts, adding administrators | `net user x /add`, `net localgroup administrators x /add` |
| turning off Defender or the firewall | `Set-MpPreference -DisableRealtimeMonitoring`, `netsh advfirewall ... off` |
| fork bombs | `%0|%0` |
| reading SSH private keys or the SAM/SYSTEM hives | `~\.ssh\id_ed25519`, `System32\config\SAM` |
| writing into the Windows folder | `C:\Windows\System32\...` |

## Asks you first

These are normal things to do, but they lose work if done by mistake:

| Action | Examples |
|---|---|
| deleting any file | the `delete_file` tool |
| recursive deletes elsewhere | `Remove-Item -Recurse .\build`, `rmdir /s`, `rm -rf` |
| discarding git changes | `git reset --hard`, `git clean -f` |
| registry edits | `reg add`, `reg delete` |
| restarting or shutting down | `shutdown`, `Restart-Computer` |
| writing outside the project folder, or into Program Files | `write_file` / `edit_file` with such a path |

The full-screen interface shows a permission dialog with the action and the exact command or
path. The line-based interface asks `[y/N]`. If you decline, the model is told so and can
try something else.

## Runs without asking

Everything else: building, testing, git commands that don't discard work, reading and editing
files in the project, searching and fetching web pages. Commands run in PowerShell
(`-NoProfile -NonInteractive`) in the project folder. They are stopped after their timeout:
60 s by default, and at most 600 s if the model asks for longer.

Every file write, edit and delete keeps a copy of what was there before, so `/undo` can
restore it.

## Limits

The guard matches patterns in the command text. That stops accidents and the obvious forms
of injection, but it is not a sandbox. A command can be written in ways a pattern check won't
recognize, for example through encoding, aliases, or a script file the model writes first and
runs later. Commands run with your user account's permissions.

So:

- Turn tools off (`/tools off`) when the model will read untrusted text, such as web pages or
  repositories you don't know.
- Keep your work in git, so that whatever happens can be reviewed with `/diff` and reverted.
- Don't run spindle as administrator.
