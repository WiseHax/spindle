<#
.SYNOPSIS
  Runs spindle's fixed benchmark and saves the result with the build and machine details.

.DESCRIPTION
  Calls `spindle --bench <Runs>` in the current folder (so spindle.toml applies: model,
  worker, layer range, threads), then writes the output to bench-logs\<date>-<commit>.txt
  together with the git commit, CPU, free RAM and the exact command. Compare two logs to
  see what a change did.

.EXAMPLE
  .\tools\bench.ps1
  .\tools\bench.ps1 -Runs 5 -Exe .\spindle.exe -Extra "--no-spec"
#>
param(
    [int]$Runs = 3,
    [string]$Exe = ".\target\release\spindle.exe",
    [string]$Extra = ""
)

$ErrorActionPreference = "Stop"
if (-not (Test-Path $Exe)) { throw "spindle not found at $Exe. Build it with: cargo build --release" }

$commit = (git rev-parse --short HEAD 2>$null)
if (-not $commit) { $commit = "nogit" }
$dirty = if (git status --porcelain 2>$null | Select-String -NotMatch '^\?\?') { "+changes" } else { "" }
$stamp = Get-Date -Format "yyyy-MM-dd_HHmm"
New-Item -ItemType Directory -Force bench-logs | Out-Null
$log = "bench-logs\$stamp-$commit$dirty.txt"

$cpu = (Get-CimInstance Win32_Processor | Select-Object -First 1).Name.Trim()
$os = Get-CimInstance Win32_OperatingSystem
$freeMiB = [math]::Round($os.FreePhysicalMemory / 1024)
$totalMiB = [math]::Round($os.TotalVisibleMemorySize / 1024)

$header = @(
    "spindle benchmark  $stamp",
    "commit   $commit$dirty",
    "binary   $Exe  ($(& $Exe --version))",
    "cpu      $cpu",
    "ram      $freeMiB MiB free of $totalMiB MiB",
    "command  $Exe --bench $Runs $Extra",
    ""
)
$header | Set-Content -Encoding utf8 $log
$header | ForEach-Object { Write-Host $_ }

$argsList = @("--bench", "$Runs") + ($Extra -split ' ' | Where-Object { $_ })
& $Exe @argsList 2>&1 | Tee-Object -FilePath $log -Append

Write-Host ""
Write-Host "Saved to $log"
