# Compares spindle against llama.cpp on the same model with greedy decoding.
#
#   .\tools\golden.ps1 -Model qwen3-0.6b-q8_0.gguf -Llama C:\llama\llama-completion.exe
#
# Both engines get the same raw prompts (no chat template), temperature 0 and no repetition
# penalty. The script prints where the first difference is.
#
# Exact matches aren't guaranteed: llama.cpp quantizes activations to 8 bits before its dot
# products, spindle keeps them in f32, so when two tokens are nearly tied the engines can pick
# differently and the texts drift apart from there. To check a difference, rerun spindle with
# -v: it prints the top two candidates and their logit gap at every step. If llama.cpp's token
# is spindle's second choice with a gap well under 1, it's rounding; anything else is a bug.

param(
    [Parameter(Mandatory)] [string] $Model,
    [Parameter(Mandatory)] [string] $Llama,
    [int] $Tokens = 32,
    [string] $Spindle = ".\target\release\spindle.exe",
    # Extra flags for llama.cpp. -no-cnv stops it from applying the chat template, which it
    # otherwise does by default.
    [string[]] $LlamaArgs = @("--no-display-prompt", "-no-cnv")
)

foreach ($exe in @($Llama, $Spindle)) {
    if (-not (Get-Command $exe -ErrorAction SilentlyContinue)) {
        Write-Host "not found: $exe"
        exit 2
    }
}
if (-not (Test-Path $Model)) {
    Write-Host "model not found: $Model"
    exit 2
}

$prompts = @(
    "The capital of France is",
    "def fibonacci(n):",
    "Once upon a time, in a small village,",
    "The three primary colors are"
)

function Clean([string] $s) {
    ($s -replace '\[end of text\]', '' -replace '\s+', ' ').Trim()
}

$failed = 0
foreach ($p in $prompts) {
    $ref = & $Llama -m $Model -p $p -n $Tokens --temp 0 --repeat-penalty 1 @LlamaArgs 2>$null | Out-String
    $got = & $Spindle $Model --raw $p -n $Tokens --temp 0 --repeat-penalty 1 2>$null | Out-String
    $ref = Clean $ref
    $got = Clean $got
    if ($ref.StartsWith($p)) { $ref = $ref.Substring($p.Length).Trim() }

    if ($ref -eq $got) {
        Write-Host "match   $p"
        continue
    }
    $failed++
    $n = [Math]::Min($ref.Length, $got.Length)
    $i = 0
    while ($i -lt $n -and $ref[$i] -eq $got[$i]) { $i++ }
    Write-Host "DIFFER  $p"
    Write-Host "  llama.cpp: $ref"
    Write-Host "  spindle:   $got"
    Write-Host "  first difference at character $i"
}

if ($failed) {
    Write-Host "$failed of $($prompts.Count) prompts differ"
    exit 1
}
Write-Host "all $($prompts.Count) prompts match"
