# Compares spindle against llama.cpp on the same model with greedy decoding.
#
#   .\tools\golden.ps1 -Model qwen3-0.6b-q8_0.gguf -Llama C:\llama\llama-completion.exe
#
# Both engines get the same raw prompts (no chat template), temperature 0 and no repetition
# penalty, so their output should match. The script prints where the first difference is.
# Small differences late in a long output can come from float rounding; a mismatch in the
# first few tokens means a real bug.

param(
    [Parameter(Mandatory)] [string] $Model,
    [Parameter(Mandatory)] [string] $Llama,
    [int] $Tokens = 32,
    [string] $Spindle = ".\target\release\spindle.exe",
    # Extra flags for llama.cpp. Newer builds call the raw-completion tool llama-completion;
    # older llama-cli needs -no-cnv to skip conversation mode.
    [string[]] $LlamaArgs = @("--no-display-prompt")
)

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
