# Start Ollama (if needed), pull Qwen, print config hints.
# Requires: Ollama installed from https://ollama.com/download
#
# Usage:
#   .\scripts\start_ollama_qwen.ps1
#   .\scripts\start_ollama_qwen.ps1 -Model "qwen2.5:14b"
#   .\scripts\start_ollama_qwen.ps1 -SkipPull

param(
    [string]$Model = "qwen2.5:7b",
    [switch]$SkipPull,
    [switch]$SkipServe
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) {
    $Py = "python"
}

$args = @("$PSScriptRoot\start_ollama_qwen.py", "--model", $Model)
if ($SkipPull) { $args += "--skip-pull" }
if ($SkipServe) { $args += "--skip-serve" }

& $Py @args
exit $LASTEXITCODE
