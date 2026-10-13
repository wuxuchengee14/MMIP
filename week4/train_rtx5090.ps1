param(
    [ValidateSet('all','prepare','sentiment','vision','caption','gemini','report')]
    [string]$Stage = 'all',
    [switch]$SkipGemini,
    [switch]$Install,
    [switch]$DryRun
)
$ErrorActionPreference = 'Stop'
$taskPython = $env:MMIP_PYTHON
if (-not $taskPython -and $env:CONDA_PREFIX) {
    $candidate = Join-Path $env:CONDA_PREFIX 'python.exe'
    if (Test-Path -LiteralPath $candidate) { $taskPython = $candidate }
}
if (-not $taskPython) {
    $taskPython = (Get-Command python -CommandType Application | Select-Object -First 1).Source
}
$taskConfig = $env:MMIP_CONFIG
if (-not $taskConfig) { $taskConfig = Join-Path $PSScriptRoot 'config.rtx5090.json' }
$taskArguments = @('-u', (Join-Path $PSScriptRoot 'run_all.py'), '--config', $taskConfig,
                   '--device', 'cuda', '--stage', $Stage)
if ($SkipGemini) { $taskArguments += '--skip-gemini' }
Write-Host "Interpreter: $taskPython"
Write-Host "Stage: $Stage"
if ($DryRun) {
    Write-Host ($taskArguments -join ' ')
    Write-Host 'Dry run: no installation, downloads, API calls or training.'
    exit 0
}
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONUTF8 = '1'
if (-not $env:OMP_NUM_THREADS) { $env:OMP_NUM_THREADS = '8' }
if (-not $env:MKL_NUM_THREADS) { $env:MKL_NUM_THREADS = '8' }
if ($Install) {
    & $taskPython -m pip install -r (Join-Path $PSScriptRoot 'requirements.txt')
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
& $taskPython (Join-Path $PSScriptRoot 'check_gpu.py')
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$taskLogs = Join-Path $PSScriptRoot 'logs'
New-Item -ItemType Directory -Path $taskLogs -Force | Out-Null
$taskLog = Join-Path $taskLogs ($Stage + '_' + (Get-Date -Format 'yyyyMMdd_HHmmss') + '.log')
# Native stderr may be normal progress/warnings under Windows PowerShell 5.1.
$ErrorActionPreference = 'Continue'
& $taskPython @taskArguments 2>&1 | Tee-Object -FilePath $taskLog
$taskExit = $LASTEXITCODE
$ErrorActionPreference = 'Stop'
if ($taskExit -eq 3) { Write-Host 'Results saved; requirements remain pending. See completion.json.' }
exit $taskExit
