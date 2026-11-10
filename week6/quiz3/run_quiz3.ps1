param([string]$PythonExecutable = "python")
$ErrorActionPreference = "Stop"
& $PythonExecutable -X utf8 (Join-Path $PSScriptRoot "Code/run_all.py")
if ($LASTEXITCODE -ne 0) { throw "Quiz 3 stopped with exit code $LASTEXITCODE. See the preceding error." }
