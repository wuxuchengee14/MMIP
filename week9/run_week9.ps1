param(
    [ValidateSet('app', 'prepare', 'train', 'benchmark', 'integration', 'report', 'test')]
    [string]$Stage = 'app',
    [string]$Python = 'python',
    [int]$Port = 5000
)

$ErrorActionPreference = 'Stop'
$env:OPENBLAS_NUM_THREADS = '2'
$env:OMP_NUM_THREADS = '2'

function Invoke-Python {
    & $Python -X utf8 @args
    if ($LASTEXITCODE -ne 0) { throw "Python exited with code $LASTEXITCODE" }
}

Push-Location (Join-Path $PSScriptRoot 'quiz3/Code')
try {
    switch ($Stage) {
        'prepare' {
            Invoke-Python -m scripts.train_b
            Invoke-Python -c 'from triage.nodes.system_one import SystemOne, load_train_rows; SystemOne.train(load_train_rows()).save()'
        }
        'train' { Invoke-Python -m scripts.week9_llm train }
        'benchmark' { Invoke-Python -m scripts.week9_llm benchmark }
        'integration' {
            Invoke-Python -c 'from triage.vision import configure_ocr; print(configure_ocr())'
            Invoke-Python -m scripts.run_exp3
            Copy-Item -LiteralPath 'results/exp3_results.json' -Destination '../Output/exp3_results.json'
        }
        'report' { Invoke-Python -m scripts.week9_report }
        'test' { Invoke-Python -m unittest discover -s tests -v }
        'app' { Invoke-Python -m flask --app app run --host 127.0.0.1 --port $Port }
    }
} finally { Pop-Location }
