$ErrorActionPreference = "Stop"
$env:PYTHONPATH = "src"
$python = "python"
if (Test-Path ".\.venv\Scripts\python.exe") {
    $python = ".\.venv\Scripts\python.exe"
}
& $python -m unittest discover -s tests
