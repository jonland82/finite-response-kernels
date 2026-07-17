param(
    [switch]$Cpu
)

$ErrorActionPreference = "Stop"
$reproductionRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvPath = Join-Path $reproductionRoot ".venv"
$pythonPath = Join-Path $venvPath "Scripts\python.exe"

if (-not (Test-Path -LiteralPath $pythonPath)) {
    py -3.10 -m venv $venvPath
}

& $pythonPath -m pip install --upgrade pip
if ($Cpu) {
    & $pythonPath -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
} else {
    & $pythonPath -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
}
& $pythonPath -m pip install -r (Join-Path $reproductionRoot "requirements.txt")

& $pythonPath -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
