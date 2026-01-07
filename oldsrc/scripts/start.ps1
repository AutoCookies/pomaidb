# PowerShell start script for Windows (PowerShell 5+) and cross-platform PowerShell Core
param(
  [string]$Port = $env:PORT,
  [string]$DataDir = $env:DATA_DIR,
  [string]$Persistence = $env:PERSISTENCE
)

if (-not $Port) { $Port = '8080' }
if (-not $DataDir) { $DataDir = '.\data' }
if (-not $Persistence) { $Persistence = 'noop' }

Write-Host "Ensuring data dir exists: $DataDir"
New-Item -ItemType Directory -Force -Path $DataDir > $null

$exe = ".\pomai-cache.exe"
if (-not (Test-Path $exe)) {
    Write-Host "Building pomai-cache.exe..."
    & go build -v -o pomai-cache.exe ./cmd/server
}

Write-Host "Starting pomai-cache (port=$Port, data=$DataDir, persistence=$Persistence)"
$env:PORT = $Port
$env:DATA_DIR = $DataDir
$env:PERSISTENCE = $Persistence

# Start inline and wait
& $exe