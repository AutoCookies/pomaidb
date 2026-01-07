@echo off
REM Simple .bat start script for Windows cmd.exe
SET PORT=%PORT%
IF NOT DEFINED PORT SET PORT=8080
IF NOT DEFINED DATA_DIR SET DATA_DIR=.\data
IF NOT DEFINED PERSISTENCE SET PERSISTENCE=noop

IF NOT EXIST "%DATA_DIR%" mkdir "%DATA_DIR%"

IF NOT EXIST pomai-cache.exe (
  echo Building pomai-cache.exe...
  go build -v -o pomai-cache.exe ./cmd/server
)

set PORT=%PORT%
set DATA_DIR=%DATA_DIR%
set PERSISTENCE=%PERSISTENCE%

echo Starting pomai-cache (port=%PORT%, data=%DATA_DIR%, persistence=%PERSISTENCE%)
pomai-cache.exe