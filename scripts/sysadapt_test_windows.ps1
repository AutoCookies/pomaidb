Write-Host "=== SYSADAPT TEST (Windows) ==="
Write-Host "Date: $(Get-Date)"
Write-Host ""
Write-Host "Total physical memory (MB):"
Get-CimInstance -ClassName Win32_ComputerSystem | Select-Object -ExpandProperty TotalPhysicalMemory
Write-Host ""
Write-Host "Run server and observe logs for SYSADAPT output."
Write-Host "Example:"
Write-Host "  .\\pomai-server.exe --mem-limit=6GB"