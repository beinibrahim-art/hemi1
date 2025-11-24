# ICT Backtest - Fixed Version
$Host.UI.RawUI.WindowTitle = "ICT Backtest - Fixed Version"

Write-Host "============================================================" -ForegroundColor Green
Write-Host "     ICT Backtest - Version with Fixed Data Filter" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
Write-Host "FIXED: Data requirements reduced" -ForegroundColor Yellow
Write-Host "  - Was: 1000 trades minimum -> Now: 100 trades" -ForegroundColor White
Write-Host "  - Was: 50 candles minimum -> Now: 20 candles" -ForegroundColor White
Write-Host ""
Write-Host "Result: Will process MORE days!" -ForegroundColor Green
Write-Host ""
Write-Host "Output: C:\Users\hemi_\Downloads\GLBX-20251120-PREEJVW86N\New folder" -ForegroundColor White
Write-Host "Charts: C:\Users\hemi_\Downloads\GLBX-20251120-PREEJVW86N\New folder\charts" -ForegroundColor White
Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""

$chartsCount = (Get-ChildItem "C:\Users\hemi_\Downloads\GLBX-20251120-PREEJVW86N\New folder\charts\*.png" -ErrorAction SilentlyContinue).Count
Write-Host "Current progress: $chartsCount charts already created" -ForegroundColor Cyan
Write-Host ""
Write-Host "Starting in 3 seconds..." -ForegroundColor Yellow
Start-Sleep -Seconds 3

Set-Location "C:\Users\hemi_\Downloads\ICT_Core_System"
$env:PYTHONIOENCODING = "utf-8"

Write-Host ""
Write-Host "Running backtest with FIXED version..." -ForegroundColor Cyan
Write-Host "Watch for fewer 'insufficient data' messages!" -ForegroundColor Green
Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""

python -u full_year_with_charts.py

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "COMPLETED!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Check results in:" -ForegroundColor White
Write-Host "  - CSV files: C:\Users\hemi_\Downloads\GLBX-20251120-PREEJVW86N\New folder" -ForegroundColor Cyan
Write-Host "  - Charts: C:\Users\hemi_\Downloads\GLBX-20251120-PREEJVW86N\New folder\charts" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press any key to exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

