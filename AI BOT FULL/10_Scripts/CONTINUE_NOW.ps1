# ICT Backtest - Continue Script
Write-Host "============================================================" -ForegroundColor Green
Write-Host "     ICT Backtest System - Continuing..." -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Progress: 83/280 charts completed (29.6%)" -ForegroundColor Yellow
Write-Host "Remaining: ~197 charts" -ForegroundColor Yellow
Write-Host ""
Write-Host "Output: C:\Users\hemi_\Downloads\GLBX-20251120-PREEJVW86N\New folder" -ForegroundColor White
Write-Host "Charts: C:\Users\hemi_\Downloads\GLBX-20251120-PREEJVW86N\New folder\charts" -ForegroundColor White
Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Starting in 3 seconds..." -ForegroundColor Yellow
Start-Sleep -Seconds 3

Set-Location "C:\Users\hemi_\Downloads\ICT_Core_System"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "Running backtest... This will take 2-3 more hours." -ForegroundColor Cyan
Write-Host ""

python -u full_year_with_charts.py

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "COMPLETED!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Press any key to exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

