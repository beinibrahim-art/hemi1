@echo off
cd /d "%~dp0"
chcp 65001 >nul 2>&1

cls
echo ============================================================
echo     Resuming ICT Backtest - Continuing...
echo ============================================================
echo.
echo Current progress: 83 charts completed
echo Remaining: ~197 charts
echo.
echo ============================================================
echo.

set PYTHONIOENCODING=utf-8
python full_year_with_charts.py

echo.
echo ============================================================
echo COMPLETED!
echo ============================================================
pause

