@echo off
cd /d "%~dp0"
chcp 65001 >nul 2>&1

cls
echo ============================================================
echo     ICT Backtest - FINAL FIXED VERSION
echo ============================================================
echo.
echo FIXES APPLIED:
echo   1. Auto-detect ALL ES symbols (ESZ4, ESH5, ESM5, etc)
echo   2. Fixed contract doubling (now always 1 contract)
echo   3. Reduced data requirements (50 trades, 10 candles)
echo.
echo Result: More days processed + Consistent position sizing
echo.
echo ============================================================
echo.

set PYTHONIOENCODING=utf-8

echo Starting backtest...
echo.

python -u full_year_with_charts.py

echo.
echo ============================================================
echo COMPLETED!
echo ============================================================
echo.
pause

