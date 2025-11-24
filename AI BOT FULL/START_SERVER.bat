@echo off
chcp 65001 >nul
echo ========================================
echo   AI Trading Bot - Server Starting...
echo ========================================
echo.

cd /d "%~dp0\01_Project_Files"

if not exist "REAL_TRADING_PLATFORM.py" (
    echo Error: REAL_TRADING_PLATFORM.py not found!
    pause
    exit /b 1
)

echo Starting server on http://localhost:5000
echo.
echo Press Ctrl+C to stop the server
echo.

python REAL_TRADING_PLATFORM.py

pause

