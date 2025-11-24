@echo off
chcp 65001 >nul
cls

echo ================================================================================
echo 🎯 ICT ML Trading Dashboard with ProjectX Integration
echo ================================================================================
echo.
echo 🚀 Starting advanced dashboard...
echo.
echo Features:
echo   ✅ Upload ML Model (.pkl file)
echo   ✅ Connect to ProjectX API (Real)
echo   ✅ TopStep, Tradeify, Funding Futures, E8X, FXIFY
echo   ✅ Auto-Trading with Real Data
echo   ✅ Live Monitoring
echo   ✅ Trade Execution
echo.
echo ================================================================================
echo.

cd C:\Users\hemi_\Downloads\ICT_Core_System

REM Install required libraries
echo 📦 Installing required libraries...
pip install flask --quiet
pip install projectx-api --quiet

echo.
echo ================================================================================
echo 🌐 Dashboard will open at: http://localhost:5000
echo.
echo 💡 Press Ctrl+C to stop
echo ================================================================================
echo.

REM Start dashboard
Start-Process "http://localhost:5000"
python dashboard_with_projectx.py

pause

