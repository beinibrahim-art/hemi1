@echo off
chcp 65001 >nul
cls

echo ================================================================================
echo 🎯 ICT ML Trading Dashboard
echo ================================================================================
echo.
echo 🚀 Starting dashboard...
echo.
echo Features:
echo   ✅ ML Model Selection
echo   ✅ Account Connection (Sim/Funded)
echo   ✅ Auto-Trading
echo   ✅ Live Monitoring
echo   ✅ Trade History
echo   ✅ Statistics
echo.
echo 🌐 Dashboard will open at: http://localhost:5000
echo.
echo 💡 Press Ctrl+C to stop
echo.
echo ================================================================================
echo.

cd C:\Users\hemi_\Downloads\ICT_Core_System

REM Install Flask if not installed
pip install flask --quiet

REM Start dashboard
python dashboard_app.py

pause

