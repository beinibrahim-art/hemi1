@echo off
chcp 65001 >nul
cls

echo ================================================================================
echo 🚀 REAL Trading Platform
echo ================================================================================
echo.
echo منصة تداول حقيقية متكاملة 100%%
echo.
echo Features:
echo   ✅ REAL ProjectX API Connection
echo   ✅ REAL Account Data from TopStep/Tradeify/etc
echo   ✅ REAL Order Execution
echo   ✅ Upload ML Model
echo   ✅ Auto-Trading with ML
echo.
echo ================================================================================
echo.

cd C:\Users\hemi_\Downloads\ICT_Core_System

REM Stop previous instances
taskkill /F /IM python.exe /T 2>nul

REM Install required libraries
echo 📦 Installing/Updating libraries...
pip install flask flask-cors projectx-api joblib pandas numpy scikit-learn --upgrade --quiet

echo.
echo ================================================================================
echo 🌐 Starting REAL platform...
echo 📍 URL: http://localhost:5000
echo.
echo 💡 Press Ctrl+C to stop
echo ================================================================================
echo.

REM Start platform
python REAL_TRADING_PLATFORM.py

pause

