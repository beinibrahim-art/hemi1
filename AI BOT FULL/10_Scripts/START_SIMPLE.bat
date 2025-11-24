@echo off
chcp 65001 >nul
cls

echo ================================================================================
echo 🎯 Simple Dashboard - API Connection
echo ================================================================================
echo.
echo صفحة بسيطة جداً لإدخال API Key
echo.
echo ================================================================================
echo.

cd C:\Users\hemi_\Downloads\ICT_Core_System

REM Stop any previous Python processes
taskkill /F /IM python.exe 2>nul

REM Install Flask if needed
pip install flask --quiet

REM Start simple dashboard
python SIMPLE_DASHBOARD.py

pause

