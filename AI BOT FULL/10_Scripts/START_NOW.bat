@echo off
cd /d "%~dp0"
chcp 65001 >nul 2>&1

cls
echo ============================================================
echo     ICT Backtest System - Starting Now
echo ============================================================
echo.
echo Output folder:
echo C:\Users\hemi_\Downloads\GLBX-20251120-PREEJVW86N\New folder
echo.
echo Charts folder:
echo C:\Users\hemi_\Downloads\GLBX-20251120-PREEJVW86N\New folder\charts
echo.
echo ============================================================
echo.
echo Starting... Please wait...
echo This will take 2-5 hours to complete.
echo.

set PYTHONIOENCODING=utf-8
python full_year_with_charts.py

echo.
echo ============================================================
echo COMPLETED!
echo ============================================================
echo.
echo Check results in:
echo - CSV files: New folder
echo - Charts: New folder\charts
echo.
pause

