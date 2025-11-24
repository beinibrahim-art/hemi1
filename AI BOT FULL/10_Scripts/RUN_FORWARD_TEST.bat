@echo off
chcp 65001 >nul 2>&1
cd /d "%~dp0"

cls
echo ================================================================================
echo     FORWARD TEST - Real Performance Test!
echo ================================================================================
echo.
echo What this does:
echo   1. Splits data by TIME (not random!)
echo   2. Trains on past data (70%%)
echo   3. Tests on FUTURE data (20%%) - UNSEEN!
echo   4. Shows REAL win rate (not 98%%!)
echo.
echo This is the RIGHT way to test ML!
echo.
echo ================================================================================
echo.

set PYTHONIOENCODING=utf-8

python -u forward_test_ml.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Forward test failed!
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo.
echo FORWARD TEST COMPLETE!
echo.
echo Check:
echo   - ml_models\forward_test_results.png (visual results)
echo   - ml_models\XGBoost_ForwardTested_model.pkl (new model)
echo.
echo The win rate shown is MORE REALISTIC than 98%%!
echo.
echo ================================================================================
pause

