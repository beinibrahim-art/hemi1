@echo off
chcp 65001 >nul 2>&1
cd /d "%~dp0"

cls
echo ================================================================================
echo     ML COMPARISON - Rule-Based vs ML-Enhanced
echo ================================================================================
echo.
echo This will run a full comparison between:
echo   - Rule-Based selection (original method)
echo   - ML-Enhanced selection (AI-powered)
echo.
echo Time: ~15-20 minutes
echo.
echo ================================================================================
echo.

set PYTHONIOENCODING=utf-8

echo Starting ML-Enhanced Backtest...
echo.

python -u ml_enhanced_backtest.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: ML-Enhanced Backtest failed!
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo.
echo COMPARISON COMPLETE!
echo.
echo Check:
echo   - ml_models\rule_vs_ml_comparison.png (visual comparison)
echo   - Console output above (detailed stats)
echo.
echo ================================================================================
pause

