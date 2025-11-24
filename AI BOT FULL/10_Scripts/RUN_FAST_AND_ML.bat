@echo off
chcp 65001 >nul 2>&1
cd /d "%~dp0"

cls
echo ================================================================================
echo     FAST BACKTEST + ML TRAINING
echo ================================================================================
echo.
echo Step 1: Fast Backtest (no charts) - 15-20 minutes
echo Step 2: ML Training - 2-3 minutes
echo Step 3: ML-Enhanced Backtest - 15-20 minutes
echo.
echo Total time: ~35-45 minutes
echo.
echo ================================================================================
echo.

set PYTHONIOENCODING=utf-8

echo [STEP 1/3] Running Fast Backtest...
echo.
python -u fast_backtest_no_charts.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Fast Backtest failed!
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo.
echo [STEP 2/3] Training ML Model...
echo.
python -u ml_trainer.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: ML Training failed!
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo.
echo COMPLETED!
echo.
echo Check results in:
echo   - backtest_trades.csv (Rule-Based results)
echo   - ml_models\ (ML models and charts)
echo.
echo Next step: Run ml_enhanced_backtest.py to compare!
echo.
echo ================================================================================
pause

