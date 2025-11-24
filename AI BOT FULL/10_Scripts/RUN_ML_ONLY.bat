@echo off
chcp 65001 >nul 2>&1
cd /d "%~dp0"

cls
echo ================================================================================
echo     ML TRAINING ONLY
echo ================================================================================
echo.
echo Training ML on existing backtest data...
echo Time: ~2-3 minutes
echo.
echo ================================================================================
echo.

set PYTHONIOENCODING=utf-8

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
echo ML TRAINING COMPLETE!
echo.
echo Check:
echo   - ml_models\RandomForest_model.pkl
echo   - ml_models\XGBoost_model.pkl
echo   - ml_models\confusion_matrix.png
echo   - ml_models\feature_importance.png
echo.
echo ================================================================================
pause

