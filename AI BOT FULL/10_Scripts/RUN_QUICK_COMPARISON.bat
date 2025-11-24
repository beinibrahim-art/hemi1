@echo off
chcp 65001 >nul 2>&1
cd /d "%~dp0"

cls
echo ================================================================================
echo     QUICK MODEL COMPARISON
echo ================================================================================
echo.
echo This will quickly test ALL models on existing data:
echo   - XGBoost (Original)
echo   - RandomForest (Original)
echo   - RandomForest_Balanced
echo   - XGBoost_Balanced
echo   - XGBoost_SMOTE
echo.
echo Shows Win Rate for each model!
echo Time: ~1 minute
echo.
echo ================================================================================
echo.

set PYTHONIOENCODING=utf-8

python -u quick_model_comparison.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Comparison failed!
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo.
echo COMPARISON COMPLETE!
echo.
echo Check: ml_models\models_quick_comparison.csv
echo.
echo ================================================================================
pause

