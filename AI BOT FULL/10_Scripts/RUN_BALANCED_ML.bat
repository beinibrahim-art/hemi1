@echo off
chcp 65001 >nul 2>&1
cd /d "%~dp0"

cls
echo ================================================================================
echo     BALANCED ML TRAINING
echo ================================================================================
echo.
echo Training ML with class imbalance handling:
echo   1. Class Weights (Random Forest)
echo   2. Scale Pos Weight (XGBoost)
echo   3. SMOTE Oversampling (XGBoost)
echo.
echo This will improve LOSS detection!
echo.
echo ================================================================================
echo.

set PYTHONIOENCODING=utf-8

echo Installing imblearn (if needed)...
python -m pip install imbalanced-learn --quiet

echo.
echo Starting Balanced ML Training...
echo.

python -u ml_trainer_balanced.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Training failed!
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo.
echo BALANCED ML TRAINING COMPLETE!
echo.
echo Check:
echo   - ml_models\RandomForest_Balanced_model.pkl
echo   - ml_models\XGBoost_Balanced_model.pkl
echo   - ml_models\XGBoost_SMOTE_model.pkl
echo   - ml_models\confusion_matrix_balanced.png
echo   - ml_models\feature_importance_balanced.png
echo   - ml_models\models_comparison.png
echo.
echo ================================================================================
pause

