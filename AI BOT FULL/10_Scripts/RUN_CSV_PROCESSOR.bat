@echo off
chcp 65001 >nul
cls

echo ================================================================================
echo 🔄 CSV Signal Processor - معالج إشارات CSV
echo ================================================================================
echo.
echo هذا السكريبت يقرأ الإشارات من المؤشر ويعطي قرارات ML
echo.
echo Workflow:
echo   1. المؤشر يحفظ Setup في signals.csv
echo   2. Python يقرأ signals.csv
echo   3. ML يقيّم كل Setup
echo   4. يكتب القرار في decisions.csv
echo   5. المؤشر يقرأ decisions.csv وينفذ
echo.
echo ================================================================================
echo.

cd C:\Users\hemi_\Downloads\ICT_Core_System
python csv_signal_processor.py

echo.
echo ================================================================================
echo.
pause

