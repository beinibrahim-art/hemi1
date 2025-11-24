@echo off
chcp 65001 > nul
color 0A

echo ============================================================
echo     🎯 نظام ICT Backtest - المخرجات المخصصة
echo ============================================================
echo.
echo المخرجات ستحفظ في:
echo C:\Users\hemi_\Downloads\GLBX-20251120-PREEJVW86N\New folder
echo.
echo ============================================================

echo [1] التحقق من المكتبات المطلوبة...
python -c "import databento, pandas, numpy" 2>nul
if errorlevel 1 (
    echo ❌ المكتبات غير مثبتة!
    echo.
    echo جاري التثبيت...
    pip install databento pandas numpy matplotlib
    if errorlevel 1 (
        echo ❌ فشل التثبيت! تأكد من تثبيت Python بشكل صحيح.
        pause
        exit /b 1
    )
)
echo ✅ المكتبات جاهزة

echo.
echo [2] تشغيل Backtest...
echo ============================================================
echo.

python full_year_backtest_v2.py

echo.
echo ============================================================
echo ✅ اكتمل التنفيذ
echo ============================================================
echo.
echo تحقق من النتائج في:
echo C:\Users\hemi_\Downloads\GLBX-20251120-PREEJVW86N\New folder
echo.
pause

