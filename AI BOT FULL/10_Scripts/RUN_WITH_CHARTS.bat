@echo off
chcp 65001 > nul
color 0A

echo ============================================================
echo     🎨 نظام ICT Backtest مع رسومات لكل يوم
echo ============================================================
echo.
echo ⚠️  تنبيه: إنشاء الرسومات يحتاج 2-5 ساعات!
echo.
echo المخرجات:
echo C:\Users\hemi_\Downloads\GLBX-20251120-PREEJVW86N\New folder
echo.
echo الرسومات:
echo C:\Users\hemi_\Downloads\GLBX-20251120-PREEJVW86N\New folder\charts
echo.
echo ============================================================
pause

echo.
echo [1] التحقق من المكتبات...
python -c "import databento, pandas, numpy, matplotlib" 2>nul
if errorlevel 1 (
    echo ❌ مكتبات ناقصة!
    echo.
    echo جاري التثبيت...
    pip install databento pandas numpy matplotlib
)
echo ✅ المكتبات جاهزة

echo.
echo [2] بدء التشغيل...
echo ============================================================
echo.

python full_year_with_charts.py

echo.
echo ============================================================
echo ✅ اكتمل!
echo ============================================================
echo.
echo تحقق من:
echo - ملفات CSV في: New folder
echo - الرسومات في: New folder\charts
echo.
pause

