@echo off
chcp 65001 >nul
cls

echo ╔══════════════════════════════════════════════════════════════════════╗
echo ║                                                                      ║
echo ║         🤖 Automatic ProjectX API Test 🤖                            ║
echo ║                                                                      ║
echo ╚══════════════════════════════════════════════════════════════════════╝
echo.
echo.
echo ⚠️  BEFORE RUNNING:
echo.
echo    1. Open: test_order_auto.py
echo    2. Edit line 13: API_KEY = "YOUR_API_KEY_HERE"
echo    3. Replace with your actual API key
echo    4. Save the file
echo.
echo    Optional:
echo    5. Set ACTUALLY_PLACE_ORDER = True (to place real order)
echo       Default is False (simulation mode)
echo.
echo.
pause
echo.
echo.

cd /d %~dp0
python test_order_auto.py

echo.
echo.
echo ═══════════════════════════════════════════════════════════════════════
echo                              TEST COMPLETE
echo ═══════════════════════════════════════════════════════════════════════
echo.
pause

