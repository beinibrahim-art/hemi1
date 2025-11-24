@echo off
chcp 65001 >nul
cls

echo ╔══════════════════════════════════════════════════════════════════════╗
echo ║                                                                      ║
echo ║           🔬 ProjectX API - Order Placement Test 🔬                  ║
echo ║                                                                      ║
echo ╚══════════════════════════════════════════════════════════════════════╝
echo.
echo.
echo ⚠️  This will attempt to place a REAL order on your account!
echo.
echo    Make sure you're ready to test with real API calls.
echo.
echo    The script will ask for confirmation before placing the order.
echo.
echo.
pause
echo.
echo.

cd /d %~dp0
python test_place_order.py

echo.
echo.
echo ═══════════════════════════════════════════════════════════════════════
echo                              TEST COMPLETE
echo ═══════════════════════════════════════════════════════════════════════
echo.
pause

