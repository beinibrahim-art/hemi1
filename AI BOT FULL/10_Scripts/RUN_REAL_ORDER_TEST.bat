@echo off
chcp 65001 >nul
cls

echo ╔══════════════════════════════════════════════════════════════════════╗
echo ║                                                                      ║
echo ║         ⚠️  REAL ORDER TEST - WILL PLACE ACTUAL ORDER! ⚠️            ║
echo ║                                                                      ║
echo ╚══════════════════════════════════════════════════════════════════════╝
echo.
echo.
echo ⚠️  ⚠️  ⚠️  WARNING ⚠️  ⚠️  ⚠️
echo.
echo    This will place a REAL order on your account!
echo.
echo    Order Details:
echo    - Account: PRAC-V2-378325-76029310 (ID: 14664883)
echo    - Contract: ESZ5 (CON.F.US.EP.Z25)
echo    - Type: BUY 1 contract at Market
echo    - Stop Loss: 20 ticks (5 points)
echo    - Take Profit: 12 ticks (3 points)
echo.
echo    Make sure:
echo    1. You have edited test_order_auto.py with your API Key
echo    2. You are ready to place a real order
echo    3. The market is open
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

