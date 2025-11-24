# PowerShell script to start the AI Trading Bot server
# Author: AI Assistant
# Date: 2025-11-21

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  AI Trading Bot - Server Starting..." -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Get script directory
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectDir = Join-Path $scriptDir "01_Project_Files"
$mainFile = Join-Path $projectDir "REAL_TRADING_PLATFORM.py"

# Check if main file exists
if (-not (Test-Path $mainFile)) {
    Write-Host "Error: REAL_TRADING_PLATFORM.py not found!" -ForegroundColor Red
    Write-Host "Path: $mainFile" -ForegroundColor Yellow
    pause
    exit 1
}

# Change to project directory
Set-Location $projectDir
Write-Host "Changed to directory: $projectDir" -ForegroundColor Green
Write-Host ""

# Check if Python is installed
$pythonCheck = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCheck) {
    Write-Host "Error: Python is not installed or not in PATH!" -ForegroundColor Red
    Write-Host "Please install Python first." -ForegroundColor Yellow
    pause
    exit 1
}

Write-Host "Starting server on http://localhost:5000" -ForegroundColor Green
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

# Start the server
python REAL_TRADING_PLATFORM.py

