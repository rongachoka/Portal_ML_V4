@echo off
:: ============================================================
:: run_pipeline.bat
:: Portal ML V4 — Pipeline launcher for Windows Task Scheduler
::
:: SETUP:
::   1. Edit PYTHON_PATH and PROJECT_DIR below to match your machine
::   2. Save this file to your project root
::   3. Register with Task Scheduler (see instructions below)
:: ============================================================

:: ── EDIT THESE TWO LINES ─────────────────────────────────────
set PYTHON_PATH=D:\Documents\Portal ML Analys\Portal_ML\portal_venv\Scripts\python.exe
set PROJECT_DIR=D:\Documents\Portal ML Analys\Portal_ML\Portal_ML_V4
:: ─────────────────────────────────────────────────────────────

set LOG_DIR=%PROJECT_DIR%\logs
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

set LOG_FILE=%LOG_DIR%\scheduler_%date:~-4,4%%date:~-7,2%%date:~0,2%_%time:~0,2%%time:~3,2%.log
set LOG_FILE=%LOG_FILE: =0%

echo ============================================================ >> "%LOG_FILE%"
echo Pipeline triggered at %date% %time% >> "%LOG_FILE%"
echo ============================================================ >> "%LOG_FILE%"

cd /d "%PROJECT_DIR%"

"%PYTHON_PATH%" run_pipeline.py >> "%LOG_FILE%" 2>&1

if %ERRORLEVEL% EQU 0 (
    echo [SUCCESS] Pipeline completed at %time% >> "%LOG_FILE%"
) else (
    echo [FAILED] Pipeline exited with code %ERRORLEVEL% at %time% >> "%LOG_FILE%"
)