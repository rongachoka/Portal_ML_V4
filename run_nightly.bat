@echo off
:: ============================================================
:: run_nightly.bat
:: Portal ML — Nightly Data Pipeline
::
:: Runs in order:
::   1. sharepoint_downloader.py  → downloads files + loads staging tables
::   2. load_to_postgres.py       → staging → fact tables + refresh MVs
::
:: item_price_loader.py and product_map_build.py are run manually
:: (item prices change infrequently, product map only after KB updates)
::
:: Logs are written to logs\pipeline_YYYY-MM-DD.log
:: Edit PYTHON and PROJECT_DIR before first run.
::
:: How to Run Manually:
::   1. Open Command Prompt as Administrator
::   2. Navigate to this script's directory
::   3. Run: SCHTASKS /RUN /TN "PortalML_Nightly_Pipeline"
:: ============================================================

SETLOCAL

:: ── Configuration — edit these two lines ─────────────────────
SET PYTHON=D:\Documents\Portal ML Analys\Portal_ML\portal_venv\Scripts\python.exe
SET PROJECT_DIR=D:\Documents\Portal ML Analys\Portal_ML\Portal_ML_V4
SET ENV_FILE=D:\Documents\Portal ML Analys\Portal_ML\Portal_ML_V4\.env
:: ─────────────────────────────────────────────────────────────

:: Log file — one per day, kept in logs\ folder
SET LOG_DIR=%PROJECT_DIR%\logs
IF NOT EXIST "%LOG_DIR%" MKDIR "%LOG_DIR%"

SET TODAY=%DATE:~-4%-%DATE:~3,2%-%DATE:~0,2%
SET LOGFILE=%LOG_DIR%\pipeline_%TODAY%.log

ECHO ============================================================ >> "%LOGFILE%"
ECHO Portal ML Nightly Pipeline >> "%LOGFILE%"
ECHO Started: %DATE% %TIME% >> "%LOGFILE%"
ECHO ============================================================ >> "%LOGFILE%"

:: Change to project directory so relative imports work
CD /D "%PROJECT_DIR%"
SET PYTHONPATH=D:\Documents\Portal ML Analys\Portal_ML

:: ── Step 1: Download from SharePoint + load staging tables ───
ECHO. >> "%LOGFILE%"
ECHO [Step 1] SharePoint Download + Staging Load >> "%LOGFILE%"
ECHO Started: %TIME% >> "%LOGFILE%"

"%PYTHON%" -m Portal_ML_V4.sharepoint.sharepoint_downloader >> "%LOGFILE%" 2>&1

IF %ERRORLEVEL% NEQ 0 (
    ECHO [FAILED] sharepoint_downloader exited with error %ERRORLEVEL% >> "%LOGFILE%"
    ECHO Pipeline aborted. Check log: %LOGFILE%
    EXIT /B 1
)
ECHO [OK] Finished: %TIME% >> "%LOGFILE%"

:: ── Step 2: Staging → Fact tables + refresh materialized views
ECHO. >> "%LOGFILE%"
ECHO [Step 2] Load Fact Tables >> "%LOGFILE%"
ECHO Started: %TIME% >> "%LOGFILE%"

"%PYTHON%" -m Portal_ML_V4.src.pipelines.pos_finance.load_to_postgres >> "%LOGFILE%" 2>&1

IF %ERRORLEVEL% NEQ 0 (
    ECHO [FAILED] load_to_postgres exited with error %ERRORLEVEL% >> "%LOGFILE%"
    ECHO Pipeline failed at Step 2. Check log: %LOGFILE%
    EXIT /B 1
)
ECHO [OK] Finished: %TIME% >> "%LOGFILE%"

:: ── Done ─────────────────────────────────────────────────────
ECHO. >> "%LOGFILE%"
ECHO ============================================================ >> "%LOGFILE%"
ECHO Pipeline completed successfully >> "%LOGFILE%"
ECHO Finished: %DATE% %TIME% >> "%LOGFILE%"
ECHO ============================================================ >> "%LOGFILE%"

ECHO Pipeline completed. Log: %LOGFILE%
ENDLOCAL