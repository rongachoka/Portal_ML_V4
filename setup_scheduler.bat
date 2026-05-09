@echo off
:: ============================================================
:: setup_scheduler.bat
:: Run this ONCE on the office server to register the nightly
:: pipeline as a Windows Task Scheduler job.
::
:: Requirements:
::   - Run as Administrator
::   - Edit PROJECT_DIR before running
::   - .env file must already exist at PROJECT_DIR\.env
:: 
:: ============================================================

SETLOCAL

:: ── Edit this line ────────────────────────────────────────────
SET PROJECT_DIR=D:\Documents\Portal ML Analys\Portal_ML\Portal_ML_V4
:: ─────────────────────────────────────────────────────────────

SET TASKNAME=PortalML_Nightly_Pipeline
SET SCRIPT=%PROJECT_DIR%\run_nightly.bat
SET RUN_TIME=01:00

:: Verify the batch script exists before registering
IF NOT EXIST "%SCRIPT%" (
    ECHO ERROR: run_nightly.bat not found at %SCRIPT%
    ECHO Make sure you have copied run_nightly.bat to the project folder
    ECHO and updated PROJECT_DIR in this script.
    PAUSE
    EXIT /B 1
)

:: Delete existing task if it exists (clean re-registration)
SCHTASKS /DELETE /TN "%TASKNAME%" /F >NUL 2>&1

:: Create the scheduled task
:: /RU SYSTEM   - runs whether or not anyone is logged in
:: /SC DAILY    - every day
:: /ST 01:00    - at 1am
:: /F           - force creation without confirmation prompt
SCHTASKS /CREATE ^
    /TN "%TASKNAME%" ^
    /TR "\"%SCRIPT%\"" ^
    /SC DAILY ^
    /ST %RUN_TIME% ^
    /RU SYSTEM ^
    /F

IF %ERRORLEVEL% NEQ 0 (
    ECHO ERROR: Failed to create scheduled task.
    ECHO Make sure you are running this script as Administrator.
    PAUSE
    EXIT /B 1
)

ECHO.
ECHO ============================================================
ECHO Task created successfully:
ECHO   Name:    %TASKNAME%
ECHO   Script:  %SCRIPT%
ECHO   Runs:    Daily at %RUN_TIME%
ECHO   User:    SYSTEM (runs without login)
ECHO ============================================================
ECHO.
ECHO Verify with:
ECHO   SCHTASKS /QUERY /TN "%TASKNAME%" /FO LIST /V
ECHO.
ECHO To run manually right now:
ECHO   SCHTASKS /RUN /TN "%TASKNAME%"
ECHO.
ECHO To disable without deleting:
ECHO   SCHTASKS /CHANGE /TN "%TASKNAME%" /DISABLE
ECHO.

PAUSE
ENDLOCAL