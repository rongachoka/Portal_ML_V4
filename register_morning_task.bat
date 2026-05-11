@echo off
:: Registers morning_runner.py as a daily 4:00 AM Windows Task Scheduler entry.
:: Must be run as Administrator (right-click -> Run as administrator).

set TASKNAME=Portal Pharmacy\Morning Runner
set PYTHON=D:\Documents\Portal ML Analys\Portal_ML\portal_venv\Scripts\python.exe
set SCRIPT=D:\Documents\Portal ML Analys\Portal_ML\Portal_ML_V4\morning_runner.py

echo Registering Task Scheduler entry...
echo   Task  : %TASKNAME%
echo   Python: %PYTHON%
echo   Script: %SCRIPT%
echo.

schtasks /Create /TN "%TASKNAME%" /TR "\"%PYTHON%\" \"%SCRIPT%\"" /SC DAILY /ST 04:00 /RL HIGHEST /F

if %ERRORLEVEL% EQU 0 (
    echo.
    echo SUCCESS: Task registered. Runs daily at 04:00.
    schtasks /Query /TN "%TASKNAME%" /FO LIST
) else (
    echo.
    echo FAILED. Make sure you are running this as Administrator.
)
pause
