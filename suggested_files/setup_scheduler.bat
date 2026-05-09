:: ============================================================
:: setup_scheduler.bat
:: Run this ONCE on the office server to schedule the pipeline.
:: Edit the paths before running.
:: ============================================================

@echo off

:: Path to your Python interpreter inside the venv
SET PYTHON=C:\path\to\Portal_ML_V4\.venv\Scripts\python.exe

:: Path to your main script
SET SCRIPT=C:\path\to\Portal_ML_V4\sharepoint\sharepoint_downloader.py

:: Task name (can be anything)
SET TASKNAME=PortalML_SharePoint_Ingestion

:: Schedule: daily at 01:00
schtasks /create ^
  /tn "%TASKNAME%" ^
  /tr "\"%PYTHON%\" \"%SCRIPT%\"" ^
  /sc DAILY ^
  /st 01:00 ^
  /ru SYSTEM ^
  /f

echo Task created. Verify with:
echo schtasks /query /tn "%TASKNAME%" /fo LIST /v
pause
