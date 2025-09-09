@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ==================================================
REM MLOps: Train -> Serve -> Drift -> Test (Windows)
REM ==================================================

REM ---- go to repo root (folder of this .bat) ----
cd /d "%~dp0"

REM ---- settings / args ----
set "ENV_NAME=id5003w"
set "DATA_REV=%~1"
if "%DATA_REV%"=="" set "DATA_REV=v1"
set "SKIPDEPS=%~2"

REM PORT logic: arg3 > APP_PORT env > default 8000
set "PORT=%~3"
if "%PORT%"=="" set "PORT=%APP_PORT%"
if "%PORT%"=="" set "PORT=8000"

echo ==========================================
echo MLOps: Train ^&gt; Serve ^&gt; Drift ^&gt; Test
echo ==========================================
echo Working directory: %CD%
echo Using conda env: %ENV_NAME%
echo DATA_REV: %DATA_REV%
echo PORT: %PORT%
echo.

REM ---- activate conda ----
call conda activate %ENV_NAME%
if errorlevel 1 (
    echo [ERROR] Failed to activate conda env %ENV_NAME%
    exit /b 1
)

REM ---- Create required directories ----
if not exist "artifacts_local" mkdir "artifacts_local"
if not exist "artifacts_local\Confusion_Matrix" mkdir "artifacts_local\Confusion_Matrix"
if not exist "artifacts_local\Feature_Importance" mkdir "artifacts_local\Feature_Importance"

REM ---- verify directories exist ----
if not exist "artifacts_local\Confusion_Matrix" (
    echo [ERROR] Failed to create artifacts_local\Confusion_Matrix directory
    exit /b 1
)

REM ---- DVC local-only cache (no remote) ----
dvc config cache.dir ".dvc/cache" >nul 2>&1
dvc config cache.type "copy" >nul 2>&1
dvc config --unset core.remote >nul 2>&1
if exist "data\train.csv.dvc" dvc commit -f "data\train.csv.dvc" >nul 2>&1
if exist "data\test.csv.dvc" dvc commit -f "data\test.csv.dvc" >nul 2>&1
dvc checkout --relink >nul 2>&1

REM ---- deps (optional) ----
if /I "%SKIPDEPS%"=="nodeps" (
    echo [INFO] Skipping dependency install
) else (
    echo [INFO] Installing/updating dependencies
    python -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] pip install failed
        exit /b 1
    )
)

REM ---- Spark -> Python bindings ----
set "PYTHONPATH=%CD%"
set "PYSPARK_PYTHON=%CONDA_PREFIX%\python.exe"
set "PYSPARK_DRIVER_PYTHON=%CONDA_PREFIX%\python.exe"
echo [INFO] PYSPARK_PYTHON=%PYSPARK_PYTHON%
echo [INFO] PYSPARK_DRIVER_PYTHON=%PYSPARK_DRIVER_PYTHON%

REM ---- train ----
echo [INFO] Starting training with DATA_REV=%DATA_REV%
set "DATA_REV=%DATA_REV%"
python -m training.training
if errorlevel 1 (
    echo [ERROR] Training failed
    exit /b 1
)

echo [OK] Training finished

REM ----- JUMP TO API DEPLOYMENT FOR DEBUGGING ----
GOTO DEPLOY_API

:DEPLOY_API
REM ---- ensure PORT is free (kill if busy) ----
echo [INFO] Checking if port %PORT% is free...
call :free_port %PORT%
if errorlevel 1 exit /b 1

REM ---- start API in a new window ----
echo [INFO] Starting API at http://127.0.0.1:%PORT%
set "APP_PORT=%PORT%"
set "MODEL_DIR=deployment\model"
start "AILab_API" cmd /k "uvicorn deployment.app:app --host 127.0.0.1 --port %PORT% && echo. && echo. && echo API Exited. Press any key to close this window... && pause > nul"
echo [INFO] Waiting 12 seconds for API to be ready
timeout /t 12 >nul

REM ---- ensure data\test.csv exists and is non-empty ----
echo [INFO] Restoring data\test.csv from DVC cache...
python deployment\safe_restore_test_csv.py
if errorlevel 1 (
    echo [ERROR] data/test.csv restore failed; aborting.
    exit /b 1
)

REM ---- drift detection ----
echo [INFO] Running drift detection on data\test.csv
python -m deployment.data_drift_detection --input "data\test.csv" --out "artifacts_local\drift_report.json" --threshold 0.30 --pval 0.05
if errorlevel 1 (
    echo [WARN] Drift detection failed (continuing)
) else (
    echo [OK] Drift report saved to artifacts_local\drift_report.json
)

REM ---- smoke test ----
echo [INFO] Running deployment\test.py
python "deployment\test.py"
if errorlevel 1 (
    echo [WARN] Smoke test reported an error; check API window logs
) else (
    echo [OK] Smoke test passed
)

echo.
echo Done. The API is running in a window titled: AILab_API
echo To stop it: taskkill /FI "WINDOWTITLE eq AILab_API" /F
echo.
endlocal
exit /b 0

REM =================== helpers ===================
:free_port
setlocal EnableExtensions EnableDelayedExpansion
set "PORTNUM=%~1"
set "KILLED=0"
echo [DEBUG] Inside :free_port, PORTNUM is %PORTNUM%

REM Using tasklist and findstr is another reliable method
for /f "tokens=2,5" %%A in ('tasklist /nh /fi "imagename eq uvicorn.exe" /fi "status eq running" 2^>nul ^| findstr ":%PORTNUM%"') do (
    echo [INFO] Port %PORTNUM% is busy by PID %%B - terminating...
    taskkill /PID %%B /F >nul 2>&1
    set "KILLED=1"
)

REM If the above tasklist doesn't work, this netstat version is the fallback.
if "!KILLED!"=="0" (
    REM Get PIDs for the listening port and write to a temporary file
    netstat -ano | findstr ":%PORTNUM%" | findstr "LISTENING" >nul 2>&1
    if not errorlevel 1 (
        for /f "tokens=5" %%P in ('netstat -ano ^| findstr ":%PORTNUM%" ^| findstr "LISTENING"') do (
            echo [INFO] Port %PORTNUM% is busy by PID %%P - terminating...
            taskkill /PID %%P /F >nul 2>&1
            set "KILLED=1"
        )
    )
)

if "!KILLED!"=="1" timeout /t 2 >nul

set "PID2="
REM Verify if the port is still busy
netstat -ano | findstr ":%PORTNUM%" | findstr "LISTENING" >nul 2>&1
if not errorlevel 1 (
    for /f "tokens=5" %%P in ('netstat -ano ^| findstr ":%PORTNUM%" ^| findstr "LISTENING"') do set "PID2=%%P"
)

if defined PID2 (
    echo [WARN] Could not free port %PORTNUM% ^(PID !PID2! still listening^). Close manually or choose another port.
    endlocal & exit /b 1
) else (
    echo [OK] Port %PORTNUM% is free.
    endlocal & exit /b 0
)

