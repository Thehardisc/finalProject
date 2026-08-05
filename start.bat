@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"

if not exist ".env" (
    echo [Warn] No .env file found next to start.bat -- using built-in defaults.
)

echo.
echo ==============================================================
echo      [System]  Emotion Analysis System - Launcher
echo ==============================================================
echo.

set MAX_EMPATHETIC_SAMPLES_VAL=25000
set MIN_DB_SAMPLES_VAL=50
set RETRAIN_INTERVAL_VAL=1800
set ACCURACY_GATE_VAL=0.30
set LLM_PROVIDER_VAL=RULE_BASED
set LOG_LEVEL_VAL=INFO
set RATE_LIMIT_MAX_VAL=60
set API_KEY_VAL=N/A
set ANTHROPIC_KEY_VAL=
set INGESTION_URL_VAL=http://localhost:8000
set API_URL_VAL=http://localhost:8001
set FRONTEND_URL_VAL=http://localhost:5173

for /f "usebackq tokens=1,* delims==" %%a in (".env") do (
    if "%%a"=="MAX_EMPATHETIC_SAMPLES"   set MAX_EMPATHETIC_SAMPLES_VAL=%%b
    if "%%a"=="MIN_DB_SAMPLES"           set MIN_DB_SAMPLES_VAL=%%b
    if "%%a"=="RETRAIN_INTERVAL_SECONDS" set RETRAIN_INTERVAL_VAL=%%b
    if "%%a"=="ACCURACY_GATE"            set ACCURACY_GATE_VAL=%%b
    if "%%a"=="LLM_PROVIDER"             set LLM_PROVIDER_VAL=%%b
    if "%%a"=="LOG_LEVEL"                set LOG_LEVEL_VAL=%%b
    if "%%a"=="RATE_LIMIT_MAX"           set RATE_LIMIT_MAX_VAL=%%b
    if "%%a"=="INTERNAL_API_KEY"         set API_KEY_VAL=%%b
    if "%%a"=="ANTHROPIC_API_KEY"        set ANTHROPIC_KEY_VAL=%%b
    if "%%a"=="INGESTION_URL"            set INGESTION_URL_VAL=%%b
    if "%%a"=="API_URL"                  set API_URL_VAL=%%b
    if "%%a"=="FRONTEND_URL"             set FRONTEND_URL_VAL=%%b
)

if "!ANTHROPIC_KEY_VAL!"=="" (
    set ANTHROPIC_STATUS=NOT SET ^(synthetic sentence generation will be skipped^)
) else (
    set ANTHROPIC_STATUS=SET ^(synthetic data generation enabled^)
)

echo [Config] Resolved environment from .env:
echo ----------------------------------------------------------
echo    LOG_LEVEL               = !LOG_LEVEL_VAL!
echo    MAX_EMPATHETIC_SAMPLES  = !MAX_EMPATHETIC_SAMPLES_VAL!  ^(bootstrap dataset cap, runs once^)
echo    MIN_DB_SAMPLES          = !MIN_DB_SAMPLES_VAL!  ^(min DB rows to trigger continuous cycle^)
echo    RETRAIN_INTERVAL_SECS   = !RETRAIN_INTERVAL_VAL!s
echo    ACCURACY_GATE           = !ACCURACY_GATE_VAL!  ^(min test accuracy to deploy model^)
echo    LLM_PROVIDER            = !LLM_PROVIDER_VAL!
echo    RATE_LIMIT_MAX          = !RATE_LIMIT_MAX_VAL! req/min per user
echo    API_KEY                 = !API_KEY_VAL:~0,6!...  ^(redacted^)
echo    ANTHROPIC_API_KEY       = !ANTHROPIC_STATUS!
echo ----------------------------------------------------------
echo.

echo [Invariant] Checking feature-vector parity ^(inference vs trainer^)...
where python >nul 2>&1
if %ERRORLEVEL% equ 0 (
    python -c "import pytest, numpy" >nul 2>&1
    if !ERRORLEVEL! equ 0 (
        python -m pytest qa_suite/unit/test_feature_parity.py -q > "%TEMP%\parity_check.log" 2>&1
        if !ERRORLEVEL! equ 0 (
            echo    [OK] Feature-vector parity holds.
        ) else (
            echo    [FAIL] Feature-vector parity VIOLATION -- inference and trainer disagree.
            echo           Launch aborted; fix build_feature_vector before starting.
            echo           Details:
            type "%TEMP%\parity_check.log"
            exit /b 1
        )
    ) else (
        echo    [SKIP] pytest/numpy not installed on host -- parity not checked.
    )
) else (
    echo    [SKIP] python not on host -- parity not checked.
)
echo.

echo [Data] Checking Claude-generated register datasets ^(.cache\^<register^>_samples.csv^)...
set MISSING_REGISTERS=
for %%r in (synthetic hyperbole banter) do (
    if not exist ".cache\%%r_samples.csv" set MISSING_REGISTERS=!MISSING_REGISTERS! %%r
)
if "!MISSING_REGISTERS!"=="" (
    echo    [OK] All register datasets present.
) else (
    set REGGEN_OK=0
    if not "!ANTHROPIC_KEY_VAL!"=="" (
        where python >nul 2>&1
        if !ERRORLEVEL! equ 0 (
            python -c "import anthropic" >nul 2>&1
            if !ERRORLEVEL! equ 0 set REGGEN_OK=1
        )
    )
    if "!REGGEN_OK!"=="1" (
        set ANTHROPIC_API_KEY=!ANTHROPIC_KEY_VAL!
        for %%r in (!MISSING_REGISTERS!) do (
            echo    [INFO] Generating '%%r' dataset via Claude API ^(one-time^)...
            python central_responder_service\trainer\data\register_gen.py %%r
            if !ERRORLEVEL! equ 0 (
                echo    [OK] %%r_samples.csv created.
            ) else (
                echo    [Warn] %%r generation failed -- trainer will skip that set.
            )
        )
    ) else (
        echo    [SKIP] Missing:!MISSING_REGISTERS! -- ANTHROPIC_API_KEY not set ^(or anthropic pkg absent^);
        echo           trainer will train without those sets. Generate later with:
        echo           python central_responder_service\trainer\data\register_gen.py all
    )
)
echo.

echo [Models] Syncing trained artifacts to .cache\ ^(what you train is what you run^)...
set SARC_SRC=central_responder_service\models\sarcasm_clf.pt
set SARC_DST=.cache\sarcasm_clf.pt
if exist "%SARC_SRC%" (
    if not exist .cache mkdir .cache
    xcopy /d /y "%SARC_SRC%" .cache\ >nul
    if exist "central_responder_service\models\sarcasm_clf_config.json" xcopy /d /y "central_responder_service\models\sarcasm_clf_config.json" .cache\ >nul
    echo    [OK] sarcasm_clf.pt synced to .cache\ ^(newer model deployed; service loads it on start^)
) else if exist "%SARC_DST%" (
    echo    [OK] sarcasm classifier in .cache\ is up to date.
) else (
    echo    [SKIP] no sarcasm_clf.pt trained yet -- sarcasm_score stays 0.0
    echo           ^(train with: python conversation_state_learner\train_sarcasm_classifier.py^)
)
echo.

echo [Search] Detecting host environment...
set BUILD_FLAG=--build
if /i "%~1"=="nobuild" (
    set BUILD_FLAG=
    echo    [INFO] 'nobuild' flag set -- starting containers without rebuilding images.
)
where nvidia-smi >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo    [OK] NVIDIA GPU detected! Launching with GPU acceleration...
    set COMPOSE_CMD=docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d !BUILD_FLAG!
) else (
    echo    [INFO] No NVIDIA GPU detected. Launching in CPU mode...
    set COMPOSE_CMD=docker compose up -d !BUILD_FLAG!
)
echo.

if not exist logs\live mkdir logs\live
for /f %%f in ('dir /b /a-d logs\live\ 2^>nul') do del /q "logs\live\%%f"

set TIMESTAMP=%date:~-4,4%-%date:~-10,2%-%date:~-7,2%_%time:~0,2%-%time:~3,2%-%time:~6,2%
set TIMESTAMP=%TIMESTAMP: =0%
set LOGDIR=logs\run_%TIMESTAMP%
if not exist "%LOGDIR%" mkdir "%LOGDIR%"

set STARTUP_LOG=%LOGDIR%\docker_startup.log
set INIT_LOG=%LOGDIR%\init.log
set LOGFILE=%LOGDIR%\all.log
set ERRORS_LOG=%LOGDIR%\errors.log
set IMPORTANT_LOG=%LOGDIR%\important.log

echo [Launch] !COMPOSE_CMD!
echo.
powershell -NoProfile -Command "& cmd /c '%COMPOSE_CMD% 2>&1' | Tee-Object -FilePath '%STARTUP_LOG%'; exit $LASTEXITCODE"
if %ERRORLEVEL% neq 0 (
    echo    [Fail] docker compose exited with error. See %STARTUP_LOG%
    exit /b 1
)
echo.

(
    echo ==============================================================
    echo    InnerLink Launcher - Init Log
    echo    Timestamp : %TIMESTAMP%
    echo ==============================================================
    echo.
    echo [Config] Resolved environment from .env:
    echo ----------------------------------------------------------
    echo    LOG_LEVEL               = !LOG_LEVEL_VAL!
    echo    MAX_EMPATHETIC_SAMPLES  = !MAX_EMPATHETIC_SAMPLES_VAL!
    echo    MIN_DB_SAMPLES          = !MIN_DB_SAMPLES_VAL!
    echo    RETRAIN_INTERVAL_SECS   = !RETRAIN_INTERVAL_VAL!s
    echo    ACCURACY_GATE           = !ACCURACY_GATE_VAL!
    echo    LLM_PROVIDER            = !LLM_PROVIDER_VAL!
    echo    RATE_LIMIT_MAX          = !RATE_LIMIT_MAX_VAL! req/min
    echo    API_KEY                 = !API_KEY_VAL:~0,6!...  ^(redacted^)
    echo    ANTHROPIC_API_KEY       = !ANTHROPIC_STATUS!
    echo ----------------------------------------------------------
    echo.
    echo [Hardware] Compose command: !COMPOSE_CMD!
    echo.
    echo ==============================================================
    echo [Docker Startup] build / create / start output
    echo ==============================================================
) > "%INIT_LOG%"
type "%STARTUP_LOG%" >> "%INIT_LOG%"

echo [Logs] Per-service ^(real-time^): logs\live\^<service^>.log
echo [Logs] Run artifacts: %LOGDIR%\

for /f %%a in ('powershell -NoProfile -Command "[DateTime]::UtcNow.ToString(\"yyyy-MM-ddTHH:mm:ssZ\")"') do set LAUNCH_TIME=%%a


start /b cmd /c "docker compose logs -f --since %LAUNCH_TIME% > %LOGFILE% 2>&1"

start /b cmd /c "docker compose logs -f --since %LAUNCH_TIME% 2>&1 | findstr /i /c:\"[ERROR\" /c:\"[CRITICAL\" /c:\"[WARNING\" /c:\"Traceback\" /c:\"Exception:\" /c:\"FATAL\" /c:\"CRASH\" > %ERRORS_LOG%"

start /b cmd /c "docker compose logs -f --since %LAUNCH_TIME% 2>&1 | findstr /v /i /c:\"GET /health\" /c:\"OPTIONS /health\" /c:\"GET / HTTP/1.1\" > %IMPORTANT_LOG%"

echo.
echo [Wait] Waiting for all containers to start...

for /f %%a in ('docker compose config --services 2^>nul ^| find /c /v ""') do set EXPECTED_SERVICES=%%a

set MAX_WAIT=120
set ELAPSED=0

:WAIT_LOOP
if !ELAPSED! geq !MAX_WAIT! goto :TIMEOUT
for /f %%b in ('docker compose ps --services --status running 2^>nul ^| find /c /v ""') do set RUNNING=%%b
if !RUNNING! geq !EXPECTED_SERVICES! (
    echo    [OK] All !EXPECTED_SERVICES! containers are running!
    goto :HEALTH_CHECKS
)
echo    [Wait] !RUNNING! / !EXPECTED_SERVICES! containers running... ^(!ELAPSED!s^)
ping 127.0.0.1 -n 6 >nul
set /a ELAPSED+=5
goto :WAIT_LOOP

:TIMEOUT
echo    [Warn] Timeout! Not all containers started within %MAX_WAIT%s.
echo    Check logs: %LOGFILE%
docker compose ps
exit /b 1

:HEALTH_CHECKS
echo.
echo [Health] Running service health checks...
set PASS=0
set FAIL=0
set FAILED_CHECKS=

docker compose exec -T redis redis-cli ping 2>nul | find "PONG" >nul
if %ERRORLEVEL% equ 0 (
    echo    [OK] Redis - PONG
    set /a PASS+=1
) else (
    echo    [Fail] Redis - not responding
    set /a FAIL+=1
    set FAILED_CHECKS=!FAILED_CHECKS!   - Redis: ping did not return PONG^

)

docker compose exec -T db pg_isready -U user -d emotion_db 2>nul | find "accepting" >nul
if %ERRORLEVEL% equ 0 (
    echo    [OK] PostgreSQL - accepting connections
    set /a PASS+=1
) else (
    echo    [Fail] PostgreSQL - not ready
    set /a FAIL+=1
    set FAILED_CHECKS=!FAILED_CHECKS!   - PostgreSQL: not accepting connections^

)

call :CheckHTTP "Ingestion Service (API :8000)" "!INGESTION_URL_VAL!/health"
call :CheckHTTP "API Service (WebSocket :8001)" "!API_URL_VAL!/health"
call :CheckHTTP "Frontend (UI :5173)" "!FRONTEND_URL_VAL!"

rem A freshly (re)built central_responder container needs ~40s to import torch and
rem load meta_weights.pkl before it logs its mode -- poll instead of a one-shot grep.
set ML_DEADLINE=90
set ML_ELAPSED=0
set ML_MODE=

:ML_WAIT
docker compose logs central_responder_service 2>nul | find "Running in META-LEARNER mode" >nul
if !ERRORLEVEL! equ 0 (
    set ML_MODE=meta
    goto :ML_DONE
)
docker compose logs central_responder_service 2>nul | find "Falling back to Rule-Based Aggregation" >nul
if !ERRORLEVEL! equ 0 (
    set ML_MODE=rule
    goto :ML_DONE
)
if !ML_ELAPSED! geq !ML_DEADLINE! goto :ML_DONE
ping 127.0.0.1 -n 4 >nul
set /a ML_ELAPSED+=3
goto :ML_WAIT

:ML_DONE
if "!ML_MODE!"=="meta" (
    echo    [OK] Meta-Learner - loaded and active
    set /a PASS+=1
) else if "!ML_MODE!"=="rule" (
    echo    [Fail] Meta-Learner - service fell back to RULE-BASED mode ^(no compatible meta_weights.pkl^)
    set /a FAIL+=1
    set FAILED_CHECKS=!FAILED_CHECKS!   - Meta-Learner: fell back to rule-based aggregation ^(no compatible meta_weights.pkl; trainer will hot-reload once a model passes the gate^)^

) else (
    echo    [Fail] Meta-Learner - no mode line logged within !ML_DEADLINE!s ^(check central_responder_service logs^)
    set /a FAIL+=1
    set FAILED_CHECKS=!FAILED_CHECKS!   - Meta-Learner: neither 'META-LEARNER mode' nor rule-based fallback logged within !ML_DEADLINE!s^

)

echo.
echo [Test] Running end-to-end pipeline test...
set PIPE_ATTEMPT=0
set HTTP_CODE=0

:PIPE_RETRY
if !PIPE_ATTEMPT! geq 10 goto :PIPE_FAIL
for /f %%a in ('powershell -NoProfile -command "try { $r = Invoke-WebRequest -Uri \"!INGESTION_URL_VAL!/messages\" -Method Post -ContentType \"application/json\" -Headers @{\"X-API-Key\"=\"!API_KEY_VAL!\"} -Body \"{\\\"conversation_id\\\": \\\"healthcheck\\\", \\\"user_id\\\": \\\"system\\\", \\\"text\\\": \\\"I am happy!\\\"}\" -UseBasicParsing; Write-Output $r.StatusCode } catch { if ($_.Exception.Response) { Write-Output ([int]$_.Exception.Response.StatusCode) } else { Write-Output 0 } }" 2^>nul') do set HTTP_CODE=%%a
if "!HTTP_CODE!"=="200" goto :PIPE_OK
ping 127.0.0.1 -n 3 >nul
set /a PIPE_ATTEMPT+=1
goto :PIPE_RETRY

:PIPE_OK
echo    [OK] Pipeline test - message accepted ^(HTTP 200^)
set /a PASS+=1
goto :SUMMARY

:PIPE_FAIL
echo    [Fail] Pipeline test - failed ^(HTTP !HTTP_CODE!^) after 10 tries
set /a FAIL+=1
set FAILED_CHECKS=!FAILED_CHECKS!   - Pipeline test: POST /messages returned HTTP !HTTP_CODE!^

:SUMMARY
echo.
echo ==============================================================
if !FAIL! equ 0 (
    echo    [Yay]  ALL CHECKS PASSED ^(!PASS!/!PASS!^)
) else (
    echo    [Warn]   !PASS! passed, !FAIL! failed
    echo.
    echo    Failed checks:
    echo    !FAILED_CHECKS!
)
echo ==============================================================
echo.
echo    [Web]  Frontend:     !FRONTEND_URL_VAL!
echo    [API]  API:          !API_URL_VAL!
echo    [In]   Ingestion:    !INGESTION_URL_VAL!
echo.
echo    [Log]  Per-service ^(real-time^): logs\live\^<service^>.log
echo    [Log]  Init report:  %INIT_LOG%
echo    [Log]  Errors only:  %ERRORS_LOG%
echo    [Log]  Important:    %IMPORTANT_LOG%
echo    [Log]  Full dump:    %LOGFILE%
echo.
echo    Tip: type logs\live\trainer_service.log
echo    Tip: type logs\live\central_responder_service.log
echo    Tip: type %ERRORS_LOG%
echo ==============================================================

(
    echo.
    echo [Health Checks]
    echo    PASS : !PASS!
    echo    FAIL : !FAIL!
    if !FAIL! equ 0 (
        echo    Result : ALL CHECKS PASSED
    ) else (
        echo    Result : !FAIL! check^(s^) FAILED
        echo.
        echo    Failed checks:
        echo    !FAILED_CHECKS!
    )
    echo.
    echo [URLs]
    echo    Frontend  : !FRONTEND_URL_VAL!
    echo    API       : !API_URL_VAL!
    echo    Ingestion : !INGESTION_URL_VAL!
    echo.
    echo [Log Files]
    echo    logs\live\^<service^>.log  ^(real-time, written directly by each service^)
    echo    %ERRORS_LOG%
    echo    %IMPORTANT_LOG%
    echo    %LOGFILE%
    echo.
    echo ==============================================================
    echo [Container Startup Logs] snapshot since launch ^(all services^)
    echo ==============================================================
) >> "%INIT_LOG%"
docker compose logs --since %LAUNCH_TIME% >> "%INIT_LOG%" 2>nul

echo.
echo    [Log] Init report saved: %INIT_LOG%
echo.
pause
exit /b !FAIL!

:CheckHTTP
set CHECK_NAME=%~1
set CHECK_URL=%~2
set MAX_RETRIES=10
set ATTEMPT=0

:CheckHTTPLoop
if !ATTEMPT! geq !MAX_RETRIES! (
    echo    [Fail] !CHECK_NAME! - NOT responding at !CHECK_URL!
    set /a FAIL+=1
    set FAILED_CHECKS=!FAILED_CHECKS!   - !CHECK_NAME!: no response after !MAX_RETRIES! tries^

    exit /b
)
powershell -NoProfile -command "try { $r = Invoke-WebRequest -Uri '!CHECK_URL!' -Headers @{'X-API-Key'='!API_KEY_VAL!'} -TimeoutSec 3 -UseBasicParsing; if ($r.StatusCode -eq 200) { exit 0 } else { exit 1 } } catch { exit 1 }" >nul 2>&1
if !ERRORLEVEL! equ 0 (
    echo    [OK] !CHECK_NAME! - responding
    set /a PASS+=1
    exit /b
)
ping 127.0.0.1 -n 3 >nul
set /a ATTEMPT+=1
goto :CheckHTTPLoop
