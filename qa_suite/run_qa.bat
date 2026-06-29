@echo off
REM run_qa.bat - run the emotion-engine QA suite (Windows).
REM
REM Usage:
REM   qa_suite\run_qa.bat              fast offline tests (no docker stack needed)
REM   qa_suite\run_qa.bat offline      same as default
REM   qa_suite\run_qa.bat slow         big batteries: per-emotion corpus + ~950 fuzz
REM   qa_suite\run_qa.bat live         live @e2e tests (needs stack up + .env)
REM   qa_suite\run_qa.bat all          offline + live, excluding slow batteries
REM   qa_suite\run_qa.bat full         EVERYTHING: offline + slow + live (~1000 cases)
REM   qa_suite\run_qa.bat calibrate    derive/refresh thresholds from the battery
REM
REM Extra pytest args pass through, e.g.:  qa_suite\run_qa.bat offline -k emoji -v
setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "ROOT=%%~fI"
cd /d "%ROOT%"

REM Prefer the project venv if present.
if exist "%ROOT%\.venv\Scripts\python.exe" (
    set "PY=%ROOT%\.venv\Scripts\python.exe"
) else (
    set "PY=python"
)

set "MODE=%~1"
if "%MODE%"=="" set "MODE=offline"
if not "%~1"=="" shift

if /i "%MODE%"=="offline" (
    call :ensure_model
    echo [run_qa] offline suite ^(no stack required^)
    "%PY%" -m pytest qa_suite\ -m "not slow and not e2e" %*
    goto :end
)
if /i "%MODE%"=="slow" (
    call :ensure_model
    echo [run_qa] slow batteries ^(real GoEmotions 2000 + corpus + edge + fuzz^)
    "%PY%" -m pytest qa_suite\ -m slow %*
    goto :end
)
if /i "%MODE%"=="live"      goto :live
if /i "%MODE%"=="e2e"       goto :live
if /i "%MODE%"=="all"       goto :all
if /i "%MODE%"=="full"      goto :full
if /i "%MODE%"=="calibrate" (
    echo [run_qa] calibrating thresholds from the sentence battery
    "%PY%" qa_suite\calibrate.py %*
    goto :end
)

echo Unknown mode: %MODE%>&2
echo Use: offline ^| slow ^| live ^| all ^| full ^| calibrate>&2
exit /b 2

:loadenv
if exist "%ROOT%\.env" (
    for /f "usebackq tokens=1* delims==" %%A in ("%ROOT%\.env") do (
        set "line=%%A"
        if not "!line:~0,1!"=="#" if not "%%A"=="" set "%%A=%%B"
    )
    echo [run_qa] loaded .env
) else (
    echo [run_qa] WARNING: no .env found - live tests may skip ^(INTERNAL_API_KEY unset^)>&2
)
goto :eof

:live
call :loadenv
echo [run_qa] live suite ^(@e2e - requires the docker stack^)
"%PY%" -m pytest qa_suite\test_live.py -m e2e %*
goto :end

:all
call :ensure_model
call :loadenv
echo [run_qa] offline + live ^(excluding slow batteries^)
"%PY%" -m pytest qa_suite\ -m "not slow" %*
goto :end

:full
call :ensure_model
call :loadenv
echo [run_qa] FULL suite - offline + slow + live ^(~1000 cases^)
"%PY%" -m pytest qa_suite\ %*
goto :end

:ensure_model
REM Fetch the trained meta-learner from the container if the host lacks it,
REM so the 3 model-gated invariants run instead of skip. Silent fallback otherwise.
set "MODELPKL=%ROOT%\central_responder_service\models\meta_weights.pkl"
if exist "%MODELPKL%" goto :eof
for /f %%i in ('docker compose ps -q central_responder_service 2^>nul') do set "CID=%%i"
if not defined CID (
    echo [run_qa] no central_responder container - fallback mode ^(3 invariants skip^)
    goto :eof
)
echo [run_qa] fetching meta_weights.pkl from container %CID% ...
docker cp "%CID%:/app/models/meta_weights.pkl" "%MODELPKL%" 2>nul && (
    docker cp "%CID%:/app/models/meta_weights_meta.json" "%ROOT%\central_responder_service\models\meta_weights_meta.json" 2>nul
    echo [run_qa] model ready - invariants will run
) || echo [run_qa] could not fetch model - fallback mode ^(3 invariants skip^)
goto :eof

:end
endlocal
