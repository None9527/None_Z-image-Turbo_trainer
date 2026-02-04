@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

:: ============================================================================
:: None Trainer - Windows Startup Script (Portable Edition)
:: ============================================================================

:: Get script directory
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

:: ============================================================================
:: Environment Setup
:: ============================================================================
echo [Setup Environment]

:: Define interpreter paths
set "PYTHON_EXE=%SCRIPT_DIR%python\python.exe"
set "NODE_EXE=%SCRIPT_DIR%nodejs\node.exe"

:: Torch/CUDA library path
set "TORCH_LIB=%SCRIPT_DIR%python\Lib\site-packages\torch\lib"

:: Inject PATH (Embedded Python/Node > Torch > System)
set "PATH=%SCRIPT_DIR%python;%SCRIPT_DIR%python\Scripts;%SCRIPT_DIR%nodejs;%TORCH_LIB%;%PATH%"

:: Python module path
set "PYTHONPATH=%SCRIPT_DIR%src"
set "PYTHONUNBUFFERED=1"

echo   Python: %PYTHON_EXE%
echo   Node:   %NODE_EXE%
echo.

:: ============================================================================
:: Default Configuration
:: ============================================================================
set "TRAINER_PORT=9198"
set "TRAINER_HOST=0.0.0.0"
set "DEV_MODE=0"

:: Parse command line arguments
:parse_args
if "%~1"=="" goto :end_parse
if "%~1"=="--port" (
    set "TRAINER_PORT=%~2"
    shift & shift
    goto :parse_args
)
if "%~1"=="-p" (
    set "TRAINER_PORT=%~2"
    shift & shift
    goto :parse_args
)
if "%~1"=="--dev" (
    set "DEV_MODE=1"
    shift
    goto :parse_args
)
if "%~1"=="--help" goto :show_help
if "%~1"=="-h" goto :show_help
    shift
    goto :parse_args

:show_help
echo None Trainer Startup Script
echo.
echo Usage: start.bat [options]
echo.
echo Options:
echo   --port, -p PORT    Set port (default: 9198)
echo   --dev              Development mode (hot reload)
echo   --help, -h         Show this help
exit /b 0

:end_parse

:: ============================================================================
:: Load .env Configuration
:: ============================================================================
if exist "%SCRIPT_DIR%.env" (
    for /f "usebackq tokens=1,* delims==" %%a in ("%SCRIPT_DIR%.env") do (
        set "line=%%a"
        if not "!line:~0,1!"=="#" (
            if not "%%b"=="" set "%%a=%%b"
        )
    )
)

:: Apply defaults
if "%MODEL_PATH%"=="" set "MODEL_PATH=%SCRIPT_DIR%zimage_models"
if "%DATASET_PATH%"=="" set "DATASET_PATH=%SCRIPT_DIR%datasets"
if "%OUTPUT_PATH%"=="" set "OUTPUT_PATH=%SCRIPT_DIR%output"
if "%OLLAMA_HOST%"=="" set "OLLAMA_HOST=http://127.0.0.1:11434"

:: Create directories
if not exist "%DATASET_PATH%" mkdir "%DATASET_PATH%"
if not exist "%OUTPUT_PATH%" mkdir "%OUTPUT_PATH%"
if not exist "%OUTPUT_PATH%\logs" mkdir "%OUTPUT_PATH%\logs"
if not exist "logs" mkdir "logs"

:: ============================================================================
:: Display Banner
:: ============================================================================
cls
echo.
echo    _   _                    _____          _
echo   ^| \ ^| ^|                  ^|_   _^|        (_)
echo   ^|  \^| ^| ___  _ __   ___    ^| ^|_ __ __ _ _ _ __   ___ _ __
echo   ^| . ` ^|/ _ \^| '_ \ / _ \   ^| ^| '__/ _` ^| ^| '_ \ / _ \ '__^|
echo   ^| ^|\  ^| (_) ^| ^| ^| ^|  __/   ^| ^| ^| ^| (_^| ^| ^| ^| ^| ^|  __/ ^|
echo   ^|_^| \_^|\___/^|_^| ^|_^|\___^|   \_/_^|  \__,_^|_^|_^| ^|_^|\___^|_^|
echo.
echo.

:: ============================================================================
:: Display Configuration
:: ============================================================================
echo ==================================================
echo    Service Configuration
echo ==================================================
echo    Port:        %TRAINER_PORT%
echo    Host:        %TRAINER_HOST%
echo    Models:      %MODEL_PATH%
echo    Datasets:    %DATASET_PATH%
echo    Output:      %OUTPUT_PATH%
echo    Ollama:      %OLLAMA_HOST%
echo ==================================================
echo.

:: ============================================================================
:: Check Services
:: ============================================================================
echo [Check Services]

:: Check Python
if exist "%PYTHON_EXE%" (
    echo    Python:  [OK] Found
) else (
    echo    Python:  [X] Not found: %PYTHON_EXE%
    echo.
    echo ERROR: Embedded Python not found!
    pause
    exit /b 1
)

:: Check Ollama
curl -s "%OLLAMA_HOST%/api/tags" >nul 2>&1
if %errorlevel%==0 (
    echo    Ollama:  [OK] Running
) else (
    echo    Ollama:  [-] Not running (tagging unavailable)
)

:: Check GPU
nvidia-smi >nul 2>&1
if %errorlevel%==0 (
    for /f "tokens=*" %%i in ('nvidia-smi --query-gpu^=name --format^=csv^,noheader 2^>nul') do (
        echo    GPU:     [OK] %%i
    )
) else (
    echo    GPU:     [-] Not detected
    )

echo.

:: ============================================================================
:: Start Server
:: ============================================================================
echo Starting Web UI...
echo.
echo    URL: http://localhost:%TRAINER_PORT%
echo.
echo    Press Ctrl+C to stop
echo.
echo ==================================================

:: Change to API directory
cd /d "%SCRIPT_DIR%webui-vue\api"

:: Open browser after 2 seconds (background)
start /b cmd /c "timeout /t 2 /nobreak >nul && start http://localhost:%TRAINER_PORT%"

:: Start server
if "%DEV_MODE%"=="1" (
    echo [Dev Mode] Hot reload enabled
    "%PYTHON_EXE%" -m uvicorn main:app --host %TRAINER_HOST% --port %TRAINER_PORT% --reload --reload-dir "%SCRIPT_DIR%webui-vue\api" --log-level info
) else (
    "%PYTHON_EXE%" -m uvicorn main:app --host %TRAINER_HOST% --port %TRAINER_PORT% --log-level warning
)

:: ============================================================================
:: Shutdown
:: ============================================================================
echo.
echo ==================================================
echo    Server stopped. Exit code: %errorlevel%
echo ==================================================
pause
