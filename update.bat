@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo.
echo ============================================
echo   None Trainer - Update Script
echo ============================================
echo.

:: 切换到脚本所在目录
cd /d "%~dp0"

:: 检查 git 是否可用
where git >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Git not found! Please install Git first.
    pause
    exit /b 1
)

:: 拉取最新代码（强制覆盖本地更改）
echo [1/3] Pulling latest code from repository...
echo.
git fetch origin
if errorlevel 1 (
    echo.
    echo [ERROR] Git fetch failed! Please check your network or repository status.
    pause
    exit /b 1
)

:: 检测当前分支
for /f "tokens=*" %%i in ('git rev-parse --abbrev-ref HEAD 2^>nul') do set CURRENT_BRANCH=%%i
if "%CURRENT_BRANCH%"=="" set CURRENT_BRANCH=main

echo Resetting to origin/%CURRENT_BRANCH%...
git reset --hard origin/%CURRENT_BRANCH%
if errorlevel 1 (
    echo.
    echo [ERROR] Git reset failed!
    pause
    exit /b 1
)

echo.
echo [2/3] Installing frontend dependencies...
echo.

:: 检查嵌入式 Node.js
set "SCRIPT_DIR=%~dp0"
:: 移除末尾反斜杠
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

if exist "%SCRIPT_DIR%\node\node.exe" (
    set "NODE_EXE=%SCRIPT_DIR%\node\node.exe"
    set "NPM_CLI=%SCRIPT_DIR%\node\node_modules\npm\bin\npm-cli.js"
    if exist "!NPM_CLI!" (
        set "NPM_CMD=!NODE_EXE! !NPM_CLI!"
        echo Using embedded Node.js: !NODE_EXE!
    ) else (
        echo [WARN] Embedded npm not found, using system npm...
        set "NPM_CMD=npm"
    )
) else (
    echo [WARN] Embedded Node.js not found, using system Node.js...
    set "NPM_CMD=npm"
)

:: 进入前端目录
cd webui-vue

:: 安装依赖（如果 node_modules 不存在或 package.json 更新了）
if not exist "node_modules" (
    echo Installing npm packages...
    call !NPM_CMD! install
    if errorlevel 1 (
        echo [ERROR] npm install failed!
        cd ..
        pause
        exit /b 1
    )
)

echo.
echo [3/3] Building frontend...
echo.

:: 构建前端
call !NPM_CMD! run build
if errorlevel 1 (
    echo.
    echo [ERROR] Frontend build failed!
    cd ..
    pause
    exit /b 1
)

cd ..

echo.
echo ============================================
echo   Update completed successfully!
echo ============================================
echo.
echo You can now restart the application:
echo   - Run start.bat
echo   - Or: python start_services.py
echo.

pause


