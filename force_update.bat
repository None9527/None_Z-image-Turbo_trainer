@echo off
chcp 65001 >nul
setlocal

echo.
echo ============================================
echo   None Trainer - Force Update Script
echo ============================================
echo.

:: 切换到脚本所在目录
cd /d "%~dp0"
set "ROOT_DIR=%CD%"

:: ============================================
:: 设置嵌入式环境路径
:: ============================================
set "NODE_EXE=%ROOT_DIR%\nodejs\node.exe"
set "NPM_CLI=%ROOT_DIR%\nodejs\node_modules\npm\bin\npm-cli.js"
set "WEBUI_DIR=%ROOT_DIR%\webui-vue"

:: ============================================
:: 检查必要组件
:: ============================================
echo [CHECK] Verifying environment...

where git >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Git not found! Please install Git first.
    pause
    exit /b 1
)

if not exist "%NODE_EXE%" (
    echo [ERROR] Embedded Node.js not found: %NODE_EXE%
    pause
    exit /b 1
)

if not exist "%NPM_CLI%" (
    echo [ERROR] Embedded npm not found: %NPM_CLI%
    pause
    exit /b 1
)

echo [OK] Git
echo [OK] Node: %NODE_EXE%
echo [OK] npm: %NPM_CLI%
echo.

:: ============================================
:: Step 1: 拉取最新代码（强制覆盖）
:: ============================================
echo [1/3] Pulling latest code...
echo.

git fetch origin
if errorlevel 1 (
    echo [ERROR] Git fetch failed!
    pause
    exit /b 1
)

for /f "tokens=*" %%i in ('git rev-parse --abbrev-ref HEAD 2^>nul') do set "BRANCH=%%i"
if "%BRANCH%"=="" set "BRANCH=main"

echo Resetting to origin/%BRANCH%...
git reset --hard "origin/%BRANCH%"
if errorlevel 1 (
    echo [ERROR] Git reset failed!
    pause
    exit /b 1
)
echo.

:: ============================================
:: Step 2: 清理并重新安装前端依赖
:: 关键：彻底删除 node_modules 解决 rollup bug
:: ============================================
echo [2/3] Installing frontend dependencies...
echo.

cd /d "%WEBUI_DIR%"
if errorlevel 1 (
    echo [ERROR] Cannot access webui-vue directory!
    pause
    exit /b 1
)

echo Cleaning node_modules and package-lock.json...
if exist "node_modules" (
    echo Removing node_modules...
    rmdir /s /q "node_modules" 2>nul
)
if exist "package-lock.json" (
    echo Removing package-lock.json...
    del /f /q "package-lock.json" 2>nul
)

echo.
echo Running npm install...
call "%NODE_EXE%" "%NPM_CLI%" install
if errorlevel 1 (
    echo.
    echo [ERROR] npm install failed!
    cd /d "%ROOT_DIR%"
    pause
    exit /b 1
)
echo.

:: ============================================
:: Step 3: 构建前端
:: ============================================
echo [3/3] Building frontend...
echo.

call "%NODE_EXE%" "%NPM_CLI%" run build
if errorlevel 1 (
    echo.
    echo [ERROR] Frontend build failed!
    cd /d "%ROOT_DIR%"
    pause
    exit /b 1
)

cd /d "%ROOT_DIR%"

:: ============================================
:: 完成
:: ============================================
echo.
echo ============================================
echo   Update completed successfully!
echo ============================================
echo.
echo Next: Run start.bat to launch the application.
echo.

pause
