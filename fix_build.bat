@echo off
setlocal
cd /d "%~dp0"

echo ============================================
echo   None Trainer - Build Fixer
echo ============================================
echo.

:: 设置正确的嵌入式 Node 路径
set "NODE_EXE=%CD%\nodejs\node.exe"
set "NPM_CLI=%CD%\nodejs\node_modules\npm\bin\npm-cli.js"
set "WEBUI_DIR=%CD%\webui-vue"

echo [Config] Node: %NODE_EXE%
echo [Config] WebUI: %WEBUI_DIR%
echo.

if not exist "%NODE_EXE%" (
    echo [ERROR] Embedded Node.js not found at %NODE_EXE%
    pause
    exit /b 1
)

cd "%WEBUI_DIR%"

echo [1/3] Cleaning cleanup old dependencies...
if exist "node_modules" (
    echo   - Removing node_modules...
    rmdir /s /q "node_modules"
)
if exist "package-lock.json" (
    echo   - Removing package-lock.json...
    del "package-lock.json"
)

echo.
echo [2/3] Installing dependencies (using embedded npm)...
echo   - Registry: https://registry.npmmirror.com
call "%NODE_EXE%" "%NPM_CLI%" install --registry=https://registry.npmmirror.com
if errorlevel 1 (
    echo [ERROR] Install failed!
    pause
    exit /b 1
)

echo.


echo.
echo [3/3] Building frontend...
call "%NODE_EXE%" "%NPM_CLI%" run build
if errorlevel 1 (
    echo [ERROR] Build failed!
    pause
    exit /b 1
)

echo.
echo [SUCCESS] Build completed. You can now close this window.
pause
