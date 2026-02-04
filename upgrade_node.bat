@echo off
setlocal
cd /d "%~dp0"

echo ============================================
echo   Node.js x64 Upgrader
echo ============================================

set "NODE_URL=https://npmmirror.com/mirrors/node/v20.15.1/node-v20.15.1-win-x64.zip"
set "ZIP_FILE=node_x64.zip"
set "TEMP_DIR=node_extract"

echo [1/4] Downloading Node.js v20.15.1 (x64)...
powershell -Command "Invoke-WebRequest -Uri '%NODE_URL%' -OutFile '%ZIP_FILE%'"
if not exist "%ZIP_FILE%" (
    echo [ERROR] Download failed!
    pause
    exit /b 1
)

echo [2/4] Extracting...
powershell -Command "Expand-Archive -Path '%ZIP_FILE%' -DestinationPath '%TEMP_DIR%' -Force"

echo [3/4] Replacing embedded files...
:: 备份自定义脚本
if not exist "nodejs" mkdir "nodejs"
copy /y "nodejs\*.bat" "nodejs\*.bat.bak" >nul 2>&1

:: 覆盖核心文件
xcopy /e /y /q "%TEMP_DIR%\node-v20.15.1-win-x64\*" "nodejs\"

:: 恢复自定义脚本（如果被覆盖）
if exist "nodejs\*.bat.bak" (
    copy /y "nodejs\*.bat.bak" "nodejs\*.bat" >nul 2>&1
    del "nodejs\*.bat.bak"
)

echo [4/4] Cleanup...
del "%ZIP_FILE%"
rmdir /s /q "%TEMP_DIR%"

echo.
echo [SUCCESS] Node.js updated to x64!
echo Verifying:
nodejs\node.exe -p "'Arch: ' + process.arch + ', Version: ' + process.version"

pause
