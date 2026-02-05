@echo off
setlocal EnableDelayedExpansion
chcp 65001 >nul

REM ============================================
REM   None Trainer - Force Update Script
REM   参考 OneTrainer 设计，增强健壮性
REM ============================================

echo.
echo ============================================
echo   None Trainer - Force Update Script
echo ============================================
echo.

REM 切换到脚本所在目录
cd /d "%~dp0"
set "ROOT_DIR=%CD%"

REM ============================================
REM 设置环境变量
REM ============================================
if not defined GIT ( set "GIT=git" )
set "NODE_EXE=%ROOT_DIR%\nodejs\node.exe"
set "NPM_CLI=%ROOT_DIR%\nodejs\node_modules\npm\bin\npm-cli.js"
set "WEBUI_DIR=%ROOT_DIR%\webui-vue"

REM ============================================
REM 检查必要组件
REM ============================================
echo [CHECK] Verifying environment...

where "%GIT%" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Git not found! Please install Git first.
    goto :end_error
)
echo [OK] Git

if not exist "%NODE_EXE%" (
    echo [WARN] Embedded Node.js not found: %NODE_EXE%
    echo        Will skip frontend rebuild. Run install.bat if needed.
    set "SKIP_FRONTEND=1"
) else (
    echo [OK] Node: %NODE_EXE%
)

if not exist "%NPM_CLI%" (
    if not defined SKIP_FRONTEND (
        echo [WARN] Embedded npm not found: %NPM_CLI%
        set "SKIP_FRONTEND=1"
    )
) else (
    echo [OK] npm
)
echo.

REM ============================================
REM Step 1: Git 状态检查
REM ============================================
echo [1/3] Checking repository status...

REM 获取当前分支
FOR /F "tokens=* USEBACKQ" %%F IN (`"%GIT%" rev-parse --abbrev-ref HEAD 2^>nul`) DO (
    set "current_branch=%%F"
)
if not defined current_branch (
    set "current_branch=main"
)
echo Current branch: !current_branch!

REM 获取当前 commit
FOR /F "tokens=* USEBACKQ" %%F IN (`"%GIT%" rev-parse HEAD 2^>nul`) DO (
    set "local_commit=%%F"
)
echo Local commit:  !local_commit:~0,8!...

REM Fetch 远程更新
echo Fetching updates from origin...
"%GIT%" fetch origin
if errorlevel 1 (
    echo [ERROR] Git fetch failed! Check your network connection.
    goto :end_error
)

REM 获取远程 commit
FOR /F "tokens=* USEBACKQ" %%F IN (`"%GIT%" rev-parse origin/!current_branch! 2^>nul`) DO (
    set "remote_commit=%%F"
)
echo Remote commit: !remote_commit:~0,8!...

REM 比较 commit
if "!local_commit!"=="!remote_commit!" (
    echo.
    echo [INFO] Repository is already up to date!
    echo        Skipping git operations.
    goto :check_frontend
)

echo.
echo Updates available, applying changes...

REM ============================================
REM Step 2: 强制重置到远程版本
REM ============================================
echo [2/3] Force resetting to origin/!current_branch!...

REM 清理工作区
"%GIT%" clean -fd 2>nul
"%GIT%" checkout -- . 2>nul

REM 强制重置
"%GIT%" reset --hard origin/!current_branch!
if errorlevel 1 (
    echo [ERROR] Git reset failed!
    goto :end_error
)
echo [OK] Git reset completed.
echo.

:check_frontend
REM ============================================
REM Step 3: 前端依赖和构建
REM ============================================
if defined SKIP_FRONTEND (
    echo [SKIP] Frontend rebuild skipped (Node.js not found)
    goto :end_success
)

echo [3/3] Rebuilding frontend...

cd /d "%WEBUI_DIR%"
if errorlevel 1 (
    echo [ERROR] Cannot access webui-vue directory!
    goto :end_error
)

REM 检查是否需要重新安装依赖
set "NEED_INSTALL=0"
if not exist "node_modules" (
    set "NEED_INSTALL=1"
    echo [INFO] node_modules not found, will install dependencies.
)

REM 检查 package.json 是否比 node_modules 新
if exist "node_modules" (
    for %%A in ("package.json") do set "pkg_time=%%~tA"
    for /D %%A in ("node_modules") do set "nm_time=%%~tA"
    REM 简化处理：如果 package-lock.json 不存在，重新安装
    if not exist "package-lock.json" (
        set "NEED_INSTALL=1"
        echo [INFO] package-lock.json not found, will reinstall dependencies.
    )
)

if "!NEED_INSTALL!"=="1" (
    echo Cleaning old dependencies...
    if exist "node_modules" rmdir /s /q "node_modules" 2>nul
    if exist "package-lock.json" del /f /q "package-lock.json" 2>nul
    
    echo Running npm install...
    call "%NODE_EXE%" "%NPM_CLI%" install
    if errorlevel 1 (
        echo [ERROR] npm install failed!
        cd /d "%ROOT_DIR%"
        goto :end_error
    )
    echo [OK] Dependencies installed.
)

REM 构建前端
echo Building frontend...
call "%NODE_EXE%" "%NPM_CLI%" run build
if errorlevel 1 (
    echo [ERROR] Frontend build failed!
    cd /d "%ROOT_DIR%"
    goto :end_error
)
echo [OK] Frontend built.

cd /d "%ROOT_DIR%"

:end_success
echo.
echo ============================================
echo   Update completed successfully!
echo ============================================
echo.
echo Next: Run start.bat to launch the application.
echo.
goto :end

:end_error
echo.
echo ============================================
echo   ERROR: Update failed!
echo ============================================
echo.
goto :end

:end
cd /d "%ROOT_DIR%"
pause
exit /b %errorlevel%
