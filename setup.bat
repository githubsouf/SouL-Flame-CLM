@echo off
REM ============================================================
REM  SouL Flame - one-time environment setup (Windows)
REM  double-click this file, or run it from cmd in this folder
REM ============================================================
setlocal
cd /d "%~dp0"

echo.
echo === SouL Flame setup ===
echo folder: %CD%
echo.

REM --- 1. find python -------------------------------------------------
where py >nul 2>&1
if %errorlevel%==0 (
    set "PY=py -3"
) else (
    where python >nul 2>&1
    if %errorlevel%==0 (
        set "PY=python"
    ) else (
        echo [FAIL] python not found.
        echo        install it from https://www.python.org/downloads/
        echo        and TICK "Add python.exe to PATH" during the install.
        pause
        exit /b 1
    )
)
echo [1/5] python found:
%PY% --version

REM --- 2. create the venv ---------------------------------------------
if exist ".venv\Scripts\python.exe" (
    echo [2/5] .venv already exists, keeping it
) else (
    echo [2/5] creating .venv ...
    %PY% -m venv .venv
    if errorlevel 1 (
        echo [FAIL] could not create the venv
        pause
        exit /b 1
    )
)

set "VPY=.venv\Scripts\python.exe"

REM --- 3. pip ----------------------------------------------------------
echo [3/5] upgrading pip ...
"%VPY%" -m pip install --upgrade pip --quiet

REM --- 4. packages -----------------------------------------------------
echo [4/5] installing torch, tiktoken, numpy, pytest ...
echo        (torch is ~200 MB, this takes a few minutes the first time)
"%VPY%" -m pip install -r requirements.txt
if errorlevel 1 (
    echo [FAIL] install failed. check your internet connection and retry.
    pause
    exit /b 1
)

REM --- 5. check --------------------------------------------------------
echo [5/5] checking the environment ...
echo.
"%VPY%" check_env.py

echo.
echo ============================================================
echo  done. from now on, in every new terminal, run:
echo.
echo      .venv\Scripts\activate
echo.
echo  then:   pytest -q
echo ============================================================
pause
