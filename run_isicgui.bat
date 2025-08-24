@echo off
setlocal

REM ————————————————————————————————
REM  ISICGUI Launcher (Windows)
REM  Запускает PyQt GUI из проводника двойным кликом
REM ————————————————————————————————

pushd %~dp0

REM Активируем venv, если есть
if exist ".venv\Scripts\activate.bat" call ".venv\Scripts\activate.bat"
if exist "venv\Scripts\activate.bat" call "venv\Scripts\activate.bat"

REM Если установлен py-launcher, используем его (берем последнюю 3.x)
where py >nul 2>&1
if %errorlevel%==0 (
    py -3 "%~dp0main.py"
) else (
    REM Фолбэк на python из PATH
    python "%~dp0main.py"
)

popd
endlocal


