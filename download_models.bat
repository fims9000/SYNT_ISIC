@echo off
chcp 65001 >nul
echo 🚀 Загрузка моделей ISIC с Google Drive
echo ================================================
echo.

REM Проверяем наличие Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python не найден! Установите Python 3.8+ и добавьте в PATH
    echo Скачать можно с: https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Проверяем наличие pip
pip --version >nul 2>&1
if errorlevel 1 (
    echo ❌ pip не найден! Установите pip
    pause
    exit /b 1
)

echo ✅ Python найден
echo.

REM Устанавливаем зависимости если нужно
echo 📦 Проверяю зависимости...
pip install requests tqdm >nul 2>&1
if errorlevel 1 (
    echo ⚠️ Не удалось установить зависимости автоматически
    echo Попробуйте вручную: pip install requests tqdm
    echo.
)

echo ✅ Зависимости готовы
echo.

REM Запускаем скрипт загрузки
echo 🚀 Запускаю скрипт загрузки...
python download_models.py

echo.
echo 💡 Если возникли ошибки, проверьте:
echo    1. Правильность ID файла в download_models.py
echo    2. Доступность интернета
echo    3. Права на запись в текущую папку
echo.
pause
