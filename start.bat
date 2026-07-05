@echo off
setlocal
cd /d "%~dp0"

set "PATH=%USERPROFILE%\.local\bin;%USERPROFILE%\.cargo\bin;%APPDATA%\uv\bin;%LOCALAPPDATA%\uv\bin;%LOCALAPPDATA%\Programs\uv;%PATH%"
where uv >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] uv was not found. Run install.bat first.
    pause
    exit /b 1
)

set "APP_CACHE=%CD%\models\.cache"
set "TEMP=%APP_CACHE%\tmp"
set "TMP=%APP_CACHE%\tmp"
set "GRADIO_TEMP_DIR=%APP_CACHE%\tmp"
set "HF_HOME=%CD%\models"
set "HUGGINGFACE_HUB_CACHE=%APP_CACHE%\huggingface"
set "HF_XET_CACHE=%APP_CACHE%\xet"
set "TRANSFORMERS_CACHE=%CD%\models"
set "TORCH_HOME=%APP_CACHE%\torch"
set "XDG_CACHE_HOME=%APP_CACHE%"
set "UV_CACHE_DIR=%APP_CACHE%\uv"
set "HF_MODULES_CACHE=%APP_CACHE%\hf_modules"
set "PYTHONIOENCODING=utf-8"
set "PYTHONUNBUFFERED=1"
if "%HIGGS_CPU_THREADS%"=="" set "HIGGS_CPU_THREADS=8"
set "OMP_NUM_THREADS=%HIGGS_CPU_THREADS%"
set "MKL_NUM_THREADS=%HIGGS_CPU_THREADS%"
set "NUMEXPR_NUM_THREADS=%HIGGS_CPU_THREADS%"

for %%D in ("%TEMP%" "%HF_HOME%" "%TORCH_HOME%" "%XDG_CACHE_HOME%" "%UV_CACHE_DIR%" "%HF_MODULES_CACHE%") do if not exist %%D mkdir %%D

uv run --no-sync python app.py
pause
