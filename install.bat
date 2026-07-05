@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

call :ensure_uv
if errorlevel 1 exit /b %errorlevel%

set "APP_CACHE=%CD%\models\.cache"
for %%D in (models samples outputs logs data exp config "%APP_CACHE%" "%APP_CACHE%\uv" "%APP_CACHE%\tmp" "%APP_CACHE%\huggingface" "%APP_CACHE%\xet") do if not exist "%%~D" mkdir "%%~D"
set "UV_CACHE_DIR=%APP_CACHE%\uv"
set "XDG_CACHE_HOME=%APP_CACHE%"
set "HUGGINGFACE_HUB_CACHE=%APP_CACHE%\huggingface"
set "HF_XET_CACHE=%APP_CACHE%\xet"
set "TEMP=%APP_CACHE%\tmp"
set "TMP=%APP_CACHE%\tmp"
set "UV_LINK_MODE=copy"

echo.
echo Select PyTorch backend:
echo   1. Auto-detect NVIDIA / CPU
echo   2. NVIDIA GTX 10xx Pascal - CUDA 11.8
echo   3. NVIDIA RTX 20xx/30xx - CUDA 12.6
echo   4. NVIDIA RTX 40xx/50xx - CUDA 12.8
echo   5. CPU only
echo   6. Windows AMD DirectML experimental
echo.
set /p TORCH_CHOICE="Choose backend (1-6, default 1): "
if "%TORCH_CHOICE%"=="" set "TORCH_CHOICE=1"

set "TORCH_BACKEND=auto"
if "%TORCH_CHOICE%"=="1" call :detect_backend
if "%TORCH_CHOICE%"=="2" set "TORCH_BACKEND=cu118"
if "%TORCH_CHOICE%"=="3" set "TORCH_BACKEND=cu126"
if "%TORCH_CHOICE%"=="4" set "TORCH_BACKEND=cu128"
if "%TORCH_CHOICE%"=="5" set "TORCH_BACKEND=cpu"
if "%TORCH_CHOICE%"=="6" set "TORCH_BACKEND=directml"
if "%TORCH_BACKEND%"=="auto" set "TORCH_BACKEND=cpu"

echo [install] Selected backend: %TORCH_BACKEND%
echo %TORCH_BACKEND%>torch_backend.txt

echo [install] Installing app dependencies...
uv sync --inexact --no-install-package torch --no-install-package torchaudio --no-install-package torchvision
if errorlevel 1 pause & exit /b %errorlevel%

call :install_torch %TORCH_BACKEND%
if errorlevel 1 pause & exit /b %errorlevel%

call :install_compile_accel %TORCH_BACKEND%
if errorlevel 1 pause & exit /b %errorlevel%

echo [install] Verifying Python/Torch runtime...
uv run --no-sync python -c "import sys, torch; backend='%TORCH_BACKEND%'; cuda=torch.cuda.is_available(); print('[torch]', torch.__version__); print('[cuda_available]', cuda); print('[cuda_version]', torch.version.cuda); print('[device]', torch.cuda.get_device_name(0) if cuda else 'cpu'); sys.exit(1 if backend.startswith('cu') and not cuda else 0)"
if errorlevel 1 pause & exit /b %errorlevel%

if "%TORCH_BACKEND%"=="directml" (
    uv run --no-sync python -c "import torch_directml; print('[directml]', torch_directml.device())"
    if errorlevel 1 pause & exit /b %errorlevel%
)

echo.
echo [DONE] Install completed. Run start.bat
pause
exit /b 0

:install_torch
set "BACKEND=%~1"
if "%BACKEND%"=="cpu" (
    echo [install] Installing PyTorch CPU wheels...
    uv pip install --reinstall torch==2.7.1 torchaudio==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
    exit /b %errorlevel%
)
if "%BACKEND%"=="cu118" (
    echo [install] Installing PyTorch CUDA 11.8 wheels...
    uv pip install --reinstall torch==2.7.1 torchaudio==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
    exit /b %errorlevel%
)
if "%BACKEND%"=="cu126" (
    echo [install] Installing PyTorch CUDA 12.6 wheels...
    uv pip install --reinstall torch==2.7.1 torchaudio==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
    exit /b %errorlevel%
)
if "%BACKEND%"=="cu128" (
    echo [install] Installing PyTorch CUDA 12.8 wheels...
    uv pip install --reinstall torch==2.7.1 torchaudio==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
    exit /b %errorlevel%
)
if "%BACKEND%"=="directml" (
    echo [install] Installing PyTorch CPU + DirectML runtime...
    uv pip install --reinstall torch==2.7.1 torchaudio==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
    if errorlevel 1 exit /b %errorlevel%
    uv pip install --reinstall torch-directml
    exit /b %errorlevel%
)
echo [ERROR] Unknown backend: %BACKEND%
exit /b 1

:install_compile_accel
set "BACKEND=%~1"
if "%BACKEND:~0,2%"=="cu" (
    echo [install] Installing Triton for torch.compile...
    uv pip install "triton-windows>=3.0.0,<3.4"
    exit /b %errorlevel%
)
echo [install] Skipping Triton; torch.compile acceleration is CUDA-only in this installer.
exit /b 0

:detect_backend
set "TORCH_BACKEND=cpu"
set "GPU_NAME="
where nvidia-smi >nul 2>nul
if errorlevel 1 (
    echo [install] No NVIDIA GPU detected. Falling back to CPU.
    exit /b 0
)
for /f "usebackq delims=" %%G in (`nvidia-smi --query-gpu=name --format=csv^,noheader 2^>nul`) do (
    set "GPU_NAME=%%G"
    goto :got_gpu
)
:got_gpu
echo [install] Detected GPU: !GPU_NAME!
echo !GPU_NAME! | findstr /i /c:"GTX 10" >nul && set "TORCH_BACKEND=cu118" && exit /b 0
echo !GPU_NAME! | findstr /i /c:"RTX 20" /c:"RTX 30" >nul && set "TORCH_BACKEND=cu126" && exit /b 0
echo !GPU_NAME! | findstr /i /c:"RTX 40" /c:"RTX 50" >nul && set "TORCH_BACKEND=cu128" && exit /b 0
set "TORCH_BACKEND=cu128"
exit /b 0

:refresh_uv_path
set "PATH=%USERPROFILE%\.local\bin;%USERPROFILE%\.cargo\bin;%APPDATA%\uv\bin;%LOCALAPPDATA%\uv\bin;%LOCALAPPDATA%\Programs\uv;%PATH%"
exit /b 0

:ensure_uv
call :refresh_uv_path
where uv >nul 2>nul
if %errorlevel% equ 0 (
    for /f "delims=" %%V in ('uv --version 2^>nul') do echo [install] %%V
    exit /b 0
)

echo [install] uv not found. Installing uv...
where winget >nul 2>nul
if %errorlevel% equ 0 (
    winget install --id astral-sh.uv --exact --silent --accept-source-agreements --accept-package-agreements
) else (
    echo [install] winget not found. Using official uv installer...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
)

if %errorlevel% neq 0 (
    echo [install] First uv install attempt failed. Trying official uv installer...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
)

if %errorlevel% neq 0 (
    echo [ERROR] uv installation failed. Install it manually from https://astral.sh/uv/ and rerun install.bat.
    pause
    exit /b 1
)

call :refresh_uv_path
where uv >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] uv was installed but is still not visible in PATH for this session.
    echo [ERROR] Close this terminal, open a new one, and run install.bat again.
    pause
    exit /b 1
)
for /f "delims=" %%V in ('uv --version 2^>nul') do echo [install] %%V ready.
exit /b 0
