@echo off
rem Runs the given command inside a VS 2026 developer environment (with Ninja on PATH) for
rem x64, or for ARM64 when PP_ARCH=arm64 (cross-compiling from x64, or native on ARM64).
if defined VSCMD_VER goto run
set "PATH=C:\Program Files (x86)\Microsoft Visual Studio\Installer;%PATH%"
set "VCVARS=vcvars64.bat"
if /i "%PP_ARCH%"=="arm64" (
    if /i "%PROCESSOR_ARCHITECTURE%"=="ARM64" (set "VCVARS=vcvarsarm64.bat") else (set "VCVARS=vcvarsamd64_arm64.bat")
)
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\%VCVARS%" >nul || exit /b 1
set "PATH=C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;%PATH%"
:run
%*
