@echo off
rem Regenerates native\src\gpu_resize.ptx from native\src\gpu_resize.cu (run after editing the
rem kernel; the normal build only embeds the checked-in .ptx and needs no nvcc).
rem
rem compute_75 PTX is JIT-compiled by the driver for whatever GPU it runs on (Turing and newer,
rem everything CUDA 13 supports). nvcc 13.0 can't use VS 2026 as its host compiler (the 14.5x
rem STL needs CUDA 13.2+), so this picks VS 2019/2022 C++ tools, e.g. VS 2019 Build Tools.
setlocal
set "ROOT=%~dp0.."
if not defined CUDA_PATH (echo CUDA_PATH is not set - install the CUDA Toolkit & exit /b 1)
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
set "VSHOST="
for /f "usebackq delims=" %%i in (`"%VSWHERE%" -version [16.0^,18.0^) -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -latest -property installationPath`) do set "VSHOST=%%i"
if not defined VSHOST (echo No VS 2019/2022 C++ tools found for nvcc's host compiler & exit /b 1)
rem vcvars64 chokes on some inherited PATH entries; start from a minimal PATH.
set "PATH=%SystemRoot%\System32;%SystemRoot%;%SystemRoot%\System32\WindowsPowerShell\v1.0;%ProgramFiles(x86)%\Microsoft Visual Studio\Installer"
call "%VSHOST%\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
"%CUDA_PATH%\bin\nvcc.exe" -ptx -arch=compute_75 -O3 -o "%ROOT%\native\src\gpu_resize.ptx" "%ROOT%\native\src\gpu_resize.cu" || exit /b 1
echo Wrote native\src\gpu_resize.ptx
