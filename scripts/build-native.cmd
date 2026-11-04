@echo off
rem Configure + build psdpreview_native.dll (Release):
rem   build-native.cmd         x64   -> build\native\bin
rem   build-native.cmd arm64   ARM64 -> build\native-arm64\bin (run setup-deps.cmd arm64 first)
setlocal
set "ROOT=%~dp0.."
set "OUT=%ROOT%\build\native"
set "PP_ARCH="
if /i "%~1"=="arm64" (
    set "PP_ARCH=arm64"
    set "OUT=%ROOT%\build\native-arm64"
)
call "%~dp0vsenv.cmd" cmake -S "%ROOT%" -B "%OUT%" -G Ninja -DCMAKE_BUILD_TYPE=Release || exit /b 1
call "%~dp0vsenv.cmd" cmake --build "%OUT%" || exit /b 1
