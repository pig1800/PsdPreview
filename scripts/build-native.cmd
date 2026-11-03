@echo off
rem Configure + build psdpreview.dll (Release) into build\native\bin.
setlocal
set "ROOT=%~dp0.."
call "%~dp0vsenv.cmd" cmake -S "%ROOT%" -B "%ROOT%\build\native" -G Ninja -DCMAKE_BUILD_TYPE=Release || exit /b 1
call "%~dp0vsenv.cmd" cmake --build "%ROOT%\build\native" || exit /b 1
