@echo off
rem One-shot build: third-party deps (if missing) -> psdpreview_native.dll -> C# library + demo.
setlocal
call "%~dp0scripts\setup-deps.cmd" || exit /b 1
call "%~dp0scripts\build-native.cmd" || exit /b 1
dotnet build "%~dp0PsdPreview.sln" -c Release || exit /b 1
echo.
echo Built: %~dp0samples\PsdPreview.Demo\bin\Release\net8.0\
