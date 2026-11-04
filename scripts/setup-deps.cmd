@echo off
rem Downloads and prepares third_party\ (Intel IPP static NuGet package, mozjpeg built from source,
rem Little CMS 2 source - compiled by the main CMake build).
rem Needs: curl + tar (built into Windows 10+), CMake, NASM, VS 2026 C++ tools (+ their ARM64 tools for arm64).
rem Usage: setup-deps.cmd [arm64] [third_party_dir]
rem   arm64: also builds mozjpeg for ARM64 (Neon) into third_party\mozjpeg-arm64 (IPP is x64-only).
setlocal
rem %~dp0 is re-resolved against the current directory when the script was called by a relative
rem path, so it is wrong after the pushd below: keep it.
set "SCRIPTS=%~dp0"
set "ARM64="
if /i "%~1"=="arm64" (
    set "ARM64=1"
    shift /1
)
set "TP=%~1"
if "%TP%"=="" set "TP=%SCRIPTS%..\third_party"
set "IPP_VER=2026.1.0.761"
set "MOZJPEG_VER=4.1.5"
set "LCMS2_VER=2.20"
rem Use the Windows-bundled tools (a GNU tar on PATH would treat "C:" as a remote host).
set "TAR=%SystemRoot%\System32\tar.exe"
set "CURL=%SystemRoot%\System32\curl.exe"
if not exist "%TP%\dl" mkdir "%TP%\dl"

rem psd_sdk is a git submodule (github.com/pig1800/psd_sdk, the fork with the fixes PsdPreview needs).
if exist "%SCRIPTS%..\external\psd_sdk\src\Psd\PsdDecompressRle.cpp" goto ipp
echo === psd_sdk submodule
git -C "%SCRIPTS%.." submodule update --init --recursive || exit /b 1

:ipp

if exist "%TP%\ipp-static-pkg\build\native\win-x64\ippimt.lib" goto mozjpeg
echo === Intel IPP %IPP_VER%
"%CURL%" -fsSL -o "%TP%\dl\ipp-static.nupkg" "https://api.nuget.org/v3-flatcontainer/intelipp.static.win-x64/%IPP_VER%/intelipp.static.win-x64.%IPP_VER%.nupkg" || exit /b 1
mkdir "%TP%\ipp-static-pkg" 2>nul
"%TAR%" -xf "%TP%\dl\ipp-static.nupkg" -C "%TP%\ipp-static-pkg" || exit /b 1

:mozjpeg
rem Built against the static CRT (/MT) like everything else, so psdpreview_native.dll needs no VC++ runtime.
rem The marker file tells a static-CRT build apart from an older /MD one, which gets rebuilt.
if exist "%TP%\mozjpeg\static-crt.txt" goto lcms2
if exist "%TP%\mozjpeg" rmdir /s /q "%TP%\mozjpeg"
echo === mozjpeg %MOZJPEG_VER%
if exist "%TP%\mozjpeg-src\CMakeLists.txt" goto mozjpeg_build
"%CURL%" -fsSL -o "%TP%\dl\mozjpeg.tar.gz" "https://github.com/mozilla/mozjpeg/archive/refs/tags/v%MOZJPEG_VER%.tar.gz" || exit /b 1
mkdir "%TP%\mozjpeg-src" 2>nul
"%TAR%" -xzf "%TP%\dl\mozjpeg.tar.gz" -C "%TP%\mozjpeg-src" --strip-components=1 || exit /b 1
:mozjpeg_build
call "%SCRIPTS%vsenv.cmd" cmake -S "%TP%\mozjpeg-src" -B "%TP%\mozjpeg-build" -G Ninja -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DENABLE_SHARED=OFF -DENABLE_STATIC=ON -DWITH_TURBOJPEG=OFF ^
    -DPNG_SUPPORTED=OFF -DWITH_CRT_DLL=OFF -DREQUIRE_SIMD=ON "-DCMAKE_INSTALL_PREFIX=%TP%\mozjpeg" || exit /b 1
call "%SCRIPTS%vsenv.cmd" cmake --build "%TP%\mozjpeg-build" --target install || exit /b 1
rmdir /s /q "%TP%\mozjpeg-build"
echo mozjpeg %MOZJPEG_VER% built with the static CRT (/MT)> "%TP%\mozjpeg\static-crt.txt"

:lcms2
if exist "%TP%\lcms2-src\include\lcms2.h" goto mozjpeg_arm64
echo === Little CMS %LCMS2_VER%
"%CURL%" -fsSL -o "%TP%\dl\lcms2.tar.gz" "https://github.com/mm2/Little-CMS/archive/refs/tags/lcms%LCMS2_VER%.tar.gz" || exit /b 1
mkdir "%TP%\lcms2-src" 2>nul
"%TAR%" -xzf "%TP%\dl\lcms2.tar.gz" -C "%TP%\lcms2-src" --strip-components=1 || exit /b 1

:mozjpeg_arm64
if not defined ARM64 goto done
if exist "%TP%\mozjpeg-arm64\neon-fdct-fixed.txt" goto done
if exist "%TP%\mozjpeg-arm64" rmdir /s /q "%TP%\mozjpeg-arm64"
echo === mozjpeg %MOZJPEG_VER% for ARM64
rem Fix for the Neon forward DCT, which overflows on blocks that mozjpeg's deringing pushed
rem past 255 (see the patch). git apply works outside a repository; the ceiling stops it from
rem taking the enclosing checkout for one. Skipped if already applied.
pushd "%TP%\mozjpeg-src" || exit /b 1
set "GIT_CEILING_DIRECTORIES=%TP%"
git apply --reverse --check -p0 "%SCRIPTS%mozjpeg-neon-fdct.patch" 2>nul || git apply -p0 "%SCRIPTS%mozjpeg-neon-fdct.patch" || (popd & exit /b 1)
set "GIT_CEILING_DIRECTORIES="
popd
rem Cross-compiled; CMAKE_SYSTEM_PROCESSOR makes mozjpeg build its Neon intrinsics.
set "PP_ARCH=arm64"
call "%SCRIPTS%vsenv.cmd" cmake -S "%TP%\mozjpeg-src" -B "%TP%\mozjpeg-build-arm64" -G Ninja -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_SYSTEM_NAME=Windows -DCMAKE_SYSTEM_PROCESSOR=ARM64 ^
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DENABLE_SHARED=OFF -DENABLE_STATIC=ON -DWITH_TURBOJPEG=OFF ^
    -DPNG_SUPPORTED=OFF -DWITH_CRT_DLL=OFF -DREQUIRE_SIMD=ON "-DCMAKE_INSTALL_PREFIX=%TP%\mozjpeg-arm64" || exit /b 1
call "%SCRIPTS%vsenv.cmd" cmake --build "%TP%\mozjpeg-build-arm64" --target install || exit /b 1
rmdir /s /q "%TP%\mozjpeg-build-arm64"
echo mozjpeg %MOZJPEG_VER% for ARM64 built with the static CRT (/MT)> "%TP%\mozjpeg-arm64\static-crt.txt"
echo with scripts\mozjpeg-neon-fdct.patch> "%TP%\mozjpeg-arm64\neon-fdct-fixed.txt"

:done
rmdir /s /q "%TP%\dl" 2>nul
echo third_party ready: %TP%
