cmake_minimum_required(VERSION 3.24)
project(PsdPreview LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Static CRT (/MT) everywhere: psdpreview_native.dll must not depend on the VC++ runtime
# DLLs (only system DLLs, plus the delay-loaded NPP DLLs). cudart_static and the IPP/mozjpeg
# static libs are all /MT-compatible.
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

set(PSD_SDK_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/psd_sdk/src/Psd" CACHE PATH "psd_sdk/src/Psd directory (git submodule)")
set(IPP_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ipp-static-pkg/build/native" CACHE PATH "Intel IPP (static NuGet layout)")
set(MOZJPEG_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/mozjpeg" CACHE PATH "mozjpeg install prefix")
set(LCMS2_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/lcms2-src" CACHE PATH "Little CMS 2 source tree")

if(NOT EXISTS "${PSD_SDK_DIR}/PsdDecompressRle.cpp")
    message(FATAL_ERROR "psd_sdk not found at ${PSD_SDK_DIR}: run `git submodule update --init` (or scripts\\setup-deps.cmd)")
endif()

find_package(CUDAToolkit REQUIRED)

# psd_sdk: document/section parser + PackBits decoder (see native/src/psd.cpp).
# Logging is compiled out so the DLL never prints to the host's stdout.
add_library(psd_sdk_min STATIC
    ${PSD_SDK_DIR}/PsdParseDocument.cpp
    ${PSD_SDK_DIR}/PsdDecompressRle.cpp
    ${PSD_SDK_DIR}/PsdFile.cpp
    ${PSD_SDK_DIR}/PsdAllocator.cpp
    ${PSD_SDK_DIR}/PsdMallocAllocator.cpp
    ${PSD_SDK_DIR}/PsdSyncFileReader.cpp)
target_include_directories(psd_sdk_min PUBLIC ${PSD_SDK_DIR})
target_compile_definitions(psd_sdk_min PUBLIC PSD_ENABLE_LOGGING=0)
target_compile_options(psd_sdk_min PRIVATE /W1 /utf-8)

# Little CMS 2, built from source as a static library.
file(GLOB LCMS2_SOURCES "${LCMS2_DIR}/src/*.c")
add_library(lcms2_static STATIC ${LCMS2_SOURCES})
target_include_directories(lcms2_static PUBLIC ${LCMS2_DIR}/include)
target_compile_definitions(lcms2_static PRIVATE _CRT_SECURE_NO_WARNINGS)
target_compile_options(lcms2_static PRIVATE /W1)

# SIMD kernels: one translation unit per instruction set, dispatched at runtime.
set(PP_SIMD_SOURCES
    native/src/simd.h native/src/simd.cpp native/src/simd_kernels.h
    native/src/simd_sse2.cpp native/src/simd_avx2.cpp native/src/simd_avx512.cpp)
set_source_files_properties(native/src/simd_avx2.cpp PROPERTIES COMPILE_OPTIONS /arch:AVX2)
set_source_files_properties(native/src/simd_avx512.cpp PROPERTIES COMPILE_OPTIONS /arch:AVX512)

add_library(psdpreview SHARED
    native/include/psdpreview.h
    native/src/common.h
    native/src/psd.h native/src/psd.cpp
    native/src/cpu_resize.h native/src/cpu_resize.cpp
    native/src/gpu.h native/src/gpu.cpp
    native/src/jpeg.h native/src/jpeg.cpp
    native/src/color.h native/src/color.cpp
    ${PP_SIMD_SOURCES}
    native/src/engine.cpp)
target_include_directories(psdpreview PRIVATE
    native/include
    ${IPP_ROOT}/include/ipp
    ${MOZJPEG_ROOT}/include)
target_compile_definitions(psdpreview PRIVATE
    PSDPREVIEW_BUILD UNICODE _UNICODE
    "PP_NPP_DLL_SUFFIX=\"${CUDAToolkit_VERSION_MAJOR}\"")
target_compile_options(psdpreview PRIVATE /W4 /utf-8 /permissive- /EHsc /Zi)
target_link_libraries(psdpreview PRIVATE
    psd_sdk_min
    lcms2_static
    CUDA::cudart_static CUDA::nppig CUDA::nppc
    ${IPP_ROOT}/win-x64/ippimt.lib ${IPP_ROOT}/win-x64/ippsmt.lib
    ${IPP_ROOT}/win-x64/ippvmmt.lib ${IPP_ROOT}/win-x64/ippcoremt.lib
    ${MOZJPEG_ROOT}/lib/jpeg-static.lib
    delayimp)
# NPP is delay-loaded so the DLL still loads (CPU path) where CUDA isn't installed.
target_link_options(psdpreview PRIVATE
    /DEBUG /OPT:REF /OPT:ICF
    /DELAYLOAD:nppig64_${CUDAToolkit_VERSION_MAJOR}.dll
    /DELAYLOAD:nppc64_${CUDAToolkit_VERSION_MAJOR}.dll)

# Ship the two NPP runtime DLLs next to psdpreview_native.dll.
file(GLOB NPP_RUNTIME_DLLS
    "${CUDAToolkit_BIN_DIR}/nppc64_*.dll" "${CUDAToolkit_BIN_DIR}/nppig64_*.dll"
    "${CUDAToolkit_BIN_DIR}/x64/nppc64_*.dll" "${CUDAToolkit_BIN_DIR}/x64/nppig64_*.dll")
add_custom_command(TARGET psdpreview POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${NPP_RUNTIME_DLLS} $<TARGET_FILE_DIR:psdpreview>)

# Distinct from the managed PsdPreview.dll (file names are case-insensitive on Windows).
set_target_properties(psdpreview PROPERTIES OUTPUT_NAME psdpreview_native)

# Tests (not built by default): scripts\build-native.cmd then
#   scripts\vsenv.cmd cmake --build build/native --target psdpreview_tests
# simd: kernels at every level vs. scalar; sdk: patched psd_sdk vs. our decoder; bench: decode speed per level.
add_executable(psdpreview_tests EXCLUDE_FROM_ALL
    tests/psdpreview_tests.cpp
    native/src/psd.cpp ${PP_SIMD_SOURCES}
    ${PSD_SDK_DIR}/PsdParseImageDataSection.cpp
    ${PSD_SDK_DIR}/PsdNativeFile.cpp
    ${PSD_SDK_DIR}/PsdMemoryFile.cpp)
target_include_directories(psdpreview_tests PRIVATE native/include native/src)
target_compile_definitions(psdpreview_tests PRIVATE PSDPREVIEW_BUILD UNICODE _UNICODE)
target_compile_options(psdpreview_tests PRIVATE /utf-8 /EHsc /permissive-)
target_link_libraries(psdpreview_tests PRIVATE psd_sdk_min)
