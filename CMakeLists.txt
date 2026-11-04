cmake_minimum_required(VERSION 3.24)
project(PsdPreview LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Static CRT (/MT) everywhere: psdpreview_native.dll must not depend on the VC++ runtime
# DLLs (only system DLLs). cudart_static and the IPP/mozjpeg static libs are all /MT-compatible.
#
# x64 or ARM64, whichever the compiler targets (scripts\build-native.cmd [arm64]):
#   x64    CUDA + D3D12 GPU backends, IPP resize, SSE2/AVX2/AVX-512 kernels
#   ARM64  D3D12 GPU backend (no CUDA on Windows on Arm), stb_image_resize2 resize, Neon kernels
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

if(CMAKE_CXX_COMPILER_ARCHITECTURE_ID STREQUAL "ARM64")
    set(PP_ARM64 ON)
    set(PP_MOZJPEG_DEFAULT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/mozjpeg-arm64")
else()
    set(PP_ARM64 OFF)
    set(PP_MOZJPEG_DEFAULT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/mozjpeg")
endif()
option(PP_WITH_CUDA "CUDA backend (x64 only)" ON)
if(PP_ARM64)
    set(PP_WITH_CUDA OFF)
endif()

set(PSD_SDK_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/psd_sdk/src/Psd" CACHE PATH "psd_sdk/src/Psd directory (git submodule)")
set(IPP_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ipp-static-pkg/build/native" CACHE PATH "Intel IPP (static NuGet layout)")
set(MOZJPEG_ROOT "${PP_MOZJPEG_DEFAULT}" CACHE PATH "mozjpeg install prefix (built for the target architecture)")
set(LCMS2_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/lcms2-src" CACHE PATH "Little CMS 2 source tree")

if(NOT EXISTS "${PSD_SDK_DIR}/PsdDecompressRle.cpp")
    message(FATAL_ERROR "psd_sdk not found at ${PSD_SDK_DIR}: run `git submodule update --init` (or scripts\\setup-deps.cmd)")
endif()

if(PP_WITH_CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()

# dxc from the Windows SDK compiles the D3D12 kernels (always an x64 or ARM64 host tool;
# DXIL is the same for every target).
if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "ARM64|aarch64")
    set(PP_SDK_HOST arm64)
else()
    set(PP_SDK_HOST x64)
endif()
file(GLOB PP_DXC_CANDIDATES "$ENV{WindowsSdkVerBinPath}${PP_SDK_HOST}/dxc.exe"
     "C:/Program Files (x86)/Windows Kits/10/bin/10.*/${PP_SDK_HOST}/dxc.exe")
list(SORT PP_DXC_CANDIDATES COMPARE NATURAL ORDER DESCENDING)
find_program(PP_DXC NAMES dxc HINTS ${PP_DXC_CANDIDATES} NO_DEFAULT_PATH)
if(NOT PP_DXC AND PP_DXC_CANDIDATES)
    list(GET PP_DXC_CANDIDATES 0 PP_DXC)
endif()
if(NOT PP_DXC)
    message(FATAL_ERROR "dxc.exe (Windows SDK) not found")
endif()

# psd_sdk: document/section parser (see native/src/psd.cpp); its PackBits decoder is only used
# by the tests, as the reference for ours.
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
target_compile_options(lcms2_static PRIVATE /W1 /utf-8)

# SIMD kernels: one translation unit per instruction set, dispatched at runtime (x64), or Neon.
set(PP_SIMD_SOURCES native/src/simd.h native/src/simd.cpp native/src/simd_kernels.h)
if(PP_ARM64)
    list(APPEND PP_SIMD_SOURCES native/src/simd_neon.cpp)
else()
    list(APPEND PP_SIMD_SOURCES native/src/simd_sse2.cpp native/src/simd_avx2.cpp native/src/simd_avx512.cpp)
    set_source_files_properties(native/src/simd_avx2.cpp PROPERTIES COMPILE_OPTIONS /arch:AVX2)
    set_source_files_properties(native/src/simd_avx512.cpp PROPERTIES COMPILE_OPTIONS /arch:AVX512)
endif()

# D3D12 kernels: native/src/gpu_d3d12.hlsl -> DXIL byte arrays in generated headers.
set(PP_HLSL "${CMAKE_CURRENT_SOURCE_DIR}/native/src/gpu_d3d12.hlsl")
set(PP_DXIL_HEADERS)
# header name | entry point | defines
foreach(k "clear|pp_clear|PP_CLEAR" "unpack|pp_unpack|PP_UNPACK"
          "resize1|pp_resize|PP_RESIZE,NC=1" "resize3|pp_resize|PP_RESIZE,NC=3" "resize4|pp_resize|PP_RESIZE,NC=4"
          "resizergba|pp_resize|PP_RESIZE,NC=4,RGBA"
          "resize3t8|pp_resize|PP_RESIZE,NC=3,FIXED=8" "resize3t12|pp_resize|PP_RESIZE,NC=3,FIXED=12"
          "resize3t16|pp_resize|PP_RESIZE,NC=3,FIXED=16" "resize3t24|pp_resize|PP_RESIZE,NC=3,FIXED=24"
          "resizergbat8|pp_resize|PP_RESIZE,NC=4,RGBA,FIXED=8" "resizergbat12|pp_resize|PP_RESIZE,NC=4,RGBA,FIXED=12"
          "resizergbat16|pp_resize|PP_RESIZE,NC=4,RGBA,FIXED=16" "resizergbat24|pp_resize|PP_RESIZE,NC=4,RGBA,FIXED=24")
    string(REPLACE "|" ";" k "${k}")
    list(GET k 0 name)
    list(GET k 1 entry)
    list(GET k 2 defs)
    string(REPLACE "," ";" defs "${defs}")
    list(TRANSFORM defs PREPEND "-D")
    string(SUBSTRING "${name}" 0 1 first)
    string(TOUPPER "${first}" first)
    string(SUBSTRING "${name}" 1 -1 rest)
    set(header "${CMAKE_BINARY_DIR}/generated/d3d12_${name}.h")
    add_custom_command(OUTPUT "${header}"
        COMMAND "${PP_DXC}" -nologo -T cs_6_0 -E ${entry} ${defs} -O3 -Fh "${header}" -Vn k${first}${rest}Dxil "${PP_HLSL}"
        DEPENDS "${PP_HLSL}" VERBATIM COMMENT "dxc ${name}")
    list(APPEND PP_DXIL_HEADERS "${header}")
endforeach()
add_custom_target(psdpreview_dxil DEPENDS ${PP_DXIL_HEADERS})

set(PP_GPU_SOURCES native/src/gpu.h native/src/gpu.cpp native/src/gpu_d3d12.cpp native/src/gpu_d3d12.hlsl)
if(PP_WITH_CUDA)
    list(APPEND PP_GPU_SOURCES native/src/gpu_cuda.cpp native/src/gpu_cuda.cu)
endif()

add_library(psdpreview SHARED
    native/include/psdpreview.h
    native/src/common.h
    native/src/psd.h native/src/psd.cpp
    native/src/resize_filter.h
    native/src/cpu_resize.h native/src/cpu_resize.cpp
    ${PP_GPU_SOURCES}
    native/src/jpeg.h native/src/jpeg.cpp
    native/src/color.h native/src/color.cpp
    ${PP_SIMD_SOURCES}
    native/src/engine.cpp)
add_dependencies(psdpreview psdpreview_dxil)
set_source_files_properties(native/src/gpu_d3d12.hlsl PROPERTIES HEADER_FILE_ONLY ON)
if(PP_WITH_CUDA)
    # The CUDA kernels: native/src/gpu_cuda.ptx, generated from gpu_cuda.cu by scripts\build-ptx.cmd
    # and checked in (so the build needs no nvcc), embedded as a NUL-terminated byte array.
    set(PP_PTX "${CMAKE_CURRENT_SOURCE_DIR}/native/src/gpu_cuda.ptx")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${PP_PTX}")
    file(READ "${PP_PTX}" PP_PTX_HEX HEX)
    string(REGEX REPLACE "(................................................................)" "\\1\n" PP_PTX_HEX "${PP_PTX_HEX}")
    string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," PP_PTX_BYTES "${PP_PTX_HEX}")
    file(CONFIGURE OUTPUT "${CMAKE_BINARY_DIR}/generated/gpu_cuda_ptx.h" @ONLY
         CONTENT "// Generated by CMake from native/src/gpu_cuda.ptx.\nstatic const unsigned char kKernelsPtx[] = {\n@PP_PTX_BYTES@0};\n")
    set_source_files_properties(native/src/gpu_cuda.cu PROPERTIES HEADER_FILE_ONLY ON)
endif()

target_include_directories(psdpreview PRIVATE
    native/include
    native/vendor
    ${CMAKE_BINARY_DIR}/generated
    ${MOZJPEG_ROOT}/include)
target_compile_definitions(psdpreview PRIVATE PSDPREVIEW_BUILD UNICODE _UNICODE
    PP_HAVE_CUDA=$<BOOL:${PP_WITH_CUDA}> PP_HAVE_IPP=$<NOT:$<BOOL:${PP_ARM64}>>)
target_compile_options(psdpreview PRIVATE /W4 /utf-8 /permissive- /EHsc /Zi)
target_link_libraries(psdpreview PRIVATE
    psd_sdk_min
    lcms2_static
    d3d12 dxgi
    ${MOZJPEG_ROOT}/lib/jpeg-static.lib)
if(PP_WITH_CUDA)
    target_link_libraries(psdpreview PRIVATE CUDA::cudart_static)
endif()
if(NOT PP_ARM64)
    target_include_directories(psdpreview PRIVATE ${IPP_ROOT}/include/ipp)
    target_link_libraries(psdpreview PRIVATE
        ${IPP_ROOT}/win-x64/ippimt.lib ${IPP_ROOT}/win-x64/ippsmt.lib
        ${IPP_ROOT}/win-x64/ippvmmt.lib ${IPP_ROOT}/win-x64/ippcoremt.lib)
endif()
# cudart_static loads the driver (nvcuda.dll) at runtime, and d3d12.dll is part of Windows, so
# the DLL loads on any machine.
target_link_options(psdpreview PRIVATE /DEBUG /OPT:REF /OPT:ICF)

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
