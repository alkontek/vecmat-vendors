# SPDX-FileCopyrightText: 2025-2026 Igal Alkon
# SPDX-FileCopyrightText: 2026 ALKONTEK <git@alkontek.com>
# SPDX-License-Identifier: BSD-3-Clause

# Intel fp16 GEMM override. Not a backend: it does not fill vm_backend_ops.
# Loaded as intel/fp16. Requires VECMAT_ENABLE_FP16.

set(VECMAT_MODULE_KIND width)
if(NOT VECMAT_ENABLE_FP16)
    message(FATAL_ERROR "intel/fp16 needs VECMAT_ENABLE_FP16")
endif()

set(_intel_fp16 "${CMAKE_CURRENT_LIST_DIR}")

function(vecmat_module_intel_fp16_sources name)
    target_sources(${name} PRIVATE
            ${_intel_fp16}/src/probe.c
            ${_intel_fp16}/src/f16c.c
            ${_intel_fp16}/src/avx512fp16.c
            ${_intel_fp16}/src/register.c)
    target_include_directories(${name} PRIVATE
            ${_intel_fp16}/include ${_intel_fp16}/src)
    if(NOT MSVC AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
        set_source_files_properties(${_intel_fp16}/src/f16c.c
                TARGET_DIRECTORY ${name}
                PROPERTIES COMPILE_OPTIONS "-mf16c;-mavx;-mfma")
        set_source_files_properties(${_intel_fp16}/src/avx512fp16.c
                TARGET_DIRECTORY ${name}
                PROPERTIES COMPILE_OPTIONS "-mavx512fp16;-mavx512bw;-mavx512vl;-mf16c")
    endif()
endfunction()

function(vecmat_module_intel_fp16_tests)
    foreach(_t vecmat_tests vecmat_benchmarks)
        if(TARGET ${_t})
            target_include_directories(${_t} PRIVATE
                    ${_intel_fp16}/include ${CMAKE_SOURCE_DIR}/test)
        endif()
    endforeach()
endfunction()

list(APPEND VECMAT_FP16_MODULE_HOOKS vecmat_module_intel_fp16_sources)
list(APPEND VECMAT_MODULE_TEST_HOOKS vecmat_module_intel_fp16_tests)
list(APPEND VECMAT_MODULE_TEST_SOURCES ${_intel_fp16}/test/test_intel_fp16.c)
string(APPEND VECMAT_FP16_MODULE_REGISTER_INCLUDES "#include <vecmat_intel_fp16.h>\n")
string(APPEND VECMAT_FP16_MODULE_REGISTER_CALLS "    vm_intel_fp16_register();\n")
