# SPDX-FileCopyrightText: 2025-2026 Igal Alkon
# SPDX-FileCopyrightText: 2026 ALKONTEK <git@alkontek.com>
# SPDX-License-Identifier: BSD-3-Clause

# Intel bf16 GEMM override. Not a backend: it does not fill vm_backend_ops.
# Loaded as intel/bf16. Requires VECMAT_ENABLE_BF16.

set(VECMAT_MODULE_KIND width)
if(NOT VECMAT_ENABLE_BF16)
    message(FATAL_ERROR "intel/bf16 needs VECMAT_ENABLE_BF16")
endif()

set(_intel_bf16 "${CMAKE_CURRENT_LIST_DIR}")

function(vecmat_module_intel_bf16_sources name)
    target_sources(${name} PRIVATE
            ${_intel_bf16}/src/probe.c
            ${_intel_bf16}/src/avx512bf16.c
            ${_intel_bf16}/src/register.c)
    target_include_directories(${name} PRIVATE
            ${_intel_bf16}/include ${_intel_bf16}/src)
    if(NOT MSVC AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
        set_source_files_properties(${_intel_bf16}/src/avx512bf16.c
                TARGET_DIRECTORY ${name}
                PROPERTIES COMPILE_OPTIONS "-mavx512bf16;-mavx512bw;-mavx512vl")
    endif()
endfunction()

function(vecmat_module_intel_bf16_tests)
    foreach(_t vecmat_tests vecmat_benchmarks)
        if(TARGET ${_t})
            target_include_directories(${_t} PRIVATE
                    ${_intel_bf16}/include ${CMAKE_SOURCE_DIR}/test)
        endif()
    endforeach()
endfunction()

list(APPEND VECMAT_BF16_MODULE_HOOKS vecmat_module_intel_bf16_sources)
list(APPEND VECMAT_MODULE_TEST_HOOKS vecmat_module_intel_bf16_tests)
list(APPEND VECMAT_MODULE_TEST_SOURCES ${_intel_bf16}/test/test_intel_bf16.c)
string(APPEND VECMAT_BF16_MODULE_REGISTER_INCLUDES "#include <vecmat_intel_bf16.h>\n")
string(APPEND VECMAT_BF16_MODULE_REGISTER_CALLS "    vm_intel_bf16_register();\n")
