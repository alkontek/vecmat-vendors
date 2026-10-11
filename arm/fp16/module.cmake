# SPDX-FileCopyrightText: 2025-2026 Igal Alkon
# SPDX-FileCopyrightText: 2026 ALKONTEK <git@alkontek.com>
# SPDX-License-Identifier: BSD-3-Clause

# Arm fp16 GEMM override. Not a backend: it does not fill vm_backend_ops.
# Loaded as arm/fp16. Requires VECMAT_ENABLE_FP16.

set(VECMAT_MODULE_KIND width)
if(NOT VECMAT_ENABLE_FP16)
    message(FATAL_ERROR "arm/fp16 needs VECMAT_ENABLE_FP16")
endif()

set(_arm_fp16 "${CMAKE_CURRENT_LIST_DIR}")

function(vecmat_module_arm_fp16_sources name)
    target_sources(${name} PRIVATE
            ${_arm_fp16}/src/probe.c
            ${_arm_fp16}/src/neon.c
            ${_arm_fp16}/src/register.c)
    target_include_directories(${name} PRIVATE
            ${_arm_fp16}/include ${_arm_fp16}/src)
    if(NOT MSVC AND CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|ARM64|arm64")
        set_source_files_properties(${_arm_fp16}/src/neon.c
                TARGET_DIRECTORY ${name}
                PROPERTIES COMPILE_OPTIONS "-march=armv8.2-a+fp16+fp16fml")
    endif()
endfunction()

function(vecmat_module_arm_fp16_tests)
    foreach(_t vecmat_tests vecmat_benchmarks)
        if(TARGET ${_t})
            target_include_directories(${_t} PRIVATE
                    ${_arm_fp16}/include ${CMAKE_SOURCE_DIR}/test)
        endif()
    endforeach()
endfunction()

list(APPEND VECMAT_FP16_MODULE_HOOKS vecmat_module_arm_fp16_sources)
list(APPEND VECMAT_MODULE_TEST_HOOKS vecmat_module_arm_fp16_tests)
list(APPEND VECMAT_MODULE_TEST_SOURCES ${_arm_fp16}/test/test_arm_fp16.c)
string(APPEND VECMAT_FP16_MODULE_REGISTER_INCLUDES "#include <vecmat_arm_fp16.h>\n")
string(APPEND VECMAT_FP16_MODULE_REGISTER_CALLS "    vm_arm_fp16_register();\n")
