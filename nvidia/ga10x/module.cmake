# SPDX-FileCopyrightText: 2025-2026 Igal Alkon
# SPDX-FileCopyrightText: 2026 ALKONTEK <git@alkontek.com>
# SPDX-License-Identifier: BSD-3-Clause

# NVIDIA GA10x device. Loaded as nvidia/ga10x.
# Its own CUDA library: not a float object, and not a CPU backend.
# Does not append VECMAT_MODULE_HOOKS or a register list.
# The test calls vm_nvidia_ga10x_register() itself.

set(VECMAT_MODULE_KIND device)

set(_ga10x "${CMAKE_CURRENT_LIST_DIR}")

if(NOT VECMAT_BUILD_ABI_32)
    message(FATAL_ERROR "nvidia/ga10x needs VECMAT_FLOAT_ABI=32 or BOTH")
endif()

include(CheckLanguage)
check_language(CUDA)
if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR "nvidia/ga10x needs the CUDA toolkit")
endif()
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

add_library(vecmat_nvidia_ga10x STATIC
        ${_ga10x}/src/register.c
        ${_ga10x}/src/ga10x.cu)
target_include_directories(vecmat_nvidia_ga10x PRIVATE
        ${_ga10x}/include
        ${CMAKE_SOURCE_DIR}/include)
set_target_properties(vecmat_nvidia_ga10x PROPERTIES
        CUDA_ARCHITECTURES 86
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        C_STANDARD 11
        C_STANDARD_REQUIRED ON
        POSITION_INDEPENDENT_CODE ON)
target_link_libraries(vecmat_nvidia_ga10x PUBLIC CUDA::cudart)

function(vecmat_module_nvidia_ga10x_tests)
    if(NOT TARGET vecmat_tests)
        return()
    endif()
    target_include_directories(vecmat_tests PRIVATE
            ${_ga10x}/include ${CMAKE_SOURCE_DIR}/test)
    if(MSVC)
        target_link_libraries(vecmat_tests PRIVATE vecmat_nvidia_ga10x vecmat_static)
    else()
        # vm_device_register lives in vecmat_static. Rescan so this archive can see it.
        target_link_libraries(vecmat_tests PRIVATE
                "-Wl,--start-group" vecmat_nvidia_ga10x vecmat_static "-Wl,--end-group")
    endif()
endfunction()

list(APPEND VECMAT_MODULE_TEST_HOOKS vecmat_module_nvidia_ga10x_tests)
list(APPEND VECMAT_MODULE_TEST_SOURCES ${_ga10x}/test/test_ga10x.c)
