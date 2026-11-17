# SPDX-FileCopyrightText: 2025-2026 Igal Alkon
# SPDX-FileCopyrightText: 2026 ALKONTEK <git@alkontek.com>
# SPDX-License-Identifier: BSD-3-Clause

# Cutoff neighbor lists, shifted-force pairs, and span kinetic energy.
# Loaded as alkontek/md. Does not register a backend or width.

set(VECMAT_MODULE_KIND feature)

set(_md "${CMAKE_CURRENT_LIST_DIR}")

function(vecmat_module_alkontek_md_sources name)
    target_sources(${name} PRIVATE
            ${_md}/src/lj_tile.c
            ${_md}/src/coulomb_tile.c
            ${_md}/src/neighbors.c
            ${_md}/src/kinetic.c)
    target_include_directories(${name} PRIVATE ${_md}/include ${_md}/src)
    if(NOT MSVC)
        set_source_files_properties(
                ${_md}/src/lj_tile.c
                ${_md}/src/coulomb_tile.c
                ${_md}/src/neighbors.c
                ${_md}/src/kinetic.c
                TARGET_DIRECTORY ${name}
                PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
    endif()
    # Same host guard Vecmat uses for its AVX2 sources: x86-64 only.
    if(VECMAT_ENABLE_AVX2 AND NOT MSVC AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
        target_sources(${name} PRIVATE
                ${_md}/src/lj_tile_avx2.c
                ${_md}/src/coulomb_tile_avx2.c)
        set_source_files_properties(
                ${_md}/src/lj_tile_avx2.c
                ${_md}/src/coulomb_tile_avx2.c
                TARGET_DIRECTORY ${name}
                PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-ffp-contract=off")
        target_compile_definitions(${name} PRIVATE VECMAT_MD_AVX2)
    endif()
    if(VECMAT_ENABLE_NEON AND NOT MSVC AND CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|ARM64|arm64")
        target_sources(${name} PRIVATE
                ${_md}/src/lj_tile_neon.c
                ${_md}/src/coulomb_tile_neon.c)
        set_source_files_properties(
                ${_md}/src/lj_tile_neon.c
                ${_md}/src/coulomb_tile_neon.c
                TARGET_DIRECTORY ${name}
                PROPERTIES COMPILE_OPTIONS "-march=armv8-a+simd;-ffp-contract=off")
        target_compile_definitions(${name} PRIVATE VECMAT_MD_NEON)
    endif()
endfunction()

function(vecmat_module_alkontek_md_tests)
    foreach(_t vecmat_tests vecmat_benchmarks)
        if(TARGET ${_t})
            target_include_directories(${_t} PRIVATE
                    ${_md}/include ${CMAKE_SOURCE_DIR}/test)
        endif()
    endforeach()
    if(NOT MSVC AND TARGET vecmat_tests)
        set_source_files_properties(
                ${_md}/test/test_md.c
                ${_md}/test/test_md_neighbors.c
                ${_md}/test/test_md_list.c
                ${_md}/test/test_md_kinetic.c
                TARGET_DIRECTORY vecmat_tests
                PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
    endif()
endfunction()

list(APPEND VECMAT_MODULE_HOOKS vecmat_module_alkontek_md_sources)
list(APPEND VECMAT_MODULE_TEST_HOOKS vecmat_module_alkontek_md_tests)
list(APPEND VECMAT_MODULE_TEST_SOURCES
        ${_md}/test/test_md.c
        ${_md}/test/test_md_neighbors.c
        ${_md}/test/test_md_list.c
        ${_md}/test/test_md_kinetic.c)
