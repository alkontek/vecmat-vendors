# SPDX-FileCopyrightText: 2025-2026 Igal Alkon
# SPDX-FileCopyrightText: 2026 ALKONTEK <git@alkontek.com>
# SPDX-License-Identifier: BSD-3-Clause

# Spectral-graph feature: bitsets and combinatorial unrank.
# Loaded as alkontek/spectral-graph. Does not register a backend or width.

set(VECMAT_MODULE_KIND feature)

set(_sg "${CMAKE_CURRENT_LIST_DIR}")

function(vecmat_module_alkontek_spectral_graph_sources name)
    target_sources(${name} PRIVATE
            ${_sg}/src/bitset.c
            ${_sg}/src/unrank.c
            ${_sg}/src/laplacian.c)
    target_include_directories(${name} PRIVATE ${_sg}/include)
endfunction()

function(vecmat_module_alkontek_spectral_graph_tests)
    foreach(_t vecmat_tests vecmat_benchmarks)
        if(TARGET ${_t})
            target_include_directories(${_t} PRIVATE
                    ${_sg}/include ${CMAKE_SOURCE_DIR}/test)
        endif()
    endforeach()
endfunction()

list(APPEND VECMAT_MODULE_HOOKS vecmat_module_alkontek_spectral_graph_sources)
list(APPEND VECMAT_MODULE_TEST_HOOKS vecmat_module_alkontek_spectral_graph_tests)
list(APPEND VECMAT_MODULE_TEST_SOURCES ${_sg}/test/test_spectral_graph.c)
