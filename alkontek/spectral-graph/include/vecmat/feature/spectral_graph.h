// SPDX-FileCopyrightText: 2025-2026 Igal Alkon
// SPDX-FileCopyrightText: 2026 ALKONTEK <git@alkontek.com>
// SPDX-License-Identifier: BSD-3-Clause

#ifndef VECMAT_FEATURE_SPECTRAL_GRAPH_H
#define VECMAT_FEATURE_SPECTRAL_GRAPH_H

/**
 * @file
 * @brief Bitsets and combinatorial unrank for spectral graph work.
 * @ingroup vecmat_extras
 *
 * Available only when the `alkontek/spectral-graph` feature module is loaded.
 */

#include <stdint.h>
#include <vecmat.h>

/**
 * @brief Fixed-capacity bitset over caller-owned or allocated words.
 */
typedef struct vm_bitset {
    uint64_t *words;  /**< Word storage. */
    int nbits;        /**< Number of bits. */
    int nwords;       /**< Number of words. */
} vm_bitset;

/**
 * @brief Allocate word storage for `nbits` bits.
 *
 * @param bs Bitset to fill.
 * @param nbits Number of bits.
 * @return false if `bs` is NULL or allocation fails.
 */
VEC_API bool vm_bitset_alloc(vm_bitset *bs, int nbits);

/**
 * @brief Free storage allocated by `vm_bitset_alloc`.
 *
 * @param bs Bitset to free.
 */
VEC_API void vm_bitset_free(vm_bitset *bs);

/**
 * @brief Set bit `i`.
 *
 * @param bs Bitset.
 * @param i Bit index.
 */
VEC_API void vm_bitset_set(vm_bitset *bs, int i);

/**
 * @brief Clear bit `i`.
 *
 * @param bs Bitset.
 * @param i Bit index.
 */
VEC_API void vm_bitset_clear(vm_bitset *bs, int i);

/**
 * @brief Test bit `i`.
 *
 * @param bs Bitset.
 * @param i Bit index.
 * @return true if the bit is set.
 */
VEC_API bool vm_bitset_test(const vm_bitset *bs, int i);

/**
 * @brief Bitwise and. `dst` must have the same capacity as `a`.
 *
 * @param dst Output.
 * @param a Left operand.
 * @param b Right operand.
 */
VEC_API void vm_bitset_and(vm_bitset *dst, const vm_bitset *a, const vm_bitset *b);

/**
 * @brief Bitwise or.
 *
 * @param dst Output.
 * @param a Left operand.
 * @param b Right operand.
 */
VEC_API void vm_bitset_or(vm_bitset *dst, const vm_bitset *a, const vm_bitset *b);

/**
 * @brief Bitwise xor.
 *
 * @param dst Output.
 * @param a Left operand.
 * @param b Right operand.
 */
VEC_API void vm_bitset_xor(vm_bitset *dst, const vm_bitset *a, const vm_bitset *b);

/**
 * @brief Population count.
 *
 * @param bs Bitset.
 * @return Number of set bits.
 */
VEC_API int vm_bitset_popcount(const vm_bitset *bs);

/**
 * @brief Build a dense Laplacian from row bitsets.
 *
 * `adj` has length `n`. Row `i` has `n` bits; bit `j` is an edge. The diagonal
 * of `L` is the degree; off-diagonal entries are -1 for an edge.
 *
 * @param L Output dense matrix; allocated to `n` by `n`.
 * @param adj Row bitsets; length `n`.
 * @param n Number of vertices.
 * @return false if a pointer is NULL or allocation fails.
 */
VEC_API bool vm_laplacian_from_bits(vm_mat *L, const vm_bitset *adj, int n);

/**
 * @brief Unrank a batch of combinations.
 *
 * `indices[b * k + j]` receives element `j` of combination `b`, in increasing
 * order. Combinations are in colex order.
 *
 * @param n Universe size.
 * @param k Combination size.
 * @param batch Number of ranks.
 * @param ranks Ranks; length `batch`.
 * @param indices Output; length `batch * k`.
 * @return false if a pointer is NULL or the arguments are invalid.
 */
VEC_API bool vm_comb_unrank_batch(int n, int k, int batch,
                                  const uint64_t *ranks, int *indices);

#endif //VECMAT_FEATURE_SPECTRAL_GRAPH_H
