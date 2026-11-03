// SPDX-FileCopyrightText: 2025-2026 Igal Alkon
// SPDX-FileCopyrightText: 2026 ALKONTEK <git@alkontek.com>
// SPDX-License-Identifier: BSD-3-Clause

#include <vecmat.h>
#include <vecmat/feature/spectral_graph.h>
#include "unitest.h"

TEST_CASE(bitset_set_and_popcount, "[spectral-graph]") {
    vm_bitset bs;
    REQUIRE(vm_bitset_alloc(&bs, 100));
    vm_bitset_set(&bs, 0);
    vm_bitset_set(&bs, 63);
    vm_bitset_set(&bs, 64);
    REQUIRE(vm_bitset_test(&bs, 0));
    REQUIRE(vm_bitset_test(&bs, 63));
    REQUIRE(!vm_bitset_test(&bs, 1));
    REQUIRE(vm_bitset_popcount(&bs) == 3);
    vm_bitset_clear(&bs, 63);
    REQUIRE(vm_bitset_popcount(&bs) == 2);
    vm_bitset_free(&bs);
}

TEST_CASE(bitset_and_or_xor, "[spectral-graph]") {
    vm_bitset a, b, d;
    REQUIRE(vm_bitset_alloc(&a, 64));
    REQUIRE(vm_bitset_alloc(&b, 64));
    REQUIRE(vm_bitset_alloc(&d, 64));
    vm_bitset_set(&a, 1);
    vm_bitset_set(&a, 2);
    vm_bitset_set(&b, 2);
    vm_bitset_set(&b, 3);
    vm_bitset_and(&d, &a, &b);
    REQUIRE(vm_bitset_popcount(&d) == 1);
    REQUIRE(vm_bitset_test(&d, 2));
    vm_bitset_or(&d, &a, &b);
    REQUIRE(vm_bitset_popcount(&d) == 3);
    vm_bitset_xor(&d, &a, &b);
    REQUIRE(vm_bitset_popcount(&d) == 2);
    vm_bitset_free(&a);
    vm_bitset_free(&b);
    vm_bitset_free(&d);
}

TEST_CASE(comb_unrank_batch, "[spectral-graph]") {
    const uint64_t ranks[] = { 0, 1, 2 };
    int idx[9];
    REQUIRE(vm_comb_unrank_batch(5, 3, 3, ranks, idx));
    /* C(5,3) colex: rank 0 = {0,1,2}, rank 1 = {0,1,3}, rank 2 = {0,2,3} */
    REQUIRE(idx[0] == 0 && idx[1] == 1 && idx[2] == 2);
    REQUIRE(idx[3] == 0 && idx[4] == 1 && idx[5] == 3);
    REQUIRE(idx[6] == 0 && idx[7] == 2 && idx[8] == 3);
}

TEST_CASE(laplacian_from_bits, "[spectral-graph]") {
    vm_bitset row[3];
    for (int i = 0; i < 3; ++i) {
        REQUIRE(vm_bitset_alloc(&row[i], 3));
    }
    /* path 0-1-2 */
    vm_bitset_set(&row[0], 1);
    vm_bitset_set(&row[1], 0);
    vm_bitset_set(&row[1], 2);
    vm_bitset_set(&row[2], 1);

    vm_mat L;
    REQUIRE(vm_laplacian_from_bits(&L, row, 3));
    REQUIRE(VECMAT_EQ(vm_mat_get(&L, 0, 0), VM_F(1.0), EPSILON));
    REQUIRE(VECMAT_EQ(vm_mat_get(&L, 0, 1), VM_F(-1.0), EPSILON));
    REQUIRE(VECMAT_EQ(vm_mat_get(&L, 1, 1), VM_F(2.0), EPSILON));
    REQUIRE(VECMAT_EQ(vm_mat_get(&L, 2, 2), VM_F(1.0), EPSILON));
    vm_mat_free(&L);
    for (int i = 0; i < 3; ++i) {
        vm_bitset_free(&row[i]);
    }
}
