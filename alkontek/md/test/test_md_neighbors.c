// SPDX-FileCopyrightText: 2025-2026 Igal Alkon
// SPDX-FileCopyrightText: 2026 ALKONTEK <git@alkontek.com>
// SPDX-License-Identifier: BSD-3-Clause

#include <math.h>
#include <vecmat.h>
#include <vecmat/feature/md.h>
#include "unitest.h"

#if !defined(_MSC_VER)
#pragma STDC FP_CONTRACT OFF
#endif

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC optimize("fp-contract=off")
#endif

enum { NB_SCRATCH = 2048, NB_PAIRS = 256 };

static int scratch_ok(int scratch[NB_SCRATCH], const int n, const int i0, const int ni,
                      const int j0, const int nj, const vector3 box, const vm_float_t rcut,
                      int *need_out)
{
    const int need = vm_md_neighbors_scratch(n, i0, ni, j0, nj, box, rcut);
    if (need < 0 || need > NB_SCRATCH) {
        return 0;
    }
    if (need_out) {
        *need_out = need;
    }
    (void)scratch;
    return 1;
}

/**
 * @brief 1 when the minimum image is inside the cutoff.
 *
 * `vm_periodic_delta_ortho` is the image. `vm_soa3_length_squared` is `r²`
 * for that one vector.
 */
static int brute_inside(const vm_float_t *x, const vm_float_t *y, const vm_float_t *z,
                        const int i, const int j, const vector3 box, const vm_float_t rcut2)
{
    const vector3 a = vec3(x[i], y[i], z[i]);
    const vector3 b = vec3(x[j], y[j], z[j]);
    vector3 d;
    vm_float_t r2 = VM_F(0.0);
    vm_csoa3 img;

    if (!vm_periodic_delta_ortho(&d, &a, &b, box, 1)) {
        return -1;
    }
    img.x = &d.x;
    img.y = &d.y;
    img.z = &d.z;
    vm_soa3_length_squared(&r2, img, 1);
    return r2 < rcut2;
}

static int listed(const int *offsets, const int *neighbors, const int row, const int j)
{
    int p;
    for (p = offsets[row]; p < offsets[row + 1]; ++p) {
        if (neighbors[p] == j) {
            return 1;
        }
    }
    return 0;
}

static void require_brute(const int *offsets, const int *neighbors, const int capacity,
                          const vm_float_t *x, const vm_float_t *y, const vm_float_t *z,
                          const int i0, const int ni, const int j0, const int nj,
                          const vector3 box, const vm_float_t rcut2)
{
    int brute = 0;
    int row;

    REQUIRE(offsets[0] == 0);
    for (row = 0; row < ni; ++row) {
        const int i = i0 + row;
        int p;
        int j;

        REQUIRE(offsets[row + 1] >= offsets[row]);
        REQUIRE(offsets[row + 1] <= capacity);
        for (p = offsets[row]; p < offsets[row + 1]; ++p) {
            int q;
            REQUIRE(neighbors[p] != i);
            REQUIRE(neighbors[p] >= j0);
            REQUIRE(neighbors[p] < j0 + nj);
            for (q = offsets[row]; q < p; ++q) {
                REQUIRE(neighbors[q] != neighbors[p]);
            }
        }
        for (j = j0; j < j0 + nj; ++j) {
            int hit = 0;
            if (i != j) {
                hit = brute_inside(x, y, z, i, j, box, rcut2);
                REQUIRE(hit >= 0);
            }
            if (hit) {
                brute += 1;
            }
            if (hit) {
                REQUIRE(listed(offsets, neighbors, row, j));
            } else {
                REQUIRE(!listed(offsets, neighbors, row, j));
            }
        }
    }
    REQUIRE(offsets[ni] == brute);
}

static int build(int *offsets, int *neighbors, const int capacity,
                 const vm_float_t *x, const vm_float_t *y, const vm_float_t *z,
                 const int n, const int i0, const int ni, const int j0, const int nj,
                 const vector3 box, const vm_float_t rcut)
{
    int scratch[NB_SCRATCH];
    int need = -1;
    if (!scratch_ok(scratch, n, i0, ni, j0, nj, box, rcut, &need)) {
        return 0;
    }
    return vm_md_neighbors(offsets, neighbors, capacity,
                           need ? scratch : NULL, need,
                           x, y, z, n, i0, ni, j0, nj, box, rcut);
}

TEST_CASE(md_neighbors_cluster_matches_brute_force, "[md]") {
    const vector3 box = vec3(VM_F(10.0), VM_F(10.0), VM_F(10.0));
    const vm_float_t rcut = VM_F(2.0);
    /* 0-1 inside, 1-2 is the exact cutoff, 0-3 and 1-3 wrap, 4 is alone.
     * Particle 5 is not finite and is not a neighbor of anyone. */
    vm_float_t x[6] = { VM_F(0.0), VM_F(1.0), VM_F(3.0), VM_F(9.5), VM_F(0.0), NAN };
    vm_float_t y[6] = { VM_F(0.0), VM_F(0.0), VM_F(0.0), VM_F(0.0), VM_F(5.0), NAN };
    vm_float_t z[6] = { VM_F(0.0), VM_F(0.0), VM_F(0.0), VM_F(0.0), VM_F(0.0), NAN };
    int offsets[8];
    int neighbors[NB_PAIRS];
    int row_offsets[2];
    int row_neighbors[8];
    int k;

    for (k = 0; k < 8; ++k) {
        offsets[k] = 7;
    }
    for (k = 0; k < NB_PAIRS; ++k) {
        neighbors[k] = 7;
    }
    REQUIRE(build(offsets, neighbors, NB_PAIRS, x, y, z, 6, 0, 6, 0, 6, box, rcut));
    REQUIRE(offsets[6] == 6);
    require_brute(offsets, neighbors, NB_PAIRS, x, y, z, 0, 6, 0, 6, box, rcut * rcut);
    REQUIRE(listed(offsets, neighbors, 0, 1));
    REQUIRE(listed(offsets, neighbors, 0, 3));
    REQUIRE(!listed(offsets, neighbors, 0, 2));
    REQUIRE(!listed(offsets, neighbors, 0, 0));
    REQUIRE(!listed(offsets, neighbors, 0, 5));
    REQUIRE(offsets[5] == offsets[4]);
    REQUIRE(offsets[6] == offsets[5]);
    REQUIRE(neighbors[6] == 7);

    REQUIRE(build(row_offsets, row_neighbors, 8, x, y, z, 6, 0, 1, 0, 6, box, rcut));
    REQUIRE(row_offsets[1] == 2);
    require_brute(row_offsets, row_neighbors, 8, x, y, z, 0, 1, 0, 6, box, rcut * rcut);

    row_offsets[0] = 9;
    row_offsets[1] = 9;
    REQUIRE(build(row_offsets, row_neighbors, 8, x, y, z, 6, 0, 1, 4, 1, box, rcut));
    REQUIRE(row_offsets[0] == 0);
    REQUIRE(row_offsets[1] == 0);
}

TEST_CASE(md_neighbors_cutoff_excludes_exact_and_outside, "[md]") {
    const vector3 box = vec3(VM_F(20.0), VM_F(20.0), VM_F(20.0));
    const vm_float_t rcut = VM_F(2.5);
    vm_float_t x[4] = { VM_F(0.0), rcut, VM_F(3.0), VM_F(2.0) };
    vm_float_t y[4] = { VM_F(0.0), VM_F(0.0), VM_F(0.0), VM_F(0.0) };
    vm_float_t z[4] = { VM_F(0.0), VM_F(0.0), VM_F(0.0), VM_F(0.0) };
    int offsets[5];
    int neighbors[16];
    int k;

    for (k = 0; k < 5; ++k) {
        offsets[k] = 4;
    }
    REQUIRE(build(offsets, neighbors, 16, x, y, z, 4, 0, 4, 0, 4, box, rcut));
    require_brute(offsets, neighbors, 16, x, y, z, 0, 4, 0, 4, box, rcut * rcut);
    REQUIRE(!listed(offsets, neighbors, 0, 1));
    REQUIRE(!listed(offsets, neighbors, 0, 2));
    REQUIRE(listed(offsets, neighbors, 0, 3));
    REQUIRE(!listed(offsets, neighbors, 0, 0));
}

TEST_CASE(md_neighbors_excludes_self, "[md]") {
    const vector3 box = vec3(VM_F(8.0), VM_F(8.0), VM_F(8.0));
    const vm_float_t rcut = VM_F(2.0);
    vm_float_t x[1] = { VM_F(1.0) };
    vm_float_t y[1] = { VM_F(2.0) };
    vm_float_t z[1] = { VM_F(3.0) };
    vm_float_t px[2] = { VM_F(1.0), VM_F(1.0) };
    vm_float_t py[2] = { VM_F(2.0), VM_F(2.0) };
    vm_float_t pz[2] = { VM_F(3.0), VM_F(3.0) };
    int offsets[3] = { 9, 9, 9 };
    int neighbors[4] = { 9, 9, 9, 9 };

    REQUIRE(build(offsets, NULL, 0, x, y, z, 1, 0, 1, 0, 1, box, rcut));
    REQUIRE(offsets[0] == 0);
    REQUIRE(offsets[1] == 0);

    REQUIRE(build(offsets, neighbors, 4, px, py, pz, 2, 0, 2, 0, 2, box, rcut));
    REQUIRE(offsets[2] == 2);
    REQUIRE(neighbors[0] == 1);
    REQUIRE(neighbors[1] == 0);
    require_brute(offsets, neighbors, 4, px, py, pz, 0, 2, 0, 2, box, rcut * rcut);
}

TEST_CASE(md_neighbors_empty_writes_nothing, "[md]") {
    const vector3 box = vec3(VM_F(5.0), VM_F(4.0), VM_F(6.0));
    const vm_float_t rcut = VM_F(1.5);
    int offsets[4] = { 7, 7, 7, 7 };
    int neighbors[4] = { 7, 7, 7, 7 };
    const vector3 bad = vec3(VM_F(0.0), VM_F(1.0), VM_F(1.0));

    REQUIRE(vm_md_neighbors_scratch(0, 0, 0, 0, 0, box, rcut) == 0);
    REQUIRE(vm_md_neighbors(offsets, neighbors, 4, NULL, -1, NULL, NULL, NULL,
                            0, 0, 0, 0, 0, box, rcut));
    REQUIRE(vm_md_neighbors(offsets, neighbors, 4, NULL, 0, NULL, NULL, NULL,
                            4, 0, 0, 0, 4, box, rcut));
    REQUIRE(vm_md_neighbors(offsets, neighbors, 4, NULL, 0, NULL, NULL, NULL,
                            4, 0, 4, 0, 0, box, rcut));
    REQUIRE(vm_md_neighbors(offsets, neighbors, 4, NULL, 0, NULL, NULL, NULL,
                            4, 4, 0, 0, 4, box, rcut));
    REQUIRE(offsets[0] == 7);
    REQUIRE(offsets[3] == 7);
    REQUIRE(neighbors[0] == 7);
    REQUIRE(neighbors[3] == 7);

    REQUIRE(vm_md_neighbors_scratch(-1, 0, 0, 0, 0, box, rcut) == -1);
    REQUIRE(!vm_md_neighbors(offsets, neighbors, 4, NULL, 0, NULL, NULL, NULL,
                             -1, 0, 0, 0, 0, box, rcut));
    REQUIRE(!vm_md_neighbors(offsets, neighbors, 4, NULL, 0, NULL, NULL, NULL,
                             4, 0, 0, 0, 0, bad, rcut));
    REQUIRE(!vm_md_neighbors(offsets, neighbors, 4, NULL, 0, NULL, NULL, NULL,
                             4, 5, 0, 0, 0, box, rcut));
    REQUIRE(offsets[0] == 7);
    REQUIRE(neighbors[2] == 7);
}

TEST_CASE(md_neighbors_periodic_boundary, "[md]") {
    const vector3 box = vec3(VM_F(5.0), VM_F(4.0), VM_F(6.0));
    const vm_float_t rcut = VM_F(1.25);
    /* 1 sits on the far x face. 3 is just outside the near face and wraps
     * onto that same image. 2 is the interior particle outside the cutoff. */
    vm_float_t x[4] = { VM_F(0.15), VM_F(4.8), VM_F(2.5), VM_F(-0.2) };
    vm_float_t y[4] = { VM_F(1.0), VM_F(1.0), VM_F(1.0), VM_F(1.0) };
    vm_float_t z[4] = { VM_F(2.0), VM_F(2.0), VM_F(2.0), VM_F(2.0) };
    int offsets[5];
    int neighbors[16];

    REQUIRE(build(offsets, neighbors, 16, x, y, z, 4, 0, 4, 0, 4, box, rcut));
    REQUIRE(offsets[4] == 6);
    require_brute(offsets, neighbors, 16, x, y, z, 0, 4, 0, 4, box, rcut * rcut);
    REQUIRE(listed(offsets, neighbors, 0, 1));
    REQUIRE(listed(offsets, neighbors, 0, 3));
    REQUIRE(!listed(offsets, neighbors, 0, 2));
    REQUIRE(listed(offsets, neighbors, 1, 3));
    REQUIRE(!listed(offsets, neighbors, 2, 0));
}

TEST_CASE(md_neighbors_one_cell_lists_each_pair_once, "[md]") {
    const vector3 box = vec3(VM_F(1.2), VM_F(0.8), VM_F(1.0));
    const vm_float_t rcut = VM_F(3.0);
    vm_float_t x[4] = { VM_F(0.0), VM_F(0.4), VM_F(1.1), VM_F(-0.3) };
    vm_float_t y[4] = { VM_F(0.0), VM_F(0.2), VM_F(0.7), VM_F(1.5) };
    vm_float_t z[4] = { VM_F(0.0), VM_F(0.1), VM_F(0.9), VM_F(0.4) };
    int offsets[5];
    int neighbors[32];

    REQUIRE(vm_md_neighbors_scratch(4, 0, 4, 0, 4, box, rcut) > 0);
    REQUIRE(build(offsets, neighbors, 32, x, y, z, 4, 0, 4, 0, 4, box, rcut));
    REQUIRE(offsets[4] == 12);
    require_brute(offsets, neighbors, 32, x, y, z, 0, 4, 0, 4, box, rcut * rcut);
}

TEST_CASE(md_neighbors_rejects_bad_arguments_without_writes, "[md]") {
    const vector3 box = vec3(VM_F(8.0), VM_F(8.0), VM_F(8.0));
    const vm_float_t rcut = VM_F(2.0);
    vm_float_t x[2] = { VM_F(0.0), VM_F(1.0) };
    vm_float_t y[2] = { VM_F(0.0), VM_F(0.2) };
    vm_float_t z[2] = { VM_F(0.0), VM_F(-0.2) };
    int offsets[3] = { 3, 3, 3 };
    int neighbors[4] = { 3, 3, 3, 3 };
    int scratch[NB_SCRATCH];
    int need;
    int short_scratch[1] = { 5 };

    need = vm_md_neighbors_scratch(2, 0, 2, 0, 2, box, rcut);
    REQUIRE(need > 1);
    REQUIRE(need <= NB_SCRATCH);

    REQUIRE(!vm_md_neighbors(offsets, neighbors, 4, scratch, need, x, y, z,
                             2, 0, 2, 0, 2, box, VM_F(0.0)));
    REQUIRE(!vm_md_neighbors(offsets, neighbors, 4, scratch, need, x, y, z,
                             2, 0, 2, 0, 2, box, NAN));
    REQUIRE(!vm_md_neighbors(offsets, neighbors, 4, scratch, need, x, y, z,
                             2, -1, 1, 0, 1, box, rcut));
    REQUIRE(!vm_md_neighbors(offsets, neighbors, 4, scratch, need, x, y, z,
                             2, 0, 3, 0, 2, box, rcut));
    REQUIRE(!vm_md_neighbors(NULL, neighbors, 4, scratch, need, x, y, z,
                             2, 0, 1, 1, 1, box, rcut));
    REQUIRE(!vm_md_neighbors(offsets, neighbors, 4, scratch, need, NULL, y, z,
                             2, 0, 1, 1, 1, box, rcut));
    REQUIRE(!vm_md_neighbors(offsets, neighbors, 4, NULL, need, x, y, z,
                             2, 0, 2, 0, 2, box, rcut));
    REQUIRE(!vm_md_neighbors(offsets, neighbors, 4, short_scratch, need - 1, x, y, z,
                             2, 0, 2, 0, 2, box, rcut));
    REQUIRE(!vm_md_neighbors(offsets, neighbors, 0, scratch, need, x, y, z,
                             2, 0, 2, 0, 2, box, rcut));
    REQUIRE(!vm_md_neighbors(offsets, NULL, 4, scratch, need, x, y, z,
                             2, 0, 2, 0, 2, box, rcut));
    REQUIRE(offsets[0] == 3);
    REQUIRE(offsets[2] == 3);
    REQUIRE(neighbors[0] == 3);
    REQUIRE(neighbors[3] == 3);
    REQUIRE(short_scratch[0] == 5);
}
