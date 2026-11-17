// SPDX-FileCopyrightText: 2025-2026 Igal Alkon
// SPDX-FileCopyrightText: 2026 ALKONTEK <git@alkontek.com>
// SPDX-License-Identifier: BSD-3-Clause

#include <stdlib.h>
#include <string.h>
#include <vecmat.h>
#include <vecmat/feature/md.h>
#include "unitest.h"

#if !defined(_MSC_VER)
#pragma STDC FP_CONTRACT OFF
#endif

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC optimize("fp-contract=off")
#endif

enum { MD_N = 19, BLOCK = 6, LIST_SCRATCH = 4096, LIST_PAIRS = 512 };

static int cmp_int(const void *a, const void *b)
{
    const int ia = *(const int *)a;
    const int ib = *(const int *)b;
    return (ia > ib) - (ia < ib);
}

static void sort_rows(int *neighbors, const int *offsets, const int ni)
{
    for (int row = 0; row < ni; ++row) {
        const int begin = offsets[row];
        const int count = offsets[row + 1] - begin;
        if (count > 1) {
            qsort(neighbors + begin, (size_t)count, sizeof(int), cmp_int);
        }
    }
}

static void require_increasing(const int *offsets, const int *neighbors, const int ni)
{
    for (int row = 0; row < ni; ++row) {
        for (int p = offsets[row] + 1; p < offsets[row + 1]; ++p) {
            REQUIRE(neighbors[p] > neighbors[p - 1]);
        }
    }
}

static int same_bits(const vm_float_t a, const vm_float_t b)
{
    return memcmp(&a, &b, sizeof a) == 0;
}

static void require_bits(const vm_float_t got, const vm_float_t exp, const char *what)
{
    if (!same_bits(got, exp)) {
        FAIL("%s got %.9g exp %.9g", what, (double)got, (double)exp);
    }
}

static void clear_state(vm_float_t *fx, vm_float_t *fy, vm_float_t *fz,
                        vm_float_t *energy, vm_float_t *virial, const int n,
                        const vm_float_t seed)
{
    for (int i = 0; i < n; ++i) {
        fx[i] = seed;
        fy[i] = seed;
        fz[i] = seed;
    }
    *energy = seed;
    for (int k = 0; k < 6; ++k) {
        virial[k] = seed;
    }
}

static void require_state(const vm_float_t *afx, const vm_float_t *afy, const vm_float_t *afz,
                          const vm_float_t aenergy, const vm_float_t *avir,
                          const vm_float_t *bfx, const vm_float_t *bfy, const vm_float_t *bfz,
                          const vm_float_t benergy, const vm_float_t *bvir, const int n)
{
    for (int i = 0; i < n; ++i) {
        require_bits(afx[i], bfx[i], "fx");
        require_bits(afy[i], bfy[i], "fy");
        require_bits(afz[i], bfz[i], "fz");
    }
    require_bits(aenergy, benergy, "energy");
    for (int k = 0; k < 6; ++k) {
        require_bits(avir[k], bvir[k], "virial");
    }
}

static void require_forces(const vm_float_t *afx, const vm_float_t *afy, const vm_float_t *afz,
                           const vm_float_t *bfx, const vm_float_t *bfy, const vm_float_t *bfz,
                           const int n)
{
    for (int i = 0; i < n; ++i) {
        require_bits(afx[i], bfx[i], "fx");
        require_bits(afy[i], bfy[i], "fy");
        require_bits(afz[i], bfz[i], "fz");
    }
}

static int force_moved(const vm_float_t *fx, const vm_float_t *fy, const vm_float_t *fz,
                       const int n, const vm_float_t seed)
{
    for (int i = 0; i < n; ++i) {
        if (!same_bits(fx[i], seed) || !same_bits(fy[i], seed) || !same_bits(fz[i], seed)) {
            return 1;
        }
    }
    return 0;
}

static int build_list(int *offsets, int *neighbors, const int capacity,
                      const vm_float_t *x, const vm_float_t *y, const vm_float_t *z,
                      const int n, const int i0, const int ni, const int j0, const int nj,
                      const vector3 box, const vm_float_t rcut)
{
    int scratch[LIST_SCRATCH];
    const int need = vm_md_neighbors_scratch(n, i0, ni, j0, nj, box, rcut);
    if (need < 0 || need > LIST_SCRATCH) {
        return 0;
    }
    return vm_md_neighbors(offsets, neighbors, capacity,
                           need ? scratch : NULL, need,
                           x, y, z, n, i0, ni, j0, nj, box, rcut);
}

/* One cell on every axis, so vm_md_neighbors emits increasing j. */
static void fill_blocks(vm_float_t *x, vm_float_t *y, vm_float_t *z)
{
    x[0] = VM_F(0.20); y[0] = VM_F(0.20); z[0] = VM_F(0.20);
    x[1] = VM_F(0.70); y[1] = VM_F(0.40); z[1] = VM_F(0.30);
    x[2] = VM_F(0.30); y[2] = VM_F(0.90); z[2] = VM_F(0.50);
    x[3] = VM_F(3.70); y[3] = VM_F(0.25); z[3] = VM_F(0.20);
    x[4] = VM_F(3.40); y[4] = VM_F(0.80); z[4] = VM_F(0.40);
    /* Outside the cutoff of the low block, including the periodic image. */
    x[5] = VM_F(1.95); y[5] = VM_F(1.95); z[5] = VM_F(1.95);
}

static void fill_block_charges(vm_float_t *q)
{
    q[0] = VM_F(1.0);
    q[1] = VM_F(-0.5);
    q[2] = VM_F(0.25);
    q[3] = VM_F(0.8);
    q[4] = VM_F(-1.2);
    q[5] = VM_F(0.0);
}

static void fill_fixture(vm_float_t *x, vm_float_t *y, vm_float_t *z, const vector3 box)
{
    const vm_float_t rcut = VM_F(2.5);
    for (int i = 0; i < MD_N; ++i) {
        x[i] = VM_F(0.17) * (vm_float_t)((i * 3) % 11) + VM_F(0.05) * (vm_float_t)i;
        y[i] = VM_F(0.13) * (vm_float_t)((i * 5) % 9) - VM_F(0.2);
        z[i] = VM_F(0.11) * (vm_float_t)((i * 7) % 13) + VM_F(0.03) * (vm_float_t)(i % 4);
    }
    x[0] = VM_F(0.0);
    y[0] = VM_F(0.0);
    z[0] = VM_F(0.0);
    x[1] = rcut;
    y[1] = VM_F(0.0);
    z[1] = VM_F(0.0);
    x[2] = box.x * VM_F(0.5);
    y[2] = VM_F(0.0);
    z[2] = VM_F(0.0);
    x[3] = box.x - VM_F(0.35);
    y[3] = VM_F(0.15);
    z[3] = VM_F(-0.2);
    x[4] = VM_F(0.4);
    y[4] = VM_F(0.2);
    z[4] = VM_F(-0.1);
}

static void fill_fixture_charges(vm_float_t *q)
{
    for (int i = 0; i < MD_N; ++i) {
        const int k = (i % 5) - 2;
        q[i] = VM_F(0.5) * (vm_float_t)k;
    }
    q[0] = VM_F(1.0);
    q[1] = VM_F(-0.5);
    q[2] = VM_F(0.8);
    q[3] = VM_F(-1.5);
    q[4] = VM_F(0.75);
    q[5] = VM_F(0.0);
}

static void match_lj_directed(const vm_float_t *x, const vm_float_t *y, const vm_float_t *z,
                              const int n, const int i0, const int ni, const int j0, const int nj,
                              const vector3 box, const vm_float_t epsilon, const vm_float_t sigma,
                              const vm_float_t rcut, int *offsets, int *neighbors,
                              const int sort)
{
    vm_float_t lfx[MD_N], lfy[MD_N], lfz[MD_N], lenergy, lvir[6];
    vm_float_t tfx[MD_N], tfy[MD_N], tfz[MD_N], tenergy, tvir[6];
    vm_float_t nfx[MD_N], nfy[MD_N], nfz[MD_N];
    if (sort) {
        sort_rows(neighbors, offsets, ni);
    }
    clear_state(lfx, lfy, lfz, &lenergy, lvir, n, VM_F(0.0));
    clear_state(tfx, tfy, tfz, &tenergy, tvir, n, VM_F(0.0));
    REQUIRE(vm_md_lj_list(lfx, lfy, lfz, &lenergy, lvir, x, y, z,
                          n, i0, ni, offsets, neighbors, box, epsilon, sigma, rcut));
    REQUIRE(vm_md_lj_tile(tfx, tfy, tfz, &tenergy, tvir, x, y, z,
                          n, i0, ni, j0, nj, box, epsilon, sigma, rcut, 0));
    require_state(lfx, lfy, lfz, lenergy, lvir, tfx, tfy, tfz, tenergy, tvir, n);
    REQUIRE(force_moved(lfx, lfy, lfz, n, VM_F(0.0)));

    clear_state(nfx, nfy, nfz, &lenergy, lvir, n, VM_F(0.0));
    REQUIRE(vm_md_lj_list(nfx, nfy, nfz, NULL, NULL, x, y, z,
                          n, i0, ni, offsets, neighbors, box, epsilon, sigma, rcut));
    require_forces(nfx, nfy, nfz, lfx, lfy, lfz, n);
}

static void match_coulomb_directed(const vm_float_t *x, const vm_float_t *y, const vm_float_t *z,
                                   const vm_float_t *q, const int n, const int i0, const int ni,
                                   const int j0, const int nj, const vector3 box,
                                   const vm_float_t prefactor, const vm_float_t rcut,
                                   int *offsets, int *neighbors, const int sort)
{
    vm_float_t lfx[MD_N], lfy[MD_N], lfz[MD_N], lenergy, lvir[6];
    vm_float_t tfx[MD_N], tfy[MD_N], tfz[MD_N], tenergy, tvir[6];
    if (sort) {
        sort_rows(neighbors, offsets, ni);
    }
    clear_state(lfx, lfy, lfz, &lenergy, lvir, n, VM_F(0.0));
    clear_state(tfx, tfy, tfz, &tenergy, tvir, n, VM_F(0.0));
    REQUIRE(vm_md_coulomb_list(lfx, lfy, lfz, &lenergy, lvir, x, y, z, q,
                               n, i0, ni, offsets, neighbors, box, prefactor, rcut));
    REQUIRE(vm_md_coulomb_tile(tfx, tfy, tfz, &tenergy, tvir, x, y, z, q,
                               n, i0, ni, j0, nj, box, prefactor, rcut, 0));
    require_state(lfx, lfy, lfz, lenergy, lvir, tfx, tfy, tfz, tenergy, tvir, n);
    REQUIRE(force_moved(lfx, lfy, lfz, n, VM_F(0.0)));
}

static void match_lj_newton(const vm_float_t *x, const vm_float_t *y, const vm_float_t *z,
                            const int n, const int i0, const int ni, const vector3 box,
                            const vm_float_t epsilon, const vm_float_t sigma,
                            const vm_float_t rcut, int *offsets, int *neighbors,
                            const int sort)
{
    vm_float_t lfx[MD_N], lfy[MD_N], lfz[MD_N], lenergy, lvir[6];
    vm_float_t nfx[MD_N], nfy[MD_N], nfz[MD_N], nenergy, nvir[6];
    vm_float_t dfx[MD_N], dfy[MD_N], dfz[MD_N], denergy, dvir[6];
    if (sort) {
        sort_rows(neighbors, offsets, ni);
    }
    clear_state(lfx, lfy, lfz, &lenergy, lvir, n, VM_F(0.0));
    clear_state(nfx, nfy, nfz, &nenergy, nvir, n, VM_F(0.0));
    clear_state(dfx, dfy, dfz, &denergy, dvir, n, VM_F(0.0));
    REQUIRE(vm_md_lj_list(lfx, lfy, lfz, &lenergy, lvir, x, y, z,
                          n, i0, ni, offsets, neighbors, box, epsilon, sigma, rcut));
    REQUIRE(vm_md_lj_tile(nfx, nfy, nfz, &nenergy, nvir, x, y, z,
                          n, i0, ni, i0, ni, box, epsilon, sigma, rcut, 1));
    REQUIRE(vm_md_lj_tile(dfx, dfy, dfz, &denergy, dvir, x, y, z,
                          n, i0, ni, i0, ni, box, epsilon, sigma, rcut, 0));
    require_forces(lfx, lfy, lfz, nfx, nfy, nfz, n);
    require_state(lfx, lfy, lfz, lenergy, lvir, dfx, dfy, dfz, denergy, dvir, n);
    REQUIRE(force_moved(lfx, lfy, lfz, n, VM_F(0.0)));
}

static void match_coulomb_newton(const vm_float_t *x, const vm_float_t *y, const vm_float_t *z,
                                 const vm_float_t *q, const int n, const int i0, const int ni,
                                 const vector3 box, const vm_float_t prefactor,
                                 const vm_float_t rcut, int *offsets, int *neighbors,
                                 const int sort)
{
    vm_float_t lfx[MD_N], lfy[MD_N], lfz[MD_N], lenergy, lvir[6];
    vm_float_t nfx[MD_N], nfy[MD_N], nfz[MD_N], nenergy, nvir[6];
    vm_float_t dfx[MD_N], dfy[MD_N], dfz[MD_N], denergy, dvir[6];
    if (sort) {
        sort_rows(neighbors, offsets, ni);
    }
    clear_state(lfx, lfy, lfz, &lenergy, lvir, n, VM_F(0.0));
    clear_state(nfx, nfy, nfz, &nenergy, nvir, n, VM_F(0.0));
    clear_state(dfx, dfy, dfz, &denergy, dvir, n, VM_F(0.0));
    REQUIRE(vm_md_coulomb_list(lfx, lfy, lfz, &lenergy, lvir, x, y, z, q,
                               n, i0, ni, offsets, neighbors, box, prefactor, rcut));
    REQUIRE(vm_md_coulomb_tile(nfx, nfy, nfz, &nenergy, nvir, x, y, z, q,
                               n, i0, ni, i0, ni, box, prefactor, rcut, 1));
    REQUIRE(vm_md_coulomb_tile(dfx, dfy, dfz, &denergy, dvir, x, y, z, q,
                               n, i0, ni, i0, ni, box, prefactor, rcut, 0));
    require_forces(lfx, lfy, lfz, nfx, nfy, nfz, n);
    require_state(lfx, lfy, lfz, lenergy, lvir, dfx, dfy, dfz, denergy, dvir, n);
    REQUIRE(force_moved(lfx, lfy, lfz, n, VM_F(0.0)));
}

TEST_CASE(md_list_disjoint_blocks_match_tile, "[md]") {
    const vector3 box = vec3(VM_F(3.9), VM_F(3.9), VM_F(3.9));
    const vm_float_t rcut = VM_F(2.0);
    const vm_float_t epsilon = VM_F(0.8);
    const vm_float_t sigma = VM_F(1.15);
    const vm_float_t prefactor = VM_F(2.3);
    vm_float_t x[BLOCK], y[BLOCK], z[BLOCK], q[BLOCK];
    int offsets[BLOCK + 1];
    int neighbors[LIST_PAIRS];
    fill_blocks(x, y, z);
    fill_block_charges(q);

    /* Low block against the high block. One stored direction. */
    REQUIRE(build_list(offsets, neighbors, LIST_PAIRS, x, y, z,
                       BLOCK, 0, 3, 3, 3, box, rcut));
    REQUIRE(offsets[3] > 0);
    REQUIRE(offsets[3] < 9);
    require_increasing(offsets, neighbors, 3);
    match_lj_directed(x, y, z, BLOCK, 0, 3, 3, 3, box, epsilon, sigma, rcut,
                      offsets, neighbors, 0);
    match_coulomb_directed(x, y, z, q, BLOCK, 0, 3, 3, 3, box, prefactor, rcut,
                           offsets, neighbors, 0);

    /* The opposite block. Still one direction, and i > j. */
    REQUIRE(build_list(offsets, neighbors, LIST_PAIRS, x, y, z,
                       BLOCK, 3, 3, 0, 3, box, rcut));
    REQUIRE(offsets[3] > 0);
    require_increasing(offsets, neighbors, 3);
    match_lj_directed(x, y, z, BLOCK, 3, 3, 0, 3, box, epsilon, sigma, rcut,
                      offsets, neighbors, 0);
    match_coulomb_directed(x, y, z, q, BLOCK, 3, 3, 0, 3, box, prefactor, rcut,
                           offsets, neighbors, 0);

    /* Fixture blocks. Sort only so the walk order is the tile's j order. */
    {
        const vector3 fixture = vec3(VM_F(5.5), VM_F(4.25), VM_F(6.0));
        const vm_float_t fixture_rcut = VM_F(2.5);
        vm_float_t fx[MD_N], fy[MD_N], fz[MD_N], fq[MD_N];
        int foff[MD_N + 1];
        int fnb[LIST_PAIRS];
        fill_fixture(fx, fy, fz, fixture);
        fill_fixture_charges(fq);
        REQUIRE(build_list(foff, fnb, LIST_PAIRS, fx, fy, fz,
                           MD_N, 0, 4, 8, 7, fixture, fixture_rcut));
        REQUIRE(foff[4] > 0);
        match_lj_directed(fx, fy, fz, MD_N, 0, 4, 8, 7, fixture, epsilon, sigma,
                          fixture_rcut, foff, fnb, 1);
        REQUIRE(build_list(foff, fnb, LIST_PAIRS, fx, fy, fz,
                           MD_N, 0, 4, 8, 7, fixture, fixture_rcut));
        match_coulomb_directed(fx, fy, fz, fq, MD_N, 0, 4, 8, 7, fixture, prefactor,
                               fixture_rcut, foff, fnb, 1);
        REQUIRE(build_list(foff, fnb, LIST_PAIRS, fx, fy, fz,
                           MD_N, 8, 7, 0, 4, fixture, fixture_rcut));
        match_lj_directed(fx, fy, fz, MD_N, 8, 7, 0, 4, fixture, epsilon, sigma,
                          fixture_rcut, foff, fnb, 1);
    }
}

TEST_CASE(md_list_self_tile_matches_newton, "[md]") {
    const vector3 box = vec3(VM_F(3.9), VM_F(3.9), VM_F(3.9));
    const vm_float_t rcut = VM_F(2.0);
    const vm_float_t epsilon = VM_F(0.8);
    const vm_float_t sigma = VM_F(1.15);
    const vm_float_t prefactor = VM_F(2.3);
    vm_float_t x[BLOCK], y[BLOCK], z[BLOCK], q[BLOCK];
    int offsets[BLOCK + 1];
    int neighbors[LIST_PAIRS];
    fill_blocks(x, y, z);
    fill_block_charges(q);

    REQUIRE(build_list(offsets, neighbors, LIST_PAIRS, x, y, z,
                       BLOCK, 0, BLOCK, 0, BLOCK, box, rcut));
    REQUIRE(offsets[BLOCK] > BLOCK);
    require_increasing(offsets, neighbors, BLOCK);
    match_lj_newton(x, y, z, BLOCK, 0, BLOCK, box, epsilon, sigma, rcut,
                    offsets, neighbors, 0);
    match_coulomb_newton(x, y, z, q, BLOCK, 0, BLOCK, box, prefactor, rcut,
                         offsets, neighbors, 0);

    {
        const vector3 fixture = vec3(VM_F(5.5), VM_F(4.25), VM_F(6.0));
        const vm_float_t fixture_rcut = VM_F(2.5);
        vm_float_t fx[MD_N], fy[MD_N], fz[MD_N], fq[MD_N];
        int foff[MD_N + 1];
        int fnb[LIST_PAIRS];
        fill_fixture(fx, fy, fz, fixture);
        fill_fixture_charges(fq);
        REQUIRE(build_list(foff, fnb, LIST_PAIRS, fx, fy, fz,
                           MD_N, 0, MD_N, 0, MD_N, fixture, fixture_rcut));
        REQUIRE(foff[MD_N] > MD_N);
        match_lj_newton(fx, fy, fz, MD_N, 0, MD_N, fixture, epsilon, sigma,
                        fixture_rcut, foff, fnb, 1);
        REQUIRE(build_list(foff, fnb, LIST_PAIRS, fx, fy, fz,
                           MD_N, 0, MD_N, 0, MD_N, fixture, fixture_rcut));
        match_coulomb_newton(fx, fy, fz, fq, MD_N, 0, MD_N, fixture, prefactor,
                             fixture_rcut, foff, fnb, 1);
    }
}

TEST_CASE(md_list_cutoff_and_self_contribute_nothing, "[md]") {
    const vector3 box = vec3(VM_F(20.0), VM_F(20.0), VM_F(20.0));
    const vm_float_t rcut = VM_F(2.5);
    const vm_float_t epsilon = VM_F(1.0);
    const vm_float_t sigma = VM_F(1.0);
    const vm_float_t prefactor = VM_F(2.3);
    /* 1 is the exact cutoff, 2 is outside, 3 is inside, 0 is self. */
    vm_float_t x[4] = { VM_F(0.0), rcut, VM_F(4.0), VM_F(1.0) };
    vm_float_t y[4] = { VM_F(0.0), VM_F(0.0), VM_F(0.0), VM_F(0.0) };
    vm_float_t z[4] = { VM_F(0.0), VM_F(0.0), VM_F(0.0), VM_F(0.0) };
    vm_float_t q[4] = { VM_F(1.0), VM_F(-0.4), VM_F(0.5), VM_F(0.8) };
    int offsets[2] = { 0, 3 };
    int neighbors[4] = { 0, 1, 2, 3 };
    vm_float_t fx[4], fy[4], fz[4], energy, virial[6];
    vm_float_t tfx[4], tfy[4], tfz[4], tenergy, tvir[6];
    const vm_float_t seed = VM_F(4.0);

    clear_state(fx, fy, fz, &energy, virial, 4, seed);
    REQUIRE(vm_md_lj_list(fx, fy, fz, &energy, virial, x, y, z,
                          4, 0, 1, offsets, neighbors, box, epsilon, sigma, rcut));
    require_bits(fx[0], seed, "lj fx");
    require_bits(fy[2], seed, "lj fy");
    require_bits(fz[3], seed, "lj fz");
    require_bits(energy, seed, "lj energy");
    require_bits(virial[0], seed, "lj virial");
    require_bits(virial[5], seed, "lj virial yz");

    clear_state(fx, fy, fz, &energy, virial, 4, seed);
    REQUIRE(vm_md_coulomb_list(fx, fy, fz, &energy, virial, x, y, z, q,
                               4, 0, 1, offsets, neighbors, box, prefactor, rcut));
    require_bits(fx[0], seed, "coulomb fx");
    require_bits(energy, seed, "coulomb energy");
    require_bits(virial[3], seed, "coulomb virial");

    /* A real pair beside a self index and a cutoff index matches that pair alone. */
    offsets[1] = 4;
    neighbors[0] = 0;
    neighbors[1] = 1;
    neighbors[2] = 2;
    neighbors[3] = 3;
    clear_state(fx, fy, fz, &energy, virial, 4, VM_F(0.0));
    clear_state(tfx, tfy, tfz, &tenergy, tvir, 4, VM_F(0.0));
    REQUIRE(vm_md_lj_list(fx, fy, fz, &energy, virial, x, y, z,
                          4, 0, 1, offsets, neighbors, box, epsilon, sigma, rcut));
    REQUIRE(vm_md_lj_tile(tfx, tfy, tfz, &tenergy, tvir, x, y, z,
                          4, 0, 1, 3, 1, box, epsilon, sigma, rcut, 0));
    require_state(fx, fy, fz, energy, virial, tfx, tfy, tfz, tenergy, tvir, 4);
    REQUIRE(force_moved(fx, fy, fz, 4, VM_F(0.0)));

    clear_state(fx, fy, fz, &energy, virial, 4, VM_F(0.0));
    clear_state(tfx, tfy, tfz, &tenergy, tvir, 4, VM_F(0.0));
    REQUIRE(vm_md_coulomb_list(fx, fy, fz, &energy, virial, x, y, z, q,
                               4, 0, 1, offsets, neighbors, box, prefactor, rcut));
    REQUIRE(vm_md_coulomb_tile(tfx, tfy, tfz, &tenergy, tvir, x, y, z, q,
                               4, 0, 1, 3, 1, box, prefactor, rcut, 0));
    require_state(fx, fy, fz, energy, virial, tfx, tfy, tfz, tenergy, tvir, 4);
}

TEST_CASE(md_list_empty_leaves_buffers, "[md]") {
    const vector3 box = vec3(VM_F(3.9), VM_F(3.9), VM_F(3.9));
    const vm_float_t seed = VM_F(7.0);
    vm_float_t q[4] = { VM_F(1.0), VM_F(1.0), VM_F(1.0), VM_F(1.0) };
    vm_float_t fx[4], fy[4], fz[4], energy, virial[6];
    int offsets[4] = { 0, 0, 0, 0 };

    clear_state(fx, fy, fz, &energy, virial, 4, seed);
    REQUIRE(vm_md_lj_list(fx, fy, fz, &energy, virial, NULL, NULL, NULL,
                          4, 0, 0, NULL, NULL, box, VM_F(1.0), VM_F(1.0), VM_F(2.0)));
    REQUIRE(vm_md_lj_list(fx, fy, fz, &energy, virial, NULL, NULL, NULL,
                          4, 1, 3, offsets, NULL, box, VM_F(1.0), VM_F(1.0), VM_F(2.0)));
    REQUIRE(vm_md_coulomb_list(fx, fy, fz, &energy, virial, NULL, NULL, NULL, NULL,
                               4, 0, 0, NULL, NULL, box, VM_F(1.0), VM_F(2.0)));
    REQUIRE(vm_md_coulomb_list(fx, fy, fz, &energy, virial, NULL, NULL, NULL, q,
                               4, 1, 3, offsets, NULL, box, VM_F(1.0), VM_F(2.0)));
    for (int i = 0; i < 4; ++i) {
        require_bits(fx[i], seed, "fx");
        require_bits(fy[i], seed, "fy");
        require_bits(fz[i], seed, "fz");
    }
    require_bits(energy, seed, "energy");
    for (int k = 0; k < 6; ++k) {
        require_bits(virial[k], seed, "virial");
    }
}

TEST_CASE(md_list_rejects_bad_arguments_without_writes, "[md]") {
    const vector3 box = vec3(VM_F(20.0), VM_F(20.0), VM_F(20.0));
    const vector3 bad = vec3(VM_F(0.0), VM_F(1.0), VM_F(1.0));
    const vm_float_t seed = VM_F(3.0);
    vm_float_t x[2] = { VM_F(0.0), VM_F(1.0) };
    vm_float_t y[2] = { VM_F(0.0), VM_F(0.0) };
    vm_float_t z[2] = { VM_F(0.0), VM_F(0.0) };
    vm_float_t q[2] = { VM_F(1.0), VM_F(-1.0) };
    vm_float_t fx[2], fy[2], fz[2], energy, virial[6];
    int offsets[3] = { 0, 2, 1 };
    int neighbors[2] = { 1, -1 };

    clear_state(fx, fy, fz, &energy, virial, 2, seed);
    /* The first neighbor is a real pair. A bad later index must not keep it. */
    offsets[0] = 0;
    offsets[1] = 2;
    REQUIRE(!vm_md_lj_list(fx, fy, fz, &energy, virial, x, y, z,
                           2, 0, 1, offsets, neighbors, box, VM_F(1.0), VM_F(1.0), VM_F(2.0)));
    neighbors[1] = 2;
    REQUIRE(!vm_md_lj_list(fx, fy, fz, &energy, virial, x, y, z,
                           2, 0, 1, offsets, neighbors, box, VM_F(1.0), VM_F(1.0), VM_F(2.0)));
    neighbors[1] = 1;
    offsets[1] = 2;
    offsets[2] = 1;
    REQUIRE(!vm_md_lj_list(fx, fy, fz, &energy, virial, x, y, z,
                           2, 0, 2, offsets, neighbors, box, VM_F(1.0), VM_F(1.0), VM_F(2.0)));
    offsets[1] = 1;
    REQUIRE(!vm_md_lj_list(fx, fy, fz, &energy, virial, x, y, z,
                           2, 0, 1, NULL, neighbors, box, VM_F(1.0), VM_F(1.0), VM_F(2.0)));
    REQUIRE(!vm_md_lj_list(fx, fy, fz, &energy, virial, x, y, z,
                           2, 0, 1, offsets, NULL, box, VM_F(1.0), VM_F(1.0), VM_F(2.0)));
    REQUIRE(!vm_md_lj_list(fx, fy, fz, &energy, virial, x, y, z,
                           2, 0, 1, offsets, neighbors, bad, VM_F(1.0), VM_F(1.0), VM_F(2.0)));
    REQUIRE(!vm_md_lj_list(fx, fy, fz, &energy, virial, x, y, z,
                           2, 0, 1, offsets, neighbors, box, VM_F(-1.0), VM_F(1.0), VM_F(2.0)));
    REQUIRE(!vm_md_lj_list(NULL, fy, fz, &energy, virial, x, y, z,
                           2, 0, 1, offsets, neighbors, box, VM_F(1.0), VM_F(1.0), VM_F(2.0)));
    REQUIRE(!vm_md_lj_list(fx, fy, fz, &energy, virial, NULL, y, z,
                           2, 0, 1, offsets, neighbors, box, VM_F(1.0), VM_F(1.0), VM_F(2.0)));
    REQUIRE(!vm_md_coulomb_list(fx, fy, fz, &energy, virial, x, y, z, NULL,
                                2, 0, 1, offsets, neighbors, box, VM_F(1.0), VM_F(2.0)));
    REQUIRE(!vm_md_coulomb_list(fx, fy, fz, &energy, virial, x, y, z, q,
                                2, 0, 1, offsets, neighbors, box, VM_F(0.0), VM_F(2.0)));
    REQUIRE(!vm_md_coulomb_list(fx, fy, fz, &energy, virial, x, y, z, q,
                                2, -1, 1, offsets, neighbors, box, VM_F(1.0), VM_F(2.0)));
    require_bits(fx[0], seed, "fx");
    require_bits(fx[1], seed, "fx1");
    require_bits(fy[0], seed, "fy");
    require_bits(fz[1], seed, "fz");
    require_bits(energy, seed, "energy");
    require_bits(virial[0], seed, "virial");
    require_bits(virial[5], seed, "virial yz");

    /* A well-formed pair is accepted, including a NULL energy and virial. */
    REQUIRE(vm_md_lj_list(fx, fy, fz, NULL, NULL, x, y, z,
                          2, 0, 1, offsets, neighbors, box, VM_F(1.0), VM_F(1.0), VM_F(2.0)));
    REQUIRE(force_moved(fx, fy, fz, 2, seed));
}

static int any_differs(const vm_float_t *a, const vm_float_t *b, const int n)
{
    for (int i = 0; i < n; ++i) {
        if (!same_bits(a[i], b[i])) {
            return 1;
        }
    }
    return 0;
}

static void require_pair_changed(const vm_float_t *afx, const vm_float_t *afy, const vm_float_t *afz,
                                 const vm_float_t aenergy, const vm_float_t *avir,
                                 const vm_float_t *bfx, const vm_float_t *bfy, const vm_float_t *bfz,
                                 const vm_float_t benergy, const vm_float_t *bvir, const int n)
{
    const int force = any_differs(afx, bfx, n) || any_differs(afy, bfy, n) || any_differs(afz, bfz, n);
    REQUIRE(force);
    REQUIRE(!same_bits(aenergy, benergy));
    REQUIRE(any_differs(avir, bvir, 6));
}

TEST_CASE(md_list_exclusions, "[md]") {
    const vector3 box = vec3(VM_F(20.0), VM_F(20.0), VM_F(20.0));
    const vm_float_t rcut = VM_F(2.5);
    const vm_float_t epsilon = VM_F(0.8);
    const vm_float_t sigma = VM_F(1.15);
    const vm_float_t prefactor = VM_F(2.3);
    const vm_float_t seed = VM_F(5.0);
    const int n = 6;
    /* Rows 0, 1, and 2. Both directions of three interior pairs. */
    vm_float_t x[6] = { VM_F(0.0), VM_F(0.6), VM_F(1.1), VM_F(0.0), VM_F(0.0), VM_F(0.0) };
    vm_float_t y[6] = { VM_F(0.0), VM_F(0.1), VM_F(-0.2), VM_F(0.0), VM_F(0.0), VM_F(0.0) };
    vm_float_t z[6] = { VM_F(0.0), VM_F(-0.15), VM_F(0.25), VM_F(0.0), VM_F(0.0), VM_F(0.0) };
    vm_float_t q[6] = { VM_F(1.0), VM_F(-0.7), VM_F(0.4), VM_F(1.0), VM_F(1.0), VM_F(1.0) };
    int offsets[4] = { 0, 2, 4, 6 };
    int neighbors[6] = { 1, 2, 0, 2, 0, 1 };
    /* Unordered pair {0,1} listed on both walked rows. Remaining neighbor order. */
    int pair_off[4] = { 0, 1, 2, 2 };
    int pair_nb[2] = { 1, 0 };
    int reduced_off[4] = { 0, 1, 2, 4 };
    int reduced_nb[4] = { 2, 2, 0, 1 };
    /* Same pair on row 0 only. Row 1 still stores neighbor 0. */
    int one_off[4] = { 0, 1, 1, 1 };
    int one_nb[1] = { 1 };
    int one_reduced_off[4] = { 0, 1, 3, 5 };
    int one_reduced_nb[5] = { 2, 0, 2, 0, 1 };
    /* Every stored neighbor, reversed inside the row. */
    int all_off[4] = { 0, 2, 4, 6 };
    int all_nb[6] = { 2, 1, 2, 0, 1, 0 };
    int empty_off[4] = { 0, 0, 0, 0 };
    int poison = -1;
    vm_float_t lfx[6], lfy[6], lfz[6], lenergy, lvir[6];
    vm_float_t cfx[6], cfy[6], cfz[6], cenergy, cvir[6];
    vm_float_t fx[6], fy[6], fz[6], energy, virial[6];
    vm_float_t rfx[6], rfy[6], rfz[6], renergy, rvir[6];
    vm_float_t sfx[6], sfy[6], sfz[6], senergy, svir[6];

    clear_state(lfx, lfy, lfz, &lenergy, lvir, n, VM_F(0.0));
    clear_state(cfx, cfy, cfz, &cenergy, cvir, n, VM_F(0.0));
    REQUIRE(vm_md_lj_list(lfx, lfy, lfz, &lenergy, lvir, x, y, z,
                          n, 0, 3, offsets, neighbors, box, epsilon, sigma, rcut));
    REQUIRE(vm_md_coulomb_list(cfx, cfy, cfz, &cenergy, cvir, x, y, z, q,
                               n, 0, 3, offsets, neighbors, box, prefactor, rcut));
    REQUIRE(force_moved(lfx, lfy, lfz, n, VM_F(0.0)));
    REQUIRE(force_moved(cfx, cfy, cfz, n, VM_F(0.0)));

    /* NULL and an empty CSR match the list kernels. The poison is not read. */
    clear_state(fx, fy, fz, &energy, virial, n, VM_F(0.0));
    REQUIRE(vm_md_lj_list_excl(fx, fy, fz, &energy, virial, x, y, z,
                               n, 0, 3, offsets, neighbors, box, epsilon, sigma, rcut,
                               NULL, &poison));
    require_state(fx, fy, fz, energy, virial, lfx, lfy, lfz, lenergy, lvir, n);
    clear_state(fx, fy, fz, &energy, virial, n, VM_F(0.0));
    REQUIRE(vm_md_lj_list_excl(fx, fy, fz, &energy, virial, x, y, z,
                               n, 0, 3, offsets, neighbors, box, epsilon, sigma, rcut,
                               empty_off, &poison));
    require_state(fx, fy, fz, energy, virial, lfx, lfy, lfz, lenergy, lvir, n);
    clear_state(fx, fy, fz, &energy, virial, n, VM_F(0.0));
    REQUIRE(vm_md_coulomb_list_excl(fx, fy, fz, &energy, virial, x, y, z, q,
                                    n, 0, 3, offsets, neighbors, box, prefactor, rcut,
                                    NULL, &poison));
    require_state(fx, fy, fz, energy, virial, cfx, cfy, cfz, cenergy, cvir, n);
    clear_state(fx, fy, fz, &energy, virial, n, VM_F(0.0));
    REQUIRE(vm_md_coulomb_list_excl(fx, fy, fz, &energy, virial, x, y, z, q,
                                    n, 0, 3, offsets, neighbors, box, prefactor, rcut,
                                    empty_off, &poison));
    require_state(fx, fy, fz, energy, virial, cfx, cfy, cfz, cenergy, cvir, n);

    /* Excluding {0,1} changes forces, energy, and virial, and matches the list with those entries removed. */
    clear_state(fx, fy, fz, &energy, virial, n, VM_F(0.0));
    clear_state(rfx, rfy, rfz, &renergy, rvir, n, VM_F(0.0));
    REQUIRE(vm_md_lj_list_excl(fx, fy, fz, &energy, virial, x, y, z,
                               n, 0, 3, offsets, neighbors, box, epsilon, sigma, rcut,
                               pair_off, pair_nb));
    REQUIRE(vm_md_lj_list(rfx, rfy, rfz, &renergy, rvir, x, y, z,
                          n, 0, 3, reduced_off, reduced_nb, box, epsilon, sigma, rcut));
    require_state(fx, fy, fz, energy, virial, rfx, rfy, rfz, renergy, rvir, n);
    require_pair_changed(fx, fy, fz, energy, virial, lfx, lfy, lfz, lenergy, lvir, n);
    REQUIRE(force_moved(rfx, rfy, rfz, n, VM_F(0.0)));

    clear_state(fx, fy, fz, &energy, virial, n, VM_F(0.0));
    clear_state(rfx, rfy, rfz, &renergy, rvir, n, VM_F(0.0));
    REQUIRE(vm_md_coulomb_list_excl(fx, fy, fz, &energy, virial, x, y, z, q,
                                    n, 0, 3, offsets, neighbors, box, prefactor, rcut,
                                    pair_off, pair_nb));
    REQUIRE(vm_md_coulomb_list(rfx, rfy, rfz, &renergy, rvir, x, y, z, q,
                               n, 0, 3, reduced_off, reduced_nb, box, prefactor, rcut));
    require_state(fx, fy, fz, energy, virial, rfx, rfy, rfz, renergy, rvir, n);
    require_pair_changed(fx, fy, fz, energy, virial, cfx, cfy, cfz, cenergy, cvir, n);
    REQUIRE(force_moved(rfx, rfy, rfz, n, VM_F(0.0)));

    /* A partner listed on one row does not drop the opposite stored neighbor. */
    clear_state(fx, fy, fz, &energy, virial, n, VM_F(0.0));
    clear_state(rfx, rfy, rfz, &renergy, rvir, n, VM_F(0.0));
    REQUIRE(vm_md_lj_list_excl(fx, fy, fz, &energy, virial, x, y, z,
                               n, 0, 3, offsets, neighbors, box, epsilon, sigma, rcut,
                               one_off, one_nb));
    REQUIRE(vm_md_lj_list(rfx, rfy, rfz, &renergy, rvir, x, y, z,
                          n, 0, 3, one_reduced_off, one_reduced_nb, box, epsilon, sigma, rcut));
    require_state(fx, fy, fz, energy, virial, rfx, rfy, rfz, renergy, rvir, n);
    require_pair_changed(fx, fy, fz, energy, virial, lfx, lfy, lfz, lenergy, lvir, n);

    /* Excluding every neighbor leaves the seeded buffers alone. */
    clear_state(sfx, sfy, sfz, &senergy, svir, n, seed);
    clear_state(fx, fy, fz, &energy, virial, n, seed);
    REQUIRE(vm_md_lj_list_excl(fx, fy, fz, &energy, virial, x, y, z,
                               n, 0, 3, offsets, neighbors, box, epsilon, sigma, rcut,
                               all_off, all_nb));
    require_state(fx, fy, fz, energy, virial, sfx, sfy, sfz, senergy, svir, n);
    clear_state(fx, fy, fz, &energy, virial, n, seed);
    REQUIRE(vm_md_coulomb_list_excl(fx, fy, fz, &energy, virial, x, y, z, q,
                                    n, 0, 3, offsets, neighbors, box, prefactor, rcut,
                                    all_off, all_nb));
    require_state(fx, fy, fz, energy, virial, sfx, sfy, sfz, senergy, svir, n);

    /* Self stays skipped. Excluding it leaves the other pairs matching the list kernel. */
    {
        int self_off[2] = { 0, 3 };
        int self_nb[3] = { 0, 1, 2 };
        int skip_off[2] = { 0, 1 };
        int skip_nb[1] = { 0 };
        clear_state(lfx, lfy, lfz, &lenergy, lvir, n, VM_F(0.0));
        clear_state(fx, fy, fz, &energy, virial, n, VM_F(0.0));
        REQUIRE(vm_md_lj_list(lfx, lfy, lfz, &lenergy, lvir, x, y, z,
                              n, 0, 1, self_off, self_nb, box, epsilon, sigma, rcut));
        REQUIRE(vm_md_lj_list_excl(fx, fy, fz, &energy, virial, x, y, z,
                                   n, 0, 1, self_off, self_nb, box, epsilon, sigma, rcut,
                                   skip_off, skip_nb));
        require_state(fx, fy, fz, energy, virial, lfx, lfy, lfz, lenergy, lvir, n);
        REQUIRE(force_moved(lfx, lfy, lfz, n, VM_F(0.0)));

        clear_state(cfx, cfy, cfz, &cenergy, cvir, n, VM_F(0.0));
        clear_state(fx, fy, fz, &energy, virial, n, VM_F(0.0));
        REQUIRE(vm_md_coulomb_list(cfx, cfy, cfz, &cenergy, cvir, x, y, z, q,
                                   n, 0, 1, self_off, self_nb, box, prefactor, rcut));
        REQUIRE(vm_md_coulomb_list_excl(fx, fy, fz, &energy, virial, x, y, z, q,
                                        n, 0, 1, self_off, self_nb, box, prefactor, rcut,
                                        skip_off, skip_nb));
        require_state(fx, fy, fz, energy, virial, cfx, cfy, cfz, cenergy, cvir, n);
    }

    /* A partner that is not a stored neighbor leaves the real pairs unchanged. */
    {
        int row_off[2] = { 0, 2 };
        int row_nb[2] = { 1, 2 };
        int miss_off[2] = { 0, 1 };
        int miss_nb[1] = { 4 };
        clear_state(lfx, lfy, lfz, &lenergy, lvir, n, VM_F(0.0));
        clear_state(fx, fy, fz, &energy, virial, n, VM_F(0.0));
        REQUIRE(vm_md_lj_list(lfx, lfy, lfz, &lenergy, lvir, x, y, z,
                              n, 0, 1, row_off, row_nb, box, epsilon, sigma, rcut));
        REQUIRE(vm_md_lj_list_excl(fx, fy, fz, &energy, virial, x, y, z,
                                   n, 0, 1, row_off, row_nb, box, epsilon, sigma, rcut,
                                   miss_off, miss_nb));
        require_state(fx, fy, fz, energy, virial, lfx, lfy, lfz, lenergy, lvir, n);

        clear_state(cfx, cfy, cfz, &cenergy, cvir, n, VM_F(0.0));
        clear_state(fx, fy, fz, &energy, virial, n, VM_F(0.0));
        REQUIRE(vm_md_coulomb_list(cfx, cfy, cfz, &cenergy, cvir, x, y, z, q,
                                   n, 0, 1, row_off, row_nb, box, prefactor, rcut));
        REQUIRE(vm_md_coulomb_list_excl(fx, fy, fz, &energy, virial, x, y, z, q,
                                        n, 0, 1, row_off, row_nb, box, prefactor, rcut,
                                        miss_off, miss_nb));
        require_state(fx, fy, fz, energy, virial, cfx, cfy, cfz, cenergy, cvir, n);
    }

    /* ni == 0 does not read the exclusion CSR. A bad CSR writes nothing and returns false. */
    {
        int ignored = 4;
        int bad_off[4] = { 0, 1, 1, 1 };
        int bad_nb = -1;
        int one_row[2] = { 0, 1 };
        int one_partner = 1;
        clear_state(sfx, sfy, sfz, &senergy, svir, n, seed);
        clear_state(fx, fy, fz, &energy, virial, n, seed);
        REQUIRE(vm_md_lj_list_excl(fx, fy, fz, &energy, virial, NULL, NULL, NULL,
                                   n, 0, 0, NULL, NULL, box, epsilon, sigma, rcut,
                                   &ignored, &poison));
        REQUIRE(vm_md_coulomb_list_excl(fx, fy, fz, &energy, virial, NULL, NULL, NULL, NULL,
                                        n, 0, 0, NULL, NULL, box, prefactor, rcut,
                                        &ignored, &poison));
        REQUIRE(!vm_md_lj_list_excl(fx, fy, fz, &energy, virial, x, y, z,
                                    n, 0, 1, one_row, &one_partner, box, epsilon, sigma, rcut,
                                    bad_off, &bad_nb));
        REQUIRE(!vm_md_coulomb_list_excl(fx, fy, fz, &energy, virial, x, y, z, q,
                                         n, 0, 1, one_row, &one_partner, box, prefactor, rcut,
                                         bad_off, &bad_nb));
        REQUIRE(!vm_md_lj_list_excl(fx, fy, fz, &energy, virial, x, y, z,
                                    n, 0, 3, empty_off, NULL, box, epsilon, sigma, rcut,
                                    bad_off, &bad_nb));
        require_state(fx, fy, fz, energy, virial, sfx, sfy, sfz, senergy, svir, n);
    }
}
