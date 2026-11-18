// SPDX-FileCopyrightText: 2025-2026 Igal Alkon
// SPDX-FileCopyrightText: 2026 ALKONTEK <git@alkontek.com>
// SPDX-License-Identifier: BSD-3-Clause

#ifndef VECMAT_MD_COULOMB_INTERNAL_H
#define VECMAT_MD_COULOMB_INTERNAL_H

#include <stddef.h>

#include <vecmat/feature/md.h>

#if !defined(_MSC_VER)
#pragma STDC FP_CONTRACT OFF
#endif

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC optimize("fp-contract=off")
#endif

#define vm_md_coulomb_tile_avx2 VECMAT_SYMBOL(vm_md_coulomb_tile_avx2)
#define vm_md_coulomb_tile_neon VECMAT_SYMBOL(vm_md_coulomb_tile_neon)

/**
 * @brief Cutoff constants shared by the scalar tile and the SIMD paths.
 *
 * `prefactor` is the caller's Coulomb scale. The cutoff shift depends on
 * the charge product, so it is formed per pair.
 */
typedef struct md_coulomb_prep {
    vm_float_t rcut;
    vm_float_t rcut2;
    vm_float_t prefactor;
} md_coulomb_prep;

static inline int md_coulomb_positive(const vm_float_t v)
{
    return isfinite(v) && v > VM_F(0.0);
}

static inline int md_coulomb_range_ok(const int n, const int i0, const int count)
{
    if (n < 0 || i0 < 0 || count < 0 || i0 > n) {
        return 0;
    }
    return count <= n - i0;
}

static inline void md_coulomb_prepare(md_coulomb_prep *prep, const vm_float_t prefactor,
                                      const vm_float_t rcut)
{
    prep->rcut = rcut;
    prep->rcut2 = rcut * rcut;
    prep->prefactor = prefactor;
}

/**
 * @brief One Neumaier step, matching `vm_sum_compensated`.
 *
 * Core publishes the batch sum only. This is the same step that function
 * uses, kept local so the tile can accumulate in `(i, j)` order.
 */
static inline vm_float_t md_coulomb_neumaier(const vm_float_t sum, vm_float_t *comp,
                                             const vm_float_t x)
{
    volatile vm_float_t t = sum + x;
    if (VECMAT_FABS(sum) >= VECMAT_FABS(x)) {
        const volatile vm_float_t lost = sum - t;
        *comp += lost + x;
    } else {
        const volatile vm_float_t lost = x - t;
        *comp += lost + sum;
    }
    return t;
}

/**
 * @brief Shifted-force pair from an already wrapped minimum-image vector.
 *
 * @return 1 when the pair is inside the cutoff.
 */
static inline int md_coulomb_pair(const md_coulomb_prep *prep, const vm_float_t qi,
                                  const vm_float_t qj, const vm_float_t dx,
                                  const vm_float_t dy, const vm_float_t dz,
                                  vm_float_t *fix, vm_float_t *fiy, vm_float_t *fiz,
                                  vm_float_t *energy)
{
    const vm_float_t r2 = (dx * dx + dy * dy) + dz * dz;
    if (r2 >= prep->rcut2) {
        return 0;
    }

    const vm_float_t scale = prep->prefactor * (qi * qj);
    const vm_float_t r = VECMAT_SQRT(r2);
    const vm_float_t u = scale / r;
    const vm_float_t u_c = scale / prep->rcut;
    const vm_float_t duc = u_c / prep->rcut;
    const vm_float_t c = (scale / r2) / r;
    *energy = (u - u_c) + ((r - prep->rcut) * duc);
    const vm_float_t coeff = c - (duc / r);
    *fix = coeff * dx;
    *fiy = coeff * dy;
    *fiz = coeff * dz;
    return 1;
}

/**
 * @brief Add one accepted pair. `sum` is NULL when energy is not wanted.
 */
static inline void md_coulomb_apply(vm_float_t *fx, vm_float_t *fy, vm_float_t *fz,
                                    vm_float_t *sum, vm_float_t *comp,
                                    vm_float_t *virial, const int i, const int j,
                                    const int newton, const vm_float_t dx,
                                    const vm_float_t dy, const vm_float_t dz,
                                    const vm_float_t fix, const vm_float_t fiy,
                                    const vm_float_t fiz, const vm_float_t energy)
{
    fx[i] += fix;
    fy[i] += fiy;
    fz[i] += fiz;
    if (newton) {
        fx[j] -= fix;
        fy[j] -= fiy;
        fz[j] -= fiz;
    }
    if (sum) {
        *sum = md_coulomb_neumaier(*sum, comp, energy);
    }
    if (virial) {
        virial[0] += dx * fix;
        virial[1] += dy * fiy;
        virial[2] += dz * fiz;
        virial[3] += dx * fiy;
        virial[4] += dx * fiz;
        virial[5] += dy * fiz;
    }
}

static inline void md_coulomb_commit_energy(vm_float_t *energy, const vm_float_t sum,
                                            const vm_float_t comp)
{
    if (!energy) {
        return;
    }
    volatile vm_float_t total = sum + comp;
    *energy += total;
}

static inline int md_coulomb_half(const int newton, const int i0, const int ni,
                                  const int j0, const int nj)
{
    const int i1 = i0 + ni;
    const int j1 = j0 + nj;
    const int overlap = i0 < j1 && j0 < i1;
    return newton && overlap;
}

static inline void md_coulomb_scalar_tile(vm_float_t *fx, vm_float_t *fy, vm_float_t *fz,
                                          vm_float_t *energy, vm_float_t *virial,
                                          const vm_float_t *x, const vm_float_t *y,
                                          const vm_float_t *z, const vm_float_t *q,
                                          const int i0, const int ni, const int j0,
                                          const int nj, const vector3 box,
                                          const md_coulomb_prep *prep, const int newton)
{
    const int half = md_coulomb_half(newton, i0, ni, j0, nj);
    const int i1 = i0 + ni;
    const int j1 = j0 + nj;
    vm_float_t sum = VM_F(0.0);
    vm_float_t comp = VM_F(0.0);

    for (int i = i0; i < i1; ++i) {
        for (int j = j0; j < j1; ++j) {
            if (i == j || (half && i > j)) {
                continue;
            }
            const vm_float_t dx = VECMAT_REMAINDER(x[i] - x[j], box.x);
            const vm_float_t dy = VECMAT_REMAINDER(y[i] - y[j], box.y);
            const vm_float_t dz = VECMAT_REMAINDER(z[i] - z[j], box.z);
            vm_float_t fix, fiy, fiz, pair_energy;
            if (!md_coulomb_pair(prep, q[i], q[j], dx, dy, dz, &fix, &fiy, &fiz,
                                 &pair_energy)) {
                continue;
            }
            md_coulomb_apply(fx, fy, fz, energy ? &sum : NULL, &comp, virial,
                             i, j, newton, dx, dy, dz, fix, fiy, fiz, pair_energy);
        }
    }

    md_coulomb_commit_energy(energy, sum, comp);
}

#if defined(VECMAT_MD_AVX2)
void vm_md_coulomb_tile_avx2(vm_float_t *fx, vm_float_t *fy, vm_float_t *fz,
                             vm_float_t *energy, vm_float_t *virial,
                             const vm_float_t *x, const vm_float_t *y,
                             const vm_float_t *z, const vm_float_t *q,
                             int i0, int ni, int j0, int nj,
                             vector3 box, const md_coulomb_prep *prep, int newton);
#endif

#if defined(VECMAT_MD_NEON)
void vm_md_coulomb_tile_neon(vm_float_t *fx, vm_float_t *fy, vm_float_t *fz,
                             vm_float_t *energy, vm_float_t *virial,
                             const vm_float_t *x, const vm_float_t *y,
                             const vm_float_t *z, const vm_float_t *q,
                             int i0, int ni, int j0, int nj,
                             vector3 box, const md_coulomb_prep *prep, int newton);
#endif

#endif /* VECMAT_MD_COULOMB_INTERNAL_H */
