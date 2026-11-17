// SPDX-FileCopyrightText: 2025-2026 Igal Alkon
// SPDX-FileCopyrightText: 2026 ALKONTEK <git@alkontek.com>
// SPDX-License-Identifier: BSD-3-Clause

#include "coulomb_internal.h"
#include "list_excl.h"

bool vm_md_coulomb_tile(
    vm_float_t *fx, vm_float_t *fy, vm_float_t *fz,
    vm_float_t *energy, vm_float_t virial[6],
    const vm_float_t *x, const vm_float_t *y, const vm_float_t *z,
    const vm_float_t *q,
    const int n, const int i0, const int ni, const int j0, const int nj,
    const vector3 box, const vm_float_t prefactor, const vm_float_t rcut,
    const int newton)
{
    if (!md_coulomb_range_ok(n, i0, ni) || !md_coulomb_range_ok(n, j0, nj)) {
        return false;
    }

    if (!md_coulomb_positive(prefactor) || !md_coulomb_positive(rcut)) {
        return false;
    }

    if (!md_coulomb_positive(box.x) || !md_coulomb_positive(box.y) ||
        !md_coulomb_positive(box.z)) {
        return false;
    }

    if (ni == 0 || nj == 0) {
        return true;
    }

    if (!fx || !fy || !fz || !x || !y || !z || !q) {
        return false;
    }

    md_coulomb_prep prep;
    md_coulomb_prepare(&prep, prefactor, rcut);

#if defined(VECMAT_MD_AVX2)
    if (vm_cpu_has(VM_LEVEL_AVX2)) {
        vm_md_coulomb_tile_avx2(fx, fy, fz, energy, virial, x, y, z, q,
                                i0, ni, j0, nj, box, &prep, newton);
        return true;
    }
#endif
#if defined(VECMAT_MD_NEON)
    if (vm_cpu_has(VM_LEVEL_NEON)) {
        vm_md_coulomb_tile_neon(fx, fy, fz, energy, virial, x, y, z, q,
                                i0, ni, j0, nj, box, &prep, newton);
        return true;
    }
#endif

    md_coulomb_scalar_tile(fx, fy, fz, energy, virial, x, y, z, q,
                           i0, ni, j0, nj, box, &prep, newton);
    return true;
}

/**
 * @brief `vm_md_neighbors` CSR. `offsets[0]` is 0 and the rows do not shrink.
 *
 * @return 1 when a neighbor is stored, 0 when the list is empty, -1 if rejected.
 */
static int md_coulomb_csr(const int *offsets, const int *neighbors, const int n,
                          const int ni)
{
    int prev = 0;

    if (!offsets || offsets[0] != 0) {
        return -1;
    }

    for (int row = 0; row < ni; ++row) {
        const int end = offsets[row + 1];
        if (end < prev) {
            return -1;
        }
        prev = end;
    }

    if (prev == 0) {
        return 0;
    }

    if (!neighbors) {
        return -1;
    }

    for (int row = 0; row < ni; ++row) {
        for (int p = offsets[row]; p < offsets[row + 1]; ++p) {
            const int j = neighbors[p];
            if (j < 0 || j >= n) {
                return -1;
            }
        }
    }

    return 1;
}

/**
 * @brief Walk one CSR with the rectangular tile's pair step. Force on `i` only.
 */
static void md_coulomb_scalar_list(vm_float_t *fx, vm_float_t *fy, vm_float_t *fz,
                                   vm_float_t *energy, vm_float_t *virial,
                                   const vm_float_t *x, const vm_float_t *y,
                                   const vm_float_t *z, const vm_float_t *q,
                                   const int i0, const int ni, const int *offsets,
                                   const int *neighbors, const vector3 box,
                                   const md_coulomb_prep *prep,
                                   const int *excl_offsets, const int *excl_partners)
{
    vm_float_t sum = VM_F(0.0);
    vm_float_t comp = VM_F(0.0);

    for (int row = 0; row < ni; ++row) {
        const int i = i0 + row;
        for (int p = offsets[row]; p < offsets[row + 1]; ++p) {
            const int j = neighbors[p];
            vm_float_t dx, dy, dz, fix, fiy, fiz, pair_energy;

            if (j == i) {
                continue;
            }

            if (md_list_excluded(excl_offsets, excl_partners, row, j)) {
                continue;
            }

            dx = VECMAT_REMAINDER(x[i] - x[j], box.x);
            dy = VECMAT_REMAINDER(y[i] - y[j], box.y);
            dz = VECMAT_REMAINDER(z[i] - z[j], box.z);
            if (!md_coulomb_pair(prep, q[i], q[j], dx, dy, dz, &fix, &fiy, &fiz,
                                 &pair_energy)) {
                continue;
            }
            md_coulomb_apply(fx, fy, fz, energy ? &sum : NULL, &comp, virial,
                             i, j, 0, dx, dy, dz, fix, fiy, fiz, pair_energy);
        }
    }

    md_coulomb_commit_energy(energy, sum, comp);
}

static bool md_coulomb_list_eval(
    vm_float_t *fx, vm_float_t *fy, vm_float_t *fz,
    vm_float_t *energy, vm_float_t virial[6],
    const vm_float_t *x, const vm_float_t *y, const vm_float_t *z,
    const vm_float_t *q,
    const int n, const int i0, const int ni,
    const int *offsets, const int *neighbors,
    const vector3 box, const vm_float_t prefactor, const vm_float_t rcut,
    const int *excl_offsets, const int *excl_partners)
{
    const int *use_excl = NULL;
    const int *use_partners = NULL;

    if (!md_coulomb_range_ok(n, i0, ni)) {
        return false;
    }

    if (!md_coulomb_positive(prefactor) || !md_coulomb_positive(rcut)) {
        return false;
    }

    if (!md_coulomb_positive(box.x) || !md_coulomb_positive(box.y) ||
        !md_coulomb_positive(box.z)) {
        return false;
    }

    if (ni == 0) {
        return true;
    }

    const int ready = md_coulomb_csr(offsets, neighbors, n, ni);
    if (ready < 0) {
        return false;
    }

    if (excl_offsets) {
        const int excl = md_coulomb_csr(excl_offsets, excl_partners, n, ni);
        if (excl < 0) {
            return false;
        }
        if (excl > 0) {
            use_excl = excl_offsets;
            use_partners = excl_partners;
        }
    }

    if (ready == 0) {
        return true;
    }

    if (!fx || !fy || !fz || !x || !y || !z || !q) {
        return false;
    }

    md_coulomb_prep prep;
    md_coulomb_prepare(&prep, prefactor, rcut);
    md_coulomb_scalar_list(fx, fy, fz, energy, virial, x, y, z, q, i0, ni,
                           offsets, neighbors, box, &prep, use_excl, use_partners);
    return true;
}

bool vm_md_coulomb_list(
    vm_float_t *fx, vm_float_t *fy, vm_float_t *fz,
    vm_float_t *energy, vm_float_t virial[6],
    const vm_float_t *x, const vm_float_t *y, const vm_float_t *z,
    const vm_float_t *q,
    const int n, const int i0, const int ni,
    const int *offsets, const int *neighbors,
    const vector3 box, const vm_float_t prefactor, const vm_float_t rcut)
{
    return md_coulomb_list_eval(fx, fy, fz, energy, virial, x, y, z, q, n, i0, ni,
                                offsets, neighbors, box, prefactor, rcut,
                                NULL, NULL);
}

bool vm_md_coulomb_list_excl(
    vm_float_t *fx, vm_float_t *fy, vm_float_t *fz,
    vm_float_t *energy, vm_float_t virial[6],
    const vm_float_t *x, const vm_float_t *y, const vm_float_t *z,
    const vm_float_t *q,
    const int n, const int i0, const int ni,
    const int *offsets, const int *neighbors,
    const vector3 box, const vm_float_t prefactor, const vm_float_t rcut,
    const int *excl_offsets, const int *excl_partners)
{
    return md_coulomb_list_eval(fx, fy, fz, energy, virial, x, y, z, q, n, i0, ni,
                                offsets, neighbors, box, prefactor, rcut,
                                excl_offsets, excl_partners);
}
