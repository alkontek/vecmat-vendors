// SPDX-FileCopyrightText: 2025-2026 Igal Alkon
// SPDX-FileCopyrightText: 2026 ALKONTEK <git@alkontek.com>
// SPDX-License-Identifier: BSD-3-Clause

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include <vecmat/feature/md.h>

#if !defined(_MSC_VER)
#pragma STDC FP_CONTRACT OFF
#endif

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC optimize("fp-contract=off")
#endif

static int range_ok(const int n, const int i0, const int count)
{
    if (n < 0 || i0 < 0 || count < 0 || i0 > n) {
        return 0;
    }
    return count <= n - i0;
}

bool vm_md_kinetic(
    vm_float_t *energy,
    const vm_float_t *vx, const vm_float_t *vy, const vm_float_t *vz,
    const vm_float_t *mass,
    const int n, const int i0, const int ni)
{
    vm_float_t *terms;

    if (!range_ok(n, i0, ni) || !energy) {
        return false;
    }

    if (ni == 0) {
        *energy = VM_F(0.0);
        return true;
    }

    if (!vx || !vy || !vz || !mass) {
        return false;
    }

    if ((size_t)ni > SIZE_MAX / sizeof(*terms)) {
        return false;
    }

    terms = (vm_float_t *)malloc((size_t)ni * sizeof(*terms));
    if (!terms) {
        return false;
    }

    for (int k = 0; k < ni; ++k) {
        const int i = i0 + k;
        const vm_float_t v2 = (vx[i] * vx[i] + vy[i] * vy[i]) + vz[i] * vz[i];
        terms[k] = (mass[i] * v2) * VM_F(0.5);
    }

    *energy = vm_sum_compensated(terms, ni);
    free(terms);
    return true;
}
