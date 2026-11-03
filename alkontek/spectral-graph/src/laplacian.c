// SPDX-FileCopyrightText: 2025-2026 Igal Alkon
// SPDX-FileCopyrightText: 2026 ALKONTEK <git@alkontek.com>
// SPDX-License-Identifier: BSD-3-Clause

#include <vecmat/feature/spectral_graph.h>

bool vm_laplacian_from_bits(vm_mat *L, const vm_bitset *adj, const int n)
{
    if (!L || !adj || n <= 0) {
        return false;
    }

    *L = vm_mat_alloc(n, n);
    if (!L->data) {
        return false;
    }

    for (int i = 0; i < n; ++i)
    {
        int degree = 0;
        for (int j = 0; j < n; ++j) {
            if (i != j && vm_bitset_test(&adj[i], j)) {
                vm_mat_set(L, i, j, -1);
                ++degree;
            } else if (i != j) {
                vm_mat_set(L, i, j, 0);
            }
        }
        vm_mat_set(L, i, i, (vm_float_t)degree);
    }

    return true;
}
