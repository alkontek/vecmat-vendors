// SPDX-FileCopyrightText: 2025-2026 Igal Alkon
// SPDX-FileCopyrightText: 2026 ALKONTEK <git@alkontek.com>
// SPDX-License-Identifier: BSD-3-Clause

#include <stdint.h>
#include <vecmat/feature/spectral_graph.h>

static uint64_t binom(const int n, const int k)
{
    if (k < 0 || k > n) {
        return 0;
    }

    if (k == 0 || k == n) {
        return 1;
    }

    uint64_t r = 1;
    for (int i = 0; i < k; ++i) {
        r = r * (uint64_t)(n - i) / (uint64_t)(i + 1);
    }

    return r;
}

static void unrank_one(const int n, const int k, uint64_t rank, int *idx)
{
    for (int i = k; i >= 1; --i) {
        int c = n - 1;
        while (c >= i && binom(c, i) > rank) {
            --c;
        }
        idx[i - 1] = c;
        rank -= binom(c, i);
    }
}

bool vm_comb_unrank_batch(const int n, const int k, const int batch,
                          const uint64_t *ranks, int *indices)
{
    if (!ranks || !indices || n <= 0 || k <= 0 || k > n || batch <= 0) {
        return false;
    }

    for (int b = 0; b < batch; ++b) {
        unrank_one(n, k, ranks[b], indices + b * k);
    }

    return true;
}
