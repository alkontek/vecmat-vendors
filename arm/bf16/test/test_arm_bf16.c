// SPDX-FileCopyrightText: 2025-2026 Igal Alkon
// SPDX-FileCopyrightText: 2026 ALKONTEK <git@alkontek.com>
// SPDX-License-Identifier: BSD-3-Clause

#include <math.h>
#include <string.h>

#include "unitest.h"
#include "vecmat/bf16.h"

TEST_CASE(arm_bf16_note, "[arm-bf16]")
{
    vm_bf16_init();
    const char *note = vm_bf16_note();
    REQUIRE(strcmp(note, "bf16=scalar") == 0
            || strcmp(note, "bf16=neonbf16") == 0);
}

TEST_CASE(arm_bf16_gemm_matches, "[arm-bf16]")
{
    enum { m = 2, n = 4, k = 3 };
    vm_bf16_t a[m * k], b[k * n];
    float c[m * n], ref[m * n];
    for (int i = 0; i < m * k; ++i)
        a[i] = vm_bf16_from_f32(0.5f * (float)(i + 1));
    for (int i = 0; i < k * n; ++i)
        b[i] = vm_bf16_from_f32((i % 3) - 1.0f);
    for (int i = 0; i < m * n; ++i) {
        c[i] = 0.0f;
        ref[i] = 0.0f;
    }
    for (int i = 0; i < m; ++i)
        for (int p = 0; p < k; ++p)
            for (int j = 0; j < n; ++j)
                ref[i * n + j] += vm_f32_from_bf16(a[i * k + p]) *
                                  vm_f32_from_bf16(b[p * n + j]);
    vm_gemm_bf16f32(c, n, a, k, b, n, m, n, k, 1.0f, 0.0f);
    for (int i = 0; i < m * n; ++i)
        REQUIRE(fabsf(c[i] - ref[i]) <= 1e-4f);
}
