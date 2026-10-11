// SPDX-FileCopyrightText: 2025-2026 Igal Alkon
// SPDX-FileCopyrightText: 2026 ALKONTEK <git@alkontek.com>
// SPDX-License-Identifier: BSD-3-Clause

#include <math.h>
#include <string.h>

#include "unitest.h"
#include "vecmat/fp16.h"

static int near(float got, float ref, float tol)
{
    return fabsf(got - ref) <= tol;
}

static int note_ok(const char *note)
{
    return strcmp(note, "fp16=scalar") == 0
        || strcmp(note, "fp16=neon") == 0
        || strcmp(note, "fp16=neon-cvt") == 0;
}

TEST_CASE(arm_fp16_note_and_short_k, "[arm-fp16]")
{
    const float af[] = {1.5f, -2.25f, 0.5f, 4.0f, 0.125f, -8.0f};
    const float bf[] = {3.0f, -1.0f, 0.5f, 2.0f, 6.0f, -0.25f};
    vm_fp16_t a[6], b[6];
    float c[4] = {2.0f, 2.0f, 2.0f, 2.0f};
    float ref[4];
    for (int i = 0; i < 6; ++i) {
        a[i] = vm_fp16_from_f32(af[i]);
        b[i] = vm_fp16_from_f32(bf[i]);
    }
    const int m = 2, n = 2, k = 3;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float acc = 0.0f;
            for (int p = 0; p < k; ++p)
                acc += vm_f32_from_fp16(a[i * k + p]) * vm_f32_from_fp16(b[p * n + j]);
            ref[i * n + j] = 0.5f * 2.0f + 1.5f * acc;
        }
    }
    vm_gemm_fp16f32(c, n, a, k, b, n, m, n, k, 1.5f, 0.5f);
    for (int i = 0; i < 4; ++i)
        REQUIRE(near(c[i], ref[i], 1e-4f));
    REQUIRE(note_ok(vm_fp16_note()));
}

TEST_CASE(arm_fp16_long_k_stays_f32, "[arm-fp16]")
{
    enum { K = 4096 };
    static vm_fp16_t a[K], b[K];
    const vm_fp16_t one = vm_fp16_from_f32(1.0f);
    float c = 0.0f;
    for (int p = 0; p < K; ++p) {
        a[p] = one;
        b[p] = one;
    }
    vm_gemm_fp16f32(&c, 1, a, K, b, 1, 1, 1, K, 1.0f, 0.0f);
    REQUIRE(c == 4096.0f);
    REQUIRE(note_ok(vm_fp16_note()));
}

TEST_CASE(arm_fp16_wide_row, "[arm-fp16]")
{
    enum { N = 10, K = 4 };
    vm_fp16_t a[K], b[K * N];
    float c[N], ref[N];
    for (int p = 0; p < K; ++p)
        a[p] = vm_fp16_from_f32(0.5f * (float)(p + 1));
    for (int i = 0; i < K * N; ++i)
        b[i] = vm_fp16_from_f32((i % 5) - 2.0f);
    for (int j = 0; j < N; ++j) {
        float acc = 0.0f;
        for (int p = 0; p < K; ++p)
            acc += vm_f32_from_fp16(a[p]) * vm_f32_from_fp16(b[p * N + j]);
        ref[j] = acc;
        c[j] = 0.0f;
    }
    vm_gemm_fp16f32(c, N, a, K, b, N, 1, N, K, 1.0f, 0.0f);
    for (int j = 0; j < N; ++j)
        REQUIRE(near(c[j], ref[j], 1e-4f));
}
