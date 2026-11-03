// SPDX-FileCopyrightText: 2025-2026 Igal Alkon
// SPDX-FileCopyrightText: 2026 ALKONTEK <git@alkontek.com>
// SPDX-License-Identifier: BSD-3-Clause

#include "arm_bf16.h"

#include <stddef.h>

#if defined(__aarch64__) && defined(__ARM_FEATURE_BF16)
#include <arm_neon.h>
#define VECMAT_ARM_BF16_KERNEL 1
#else
#define VECMAT_ARM_BF16_KERNEL 0
#endif

#if VECMAT_ARM_BF16_KERNEL
static float32x4_t load_bf16x4(const vm_bf16_t *p)
{
    uint16_t h[4];
    __builtin_memcpy(h, p, sizeof h);
    float f[4];
    for (int i = 0; i < 4; ++i) {
        const uint32_t bits = (uint32_t)h[i] << 16;
        __builtin_memcpy(&f[i], &bits, sizeof f[i]);
    }
    return vld1q_f32(f);
}

void vm_arm_bf16_gemm(
    float *c, int ldc,
    const vm_bf16_t *a, int lda,
    const vm_bf16_t *b, int ldb,
    int m, int n, int k,
    float alpha, float beta)
{
    if (m <= 0 || n <= 0 || !c)
        return;

    for (int i = 0; i < m; ++i) {
        float *crow = c + (size_t)i * (size_t)ldc;
        const vm_bf16_t *arow = a + (size_t)i * (size_t)lda;

        if (beta == 0.0f) {
            for (int j = 0; j < n; ++j)
                crow[j] = 0.0f;
        } else if (beta != 1.0f) {
            for (int j = 0; j < n; ++j)
                crow[j] *= beta;
        }
        if (alpha == 0.0f || k <= 0 || !a || !b)
            continue;

        int j = 0;
        for (; j + 4 <= n; j += 4) {
            float32x4_t acc = vdupq_n_f32(0.0f);
            for (int p = 0; p < k; ++p) {
                const float av = vm_f32_from_bf16(arow[p]);
                if (av == 0.0f)
                    continue;
                const float32x4_t bv = load_bf16x4(
                    b + (size_t)p * (size_t)ldb + j);
                acc = vfmaq_n_f32(acc, bv, av);
            }
            float32x4_t prev = vld1q_f32(crow + j);
            vst1q_f32(crow + j, vfmaq_n_f32(prev, acc, alpha));
        }
        for (; j < n; ++j) {
            float acc = 0.0f;
            for (int p = 0; p < k; ++p)
                acc += vm_f32_from_bf16(arow[p]) *
                       vm_f32_from_bf16(b[(size_t)p * (size_t)ldb + j]);
            crow[j] += alpha * acc;
        }
    }
}
#else
void vm_arm_bf16_gemm(
    float *c, int ldc,
    const vm_bf16_t *a, int lda,
    const vm_bf16_t *b, int ldb,
    int m, int n, int k,
    float alpha, float beta)
{
    (void)c; (void)ldc; (void)a; (void)lda;
    (void)b; (void)ldb; (void)m; (void)n; (void)k;
    (void)alpha; (void)beta;
}
#endif
