// SPDX-FileCopyrightText: 2025-2026 Igal Alkon
// SPDX-FileCopyrightText: 2026 ALKONTEK <git@alkontek.com>
// SPDX-License-Identifier: BSD-3-Clause

#include "arm_fp16.h"

#include <stddef.h>
#include <string.h>

#if defined(__aarch64__) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
#include <arm_neon.h>

static float16_t load_f16(const vm_fp16_t *p)
{
    float16_t h;
    memcpy(&h, p, sizeof h);
    return h;
}

static float to_f32(const vm_fp16_t *p)
{
    return vgetq_lane_f32(vcvt_f32_f16(vdup_n_f16(load_f16(p))), 0);
}

static void scale_store(float *row, float32x4_t sum, float alpha, float beta)
{
    float32x4_t c = vld1q_f32(row);
    if (beta == 0.0f)
        c = vdupq_n_f32(0.0f);
    else if (beta != 1.0f)
        c = vmulq_n_f32(c, beta);
    vst1q_f32(row, vfmaq_n_f32(c, sum, alpha));
}

void vm_arm_fp16_gemm_cvt(
    float *c, int ldc,
    const vm_fp16_t *a, int lda,
    const vm_fp16_t *b, int ldb,
    int m, int n, int k,
    float alpha, float beta)
{
    if (m <= 0 || n <= 0 || !c)
        return;
    for (int i = 0; i < m; ++i) {
        float *crow = c + (size_t)i * (size_t)ldc;
        const vm_fp16_t *arow = a + (size_t)i * (size_t)lda;
        int j = 0;
        if (alpha != 0.0f && k > 0 && a && b) {
            for (; j + 4 <= n; j += 4) {
                float32x4_t sum = vdupq_n_f32(0.0f);
                for (int p = 0; p < k; ++p) {
                    const float av = to_f32(arow + p);
                    if (av == 0.0f)
                        continue;
                    const vm_fp16_t *brow = b + (size_t)p * (size_t)ldb + j;
                    const float32x4_t bv = vcvt_f32_f16(vld1_f16((const float16_t *)brow));
                    sum = vfmaq_n_f32(sum, bv, av);
                }
                scale_store(crow + j, sum, alpha, beta);
            }
        }
        for (; j < n; ++j) {
            float acc = (beta == 0.0f) ? 0.0f : crow[j] * beta;
            if (alpha != 0.0f && k > 0 && a && b) {
                for (int p = 0; p < k; ++p) {
                    const vm_fp16_t *brow = b + (size_t)p * (size_t)ldb;
                    acc += alpha * to_f32(arow + p) * to_f32(brow + j);
                }
            }
            crow[j] = acc;
        }
    }
}

#if defined(__ARM_FEATURE_FP16_FML)
void vm_arm_fp16_gemm_fmlal(
    float *c, int ldc,
    const vm_fp16_t *a, int lda,
    const vm_fp16_t *b, int ldb,
    int m, int n, int k,
    float alpha, float beta)
{
    if (m <= 0 || n <= 0 || !c)
        return;
    for (int i = 0; i < m; ++i) {
        float *crow = c + (size_t)i * (size_t)ldc;
        const vm_fp16_t *arow = a + (size_t)i * (size_t)lda;
        int j = 0;
        if (alpha != 0.0f && k > 0 && a && b) {
            for (; j + 8 <= n; j += 8) {
                float32x4_t sum0 = vdupq_n_f32(0.0f);
                float32x4_t sum1 = vdupq_n_f32(0.0f);
                for (int p = 0; p < k; ++p) {
                    const vm_fp16_t *brow = b + (size_t)p * (size_t)ldb + j;
                    const float16x8_t as = vdupq_n_f16(load_f16(arow + p));
                    const float16x8_t bs = vld1q_f16((const float16_t *)brow);
                    sum0 = vfmlalq_low_f16(sum0, as, bs);
                    sum1 = vfmlalq_high_f16(sum1, as, bs);
                }
                scale_store(crow + j, sum0, alpha, beta);
                scale_store(crow + j + 4, sum1, alpha, beta);
            }
        }
        for (; j < n; ++j) {
            float acc = (beta == 0.0f) ? 0.0f : crow[j] * beta;
            if (alpha != 0.0f && k > 0 && a && b) {
                for (int p = 0; p < k; ++p) {
                    const vm_fp16_t *brow = b + (size_t)p * (size_t)ldb;
                    acc += alpha * to_f32(arow + p) * to_f32(brow + j);
                }
            }
            crow[j] = acc;
        }
    }
}
#else
void vm_arm_fp16_gemm_fmlal(
    float *c, int ldc,
    const vm_fp16_t *a, int lda,
    const vm_fp16_t *b, int ldb,
    int m, int n, int k,
    float alpha, float beta)
{
    vm_arm_fp16_gemm_cvt(c, ldc, a, lda, b, ldb, m, n, k, alpha, beta);
}
#endif

#else

void vm_arm_fp16_gemm_fmlal(
    float *c, int ldc,
    const vm_fp16_t *a, int lda,
    const vm_fp16_t *b, int ldb,
    int m, int n, int k,
    float alpha, float beta)
{
    (void)c; (void)ldc; (void)a; (void)lda; (void)b; (void)ldb;
    (void)m; (void)n; (void)k; (void)alpha; (void)beta;
}

void vm_arm_fp16_gemm_cvt(
    float *c, int ldc,
    const vm_fp16_t *a, int lda,
    const vm_fp16_t *b, int ldb,
    int m, int n, int k,
    float alpha, float beta)
{
    (void)c; (void)ldc; (void)a; (void)lda; (void)b; (void)ldb;
    (void)m; (void)n; (void)k; (void)alpha; (void)beta;
}

#endif
