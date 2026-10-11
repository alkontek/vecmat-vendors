// SPDX-FileCopyrightText: 2025-2026 Igal Alkon
// SPDX-FileCopyrightText: 2026 ALKONTEK <git@alkontek.com>
// SPDX-License-Identifier: BSD-3-Clause

#include "intel_fp16.h"

#include <stddef.h>

#if defined(__F16C__) && defined(__AVX__)
#include <immintrin.h>

static float to_f32(const vm_fp16_t *p)
{
    const __m128i h = _mm_cvtsi32_si128(*p);
    return _mm_cvtss_f32(_mm_cvtph_ps(h));
}

static void scale_store(float *row, __m256 sum, float alpha, float beta)
{
    __m256 c = _mm256_loadu_ps(row);
    if (beta == 0.0f)
        c = _mm256_setzero_ps();
    else if (beta != 1.0f)
        c = _mm256_mul_ps(c, _mm256_set1_ps(beta));
    _mm256_storeu_ps(row, _mm256_fmadd_ps(sum, _mm256_set1_ps(alpha), c));
}

void vm_intel_fp16_gemm_f16c(
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
                __m256 sum = _mm256_setzero_ps();
                for (int p = 0; p < k; ++p) {
                    const float av = to_f32(arow + p);
                    if (av == 0.0f)
                        continue;
                    const vm_fp16_t *brow = b + (size_t)p * (size_t)ldb + j;
                    const __m128i bh = _mm_loadu_si128((const __m128i *)brow);
                    sum = _mm256_fmadd_ps(_mm256_set1_ps(av), _mm256_cvtph_ps(bh), sum);
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
#else
void vm_intel_fp16_gemm_f16c(
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
