// SPDX-FileCopyrightText: 2025-2026 Igal Alkon
// SPDX-FileCopyrightText: 2026 ALKONTEK <git@alkontek.com>
// SPDX-License-Identifier: BSD-3-Clause

#include "intel_fp16.h"

#include <stddef.h>

#if defined(__AVX512FP16__)
#include <immintrin.h>

static float to_f32(const vm_fp16_t *p)
{
    const __m128i h = _mm_cvtsi32_si128(*p);
    return _mm_cvtss_f32(_mm_cvtph_ps(h));
}

static void scale_store(float *row, __m512 sum, float beta)
{
    __m512 c = _mm512_loadu_ps(row);
    if (beta == 0.0f)
        c = _mm512_setzero_ps();
    else if (beta != 1.0f)
        c = _mm512_mul_ps(c, _mm512_set1_ps(beta));
    _mm512_storeu_ps(row, _mm512_add_ps(c, sum));
}

static void scale_store_mask(float *row, __mmask16 mask, __m512 sum, float beta)
{
    __m512 c = _mm512_maskz_loadu_ps(mask, row);
    if (beta == 0.0f)
        c = _mm512_setzero_ps();
    else if (beta != 1.0f)
        c = _mm512_mul_ps(c, _mm512_set1_ps(beta));
    _mm512_mask_storeu_ps(row, mask, _mm512_add_ps(c, sum));
}

static __m512 dot16(const vm_fp16_t *arow, const vm_fp16_t *b, int ldb, int j, int k, float alpha)
{
    __m512 sum = _mm512_setzero_ps();
    for (int p = 0; p < k; ++p) {
        const float av = alpha * to_f32(arow + p);
        if (av == 0.0f)
            continue;
        const vm_fp16_t *brow = b + (size_t)p * (size_t)ldb + j;
        const __m256i bh = _mm256_loadu_si256((const __m256i *)brow);
        sum = _mm512_fmadd_ps(_mm512_set1_ps(av), _mm512_cvtph_ps(bh), sum);
    }
    return sum;
}

static void dot32(const vm_fp16_t *arow, const vm_fp16_t *b, int ldb, int j, int k,
                   float alpha, __m512 *sum0, __m512 *sum1)
{
    __m512 s0 = _mm512_setzero_ps();
    __m512 s1 = _mm512_setzero_ps();
    for (int p = 0; p < k; ++p) {
        const float av = alpha * to_f32(arow + p);
        if (av == 0.0f)
            continue;
        const __m512 as = _mm512_set1_ps(av);
        const vm_fp16_t *brow = b + (size_t)p * (size_t)ldb + j;
        const __m256i b0 = _mm256_loadu_si256((const __m256i *)brow);
        const __m256i b1 = _mm256_loadu_si256((const __m256i *)(brow + 16));
        s0 = _mm512_fmadd_ps(as, _mm512_cvtph_ps(b0), s0);
        s1 = _mm512_fmadd_ps(as, _mm512_cvtph_ps(b1), s1);
    }
    *sum0 = s0;
    *sum1 = s1;
}

static __m512 dot16_mask(const vm_fp16_t *arow, const vm_fp16_t *b, int ldb, int j, int k,
                         __mmask16 mask, float alpha)
{
    __m512 sum = _mm512_setzero_ps();
    for (int p = 0; p < k; ++p) {
        const float av = alpha * to_f32(arow + p);
        if (av == 0.0f)
            continue;
        const vm_fp16_t *brow = b + (size_t)p * (size_t)ldb + j;
        const __m256i bh = _mm256_maskz_loadu_epi16(mask, brow);
        sum = _mm512_fmadd_ps(_mm512_set1_ps(av), _mm512_cvtph_ps(bh), sum);
    }
    return sum;
}

void vm_intel_fp16_gemm_avx512fp16(
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
        const int use = alpha != 0.0f && k > 0 && a && b;
        for (; j + 32 <= n; j += 32) {
            __m512 sum0 = _mm512_setzero_ps();
            __m512 sum1 = _mm512_setzero_ps();
            if (use)
                dot32(arow, b, ldb, j, k, alpha, &sum0, &sum1);
            scale_store(crow + j, sum0, beta);
            scale_store(crow + j + 16, sum1, beta);
        }
        for (; j + 16 <= n; j += 16) {
            const __m512 sum = use ? dot16(arow, b, ldb, j, k, alpha) : _mm512_setzero_ps();
            scale_store(crow + j, sum, beta);
        }
        if (j < n) {
            const __mmask16 mask = (__mmask16)((1u << (n - j)) - 1u);
            const __m512 sum = use ? dot16_mask(arow, b, ldb, j, k, mask, alpha) : _mm512_setzero_ps();
            scale_store_mask(crow + j, mask, sum, beta);
        }
    }
}
#else
void vm_intel_fp16_gemm_avx512fp16(
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
