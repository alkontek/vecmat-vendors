// SPDX-FileCopyrightText: 2025-2026 Igal Alkon
// SPDX-FileCopyrightText: 2026 ALKONTEK <git@alkontek.com>
// SPDX-License-Identifier: BSD-3-Clause

#ifndef VECMAT_INTEL_FP16_INT_H
#define VECMAT_INTEL_FP16_INT_H

#include "vecmat/fp16.h"

enum {
    VM_INTEL_FP16_F16C = 1u << 0,
    VM_INTEL_FP16_AVX512FP16 = 1u << 1
};

/** @brief Runtime fp16 bits. 0 off x86-64. */
unsigned vm_intel_fp16_features(void);

void vm_intel_fp16_gemm_f16c(
    float *c, int ldc,
    const vm_fp16_t *a, int lda,
    const vm_fp16_t *b, int ldb,
    int m, int n, int k,
    float alpha, float beta);

void vm_intel_fp16_gemm_avx512fp16(
    float *c, int ldc,
    const vm_fp16_t *a, int lda,
    const vm_fp16_t *b, int ldb,
    int m, int n, int k,
    float alpha, float beta);

#endif
