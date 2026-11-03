// SPDX-FileCopyrightText: 2025-2026 Igal Alkon
// SPDX-FileCopyrightText: 2026 ALKONTEK <git@alkontek.com>
// SPDX-License-Identifier: BSD-3-Clause

#ifndef VECMAT_ARM_BF16_INTERNAL_H
#define VECMAT_ARM_BF16_INTERNAL_H

#include "vecmat/bf16.h"

#define VM_ARM_BF16_BF16 (1u << 0)

unsigned vm_arm_bf16_features(void);

void vm_arm_bf16_gemm(
    float *c, int ldc,
    const vm_bf16_t *a, int lda,
    const vm_bf16_t *b, int ldb,
    int m, int n, int k,
    float alpha, float beta);

#endif /* VECMAT_ARM_BF16_INTERNAL_H */
