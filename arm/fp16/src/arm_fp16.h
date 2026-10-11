// SPDX-FileCopyrightText: 2025-2026 Igal Alkon
// SPDX-FileCopyrightText: 2026 ALKONTEK <git@alkontek.com>
// SPDX-License-Identifier: BSD-3-Clause

#ifndef VECMAT_ARM_FP16_INT_H
#define VECMAT_ARM_FP16_INT_H

#include "vecmat/fp16.h"

enum {
    VM_ARM_FP16_HP = 1u << 0,  /**< HWCAP_ASIMDHP, FEAT_FP16. */
    VM_ARM_FP16_FHM = 1u << 1  /**< HWCAP_ASIMDFHM, FEAT_FHM. */
};

/** @brief Runtime fp16 bits. 0 off AArch64 Linux. */
unsigned vm_arm_fp16_features(void);

void vm_arm_fp16_gemm_fmlal(
    float *c, int ldc,
    const vm_fp16_t *a, int lda,
    const vm_fp16_t *b, int ldb,
    int m, int n, int k,
    float alpha, float beta);

void vm_arm_fp16_gemm_cvt(
    float *c, int ldc,
    const vm_fp16_t *a, int lda,
    const vm_fp16_t *b, int ldb,
    int m, int n, int k,
    float alpha, float beta);

#endif
