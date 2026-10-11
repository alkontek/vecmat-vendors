// SPDX-FileCopyrightText: 2025-2026 Igal Alkon
// SPDX-FileCopyrightText: 2026 ALKONTEK <git@alkontek.com>
// SPDX-License-Identifier: BSD-3-Clause

#include "arm_fp16.h"
#include "vecmat_arm_fp16.h"

int vm_arm_fp16_register(void)
{
    const unsigned features = vm_arm_fp16_features();
    if ((features & VM_ARM_FP16_HP) == 0)
        return 0;
    if (features & VM_ARM_FP16_FHM) {
        vm_fp16_set_gemm(vm_arm_fp16_gemm_fmlal, "fp16=neon");
        return 1;
    }
    vm_fp16_set_gemm(vm_arm_fp16_gemm_cvt, "fp16=neon-cvt");
    return 1;
}
