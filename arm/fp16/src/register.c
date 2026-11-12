// SPDX-FileCopyrightText: 2025-2026 Igal Alkon
// SPDX-FileCopyrightText: 2026 ALKONTEK <git@alkontek.com>
// SPDX-License-Identifier: BSD-3-Clause

#include "arm_fp16.h"
#include "vecmat/cpu.h"
#include "vecmat_arm_fp16.h"

int vm_arm_fp16_register(void)
{
    if (!vm_cpu_has(VM_CPU_ARM_FP16))
        return 0;
    /* FHM is finer than the core bit; still probed here. */
    if (vm_arm_fp16_features() & VM_ARM_FP16_FHM) {
        vm_fp16_set_gemm(vm_arm_fp16_gemm_fmlal, "fp16=neon");
        return 1;
    }
    vm_fp16_set_gemm(vm_arm_fp16_gemm_cvt, "fp16=neon-cvt");
    return 1;
}
