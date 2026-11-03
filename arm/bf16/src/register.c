// SPDX-FileCopyrightText: 2025-2026 Igal Alkon
// SPDX-FileCopyrightText: 2026 ALKONTEK <git@alkontek.com>
// SPDX-License-Identifier: BSD-3-Clause

#include "arm_bf16.h"
#include "vecmat_arm_bf16.h"

int vm_arm_bf16_register(void)
{
    if ((vm_arm_bf16_features() & VM_ARM_BF16_BF16) == 0)
        return 0;
    vm_bf16_set_gemm(vm_arm_bf16_gemm, "bf16=neonbf16");
    return 1;
}
