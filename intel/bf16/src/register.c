// SPDX-FileCopyrightText: 2025-2026 Igal Alkon
// SPDX-FileCopyrightText: 2026 ALKONTEK <git@alkontek.com>
// SPDX-License-Identifier: BSD-3-Clause

#include "vecmat/bf16.h"
#include "vecmat/cpu.h"
#include "vecmat_intel_bf16.h"

void vm_intel_bf16_set(void);

void vm_intel_bf16_register(void)
{
    if (vm_cpu_has(VM_CPU_AVX512BF16))
        vm_intel_bf16_set();
}
