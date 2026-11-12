// SPDX-FileCopyrightText: 2025-2026 Igal Alkon
// SPDX-FileCopyrightText: 2026 ALKONTEK <git@alkontek.com>
// SPDX-License-Identifier: BSD-3-Clause

#include "intel_fp16.h"
#include "vecmat/cpu.h"
#include "vecmat_intel_fp16.h"

int vm_intel_fp16_register(void)
{
    /* AVX-512 FP16 has no core bit yet; still probed here. */
    if (vm_intel_fp16_features() & VM_INTEL_FP16_AVX512FP16) {
        vm_fp16_set_gemm(vm_intel_fp16_gemm_avx512fp16, "fp16=avx512fp16");
        return 1;
    }
    if (vm_cpu_has(VM_CPU_F16C)) {
        vm_fp16_set_gemm(vm_intel_fp16_gemm_f16c, "fp16=f16c");
        return 1;
    }
    return 0;
}
