// SPDX-FileCopyrightText: 2025-2026 Igal Alkon
// SPDX-FileCopyrightText: 2026 ALKONTEK <git@alkontek.com>
// SPDX-License-Identifier: BSD-3-Clause

#include <stdbool.h>

#if defined(__x86_64__) || defined(_M_X64)
#include <cpuid.h>
#define VECMAT_INTEL_BF16_HOST 1
#else
#define VECMAT_INTEL_BF16_HOST 0
#endif

bool vm_intel_bf16_supported(void)
{
#if VECMAT_INTEL_BF16_HOST
    unsigned int eax, ebx, ecx, edx;
    if (!__get_cpuid_count(7, 1, &eax, &ebx, &ecx, &edx))
        return false;
    return (eax & (1u << 5)) != 0;
#else
    return false;
#endif
}
