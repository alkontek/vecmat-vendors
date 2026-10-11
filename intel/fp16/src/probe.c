// SPDX-FileCopyrightText: 2025-2026 Igal Alkon
// SPDX-FileCopyrightText: 2026 ALKONTEK <git@alkontek.com>
// SPDX-License-Identifier: BSD-3-Clause

#include "intel_fp16.h"

unsigned vm_intel_fp16_features(void)
{
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    unsigned features = 0;
    __builtin_cpu_init();
    if (__builtin_cpu_supports("f16c"))
        features |= VM_INTEL_FP16_F16C;
    if (__builtin_cpu_supports("avx512fp16"))
        features |= VM_INTEL_FP16_AVX512FP16;
    return features;
#else
    return 0;
#endif
}
