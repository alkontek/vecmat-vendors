// SPDX-FileCopyrightText: 2025-2026 Igal Alkon
// SPDX-FileCopyrightText: 2026 ALKONTEK <git@alkontek.com>
// SPDX-License-Identifier: BSD-3-Clause

#include "arm_bf16.h"

#if defined(__aarch64__) && defined(__linux__)
#include <sys/auxv.h>
#ifndef HWCAP2_BF16
#define HWCAP2_BF16 (1UL << 14)
#endif
#endif

unsigned vm_arm_bf16_features(void)
{
#if defined(__aarch64__) && defined(__linux__)
    const unsigned long hw = getauxval(AT_HWCAP2);
    if (hw & HWCAP2_BF16)
        return VM_ARM_BF16_BF16;
    return 0;
#else
    return 0;
#endif
}
