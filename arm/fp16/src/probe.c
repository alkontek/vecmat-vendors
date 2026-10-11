// SPDX-FileCopyrightText: 2025-2026 Igal Alkon
// SPDX-FileCopyrightText: 2026 ALKONTEK <git@alkontek.com>
// SPDX-License-Identifier: BSD-3-Clause

#include "arm_fp16.h"

#if defined(__aarch64__) && defined(__linux__)
#include <sys/auxv.h>
#ifndef HWCAP_ASIMDHP
#define HWCAP_ASIMDHP (1UL << 10)
#endif
#ifndef HWCAP_ASIMDFHM
#define HWCAP_ASIMDFHM (1UL << 23)
#endif
#endif

unsigned vm_arm_fp16_features(void)
{
#if defined(__aarch64__) && defined(__linux__)
    const unsigned long hw = getauxval(AT_HWCAP);
    unsigned features = 0;
    if (hw & HWCAP_ASIMDHP)
        features |= VM_ARM_FP16_HP;
    if (hw & HWCAP_ASIMDFHM)
        features |= VM_ARM_FP16_FHM;
    return features;
#else
    return 0;
#endif
}
