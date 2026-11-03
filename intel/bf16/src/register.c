// SPDX-FileCopyrightText: 2025-2026 Igal Alkon
// SPDX-FileCopyrightText: 2026 ALKONTEK <git@alkontek.com>
// SPDX-License-Identifier: BSD-3-Clause

#include <stdbool.h>

#include "vecmat/bf16.h"
#include "vecmat_intel_bf16.h"

bool vm_intel_bf16_supported(void);
void vm_intel_bf16_set(void);

void vm_intel_bf16_register(void)
{
    if (vm_intel_bf16_supported())
        vm_intel_bf16_set();
}
