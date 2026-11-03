// SPDX-FileCopyrightText: 2025-2026 Igal Alkon
// SPDX-FileCopyrightText: 2026 ALKONTEK <git@alkontek.com>
// SPDX-License-Identifier: BSD-3-Clause

#include <string.h>

#include "unitest.h"
#include "vecmat/bf16.h"

TEST_CASE(intel_bf16_note, "[intel-bf16]")
{
    vm_bf16_init();
    const char *note = vm_bf16_note();
    REQUIRE(strcmp(note, "bf16=scalar") == 0
            || strcmp(note, "bf16=avx512bf16") == 0);
}
