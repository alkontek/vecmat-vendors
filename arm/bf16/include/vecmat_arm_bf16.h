// SPDX-FileCopyrightText: 2025-2026 Igal Alkon
// SPDX-FileCopyrightText: 2026 ALKONTEK <git@alkontek.com>
// SPDX-License-Identifier: BSD-3-Clause

#ifndef VECMAT_ARM_BF16_H
#define VECMAT_ARM_BF16_H

/** @brief Install the NEON BF16 slot if the CPU supports it. */
int vm_arm_bf16_register(void);

#endif /* VECMAT_ARM_BF16_H */
