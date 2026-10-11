// SPDX-FileCopyrightText: 2025-2026 Igal Alkon
// SPDX-FileCopyrightText: 2026 ALKONTEK <git@alkontek.com>
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file
 * @brief Optional Arm fp16 GEMM override. Not a `vm_backend_ops` filler.
 *
 * Not part of the default library. `vm_fp16_init()` calls
 * `vm_arm_fp16_register()` through the generated fp16 hook. See doc/fp16.md.
 */

#ifndef VECMAT_ARM_FP16_H
#define VECMAT_ARM_FP16_H

#include "vecmat/fp16.h"

/**
 * @brief Override `vm_gemm_fp16f32` when this CPU has `FEAT_FP16`.
 *
 * No `HWCAP_ASIMDHP`: returns 0 and leaves the scalar slot. Both
 * `HWCAP_ASIMDHP` and `HWCAP_ASIMDFHM`: FMLAL kernel, note `fp16=neon`.
 * `HWCAP_ASIMDHP` only: convert to f32 and use f32 FMA, note `fp16=neon-cvt`.
 * No MIDR check. Does not call `vm_backend_register()`.
 */
VEC_API int vm_arm_fp16_register(void);

#endif
