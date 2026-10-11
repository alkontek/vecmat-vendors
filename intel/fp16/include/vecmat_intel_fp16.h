// SPDX-FileCopyrightText: 2025-2026 Igal Alkon
// SPDX-FileCopyrightText: 2026 ALKONTEK <git@alkontek.com>
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file
 * @brief Optional Intel fp16 GEMM override. Not a `vm_backend_ops` filler.
 *
 * Not part of the default library. `vm_fp16_init()` calls
 * `vm_intel_fp16_register()` through the generated fp16 hook.
 */

#ifndef VECMAT_INTEL_FP16_H
#define VECMAT_INTEL_FP16_H

#include "vecmat/fp16.h"

/**
 * @brief Override `vm_gemm_fp16f32` when this CPU can convert binary16.
 *
 * No F16C: returns 0 and leaves the scalar slot. AVX-512 FP16: 16-wide
 * convert plus f32 FMA, note `fp16=avx512fp16`. F16C only: 8-wide convert
 * plus f32 FMA, note `fp16=f16c`. Accumulators stay f32. Does not call
 * `vm_backend_register()`.
 */
VEC_API int vm_intel_fp16_register(void);

#endif
