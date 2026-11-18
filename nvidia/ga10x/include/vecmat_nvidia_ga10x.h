// SPDX-FileCopyrightText: 2025-2026 Igal Alkon
// SPDX-FileCopyrightText: 2026 ALKONTEK <git@alkontek.com>
// SPDX-License-Identifier: BSD-3-Clause

#ifndef VECMAT_NVIDIA_GA10X_H
#define VECMAT_NVIDIA_GA10X_H

/**
 * @file
 * @brief NVIDIA GA10x (GeForce 30xx, compute capability 8.6) device module.
 *
 * Call `vm_nvidia_ga10x_register()` after the Vecmat device API is available.
 * A machine without a GA10x part registers nothing and returns
 * `VM_DEVICE_ERR_UNSUPPORTED`.
 */

#include <vecmat.h>

/**
 * @brief Probe for a compute-capability 8.6 GPU and register it.
 *
 * @return `VM_DEVICE_OK` when `nvidia-ga10x` is registered (including a
 *         repeat call for the same device). `VM_DEVICE_ERR_UNSUPPORTED`
 *         when no GA10x is present. Other `VM_DEVICE_ERR_*` values come
 *         from `vm_device_register()`.
 */
VEC_API int vm_nvidia_ga10x_register(void);

#endif /* VECMAT_NVIDIA_GA10X_H */
