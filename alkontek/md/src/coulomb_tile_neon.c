// SPDX-FileCopyrightText: 2025-2026 Igal Alkon
// SPDX-FileCopyrightText: 2026 ALKONTEK <git@alkontek.com>
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file
 * @brief NEON shifted-force Coulomb tile.
 *
 * Same split as the AVX2 tile: scalar `remainder` for the minimum image,
 * NEON mul/add/div/sqrt in the scalar order with contraction off, and the
 * scalar `r2 >= rcut2` test on the stored `r2` bits.
 */

#include <arm_neon.h>

#include "coulomb_internal.h"

#if defined(VECMAT_USE_F64)
#define LANES 2
typedef float64x2_t vec_t;
#define vload(p)     vld1q_f64((p))
#define vstore(p, v) vst1q_f64((p), (v))
#define vadd(a, b)   vaddq_f64((a), (b))
#define vsub(a, b)   vsubq_f64((a), (b))
#define vmul(a, b)   vmulq_f64((a), (b))
#define vdiv(a, b)   vdivq_f64((a), (b))
#define vsqrt(a)     vsqrtq_f64((a))
#define vset(x)      vdupq_n_f64((x))
#else
#define LANES 4
typedef float32x4_t vec_t;
#define vload(p)     vld1q_f32((p))
#define vstore(p, v) vst1q_f32((p), (v))
#define vadd(a, b)   vaddq_f32((a), (b))
#define vsub(a, b)   vsubq_f32((a), (b))
#define vmul(a, b)   vmulq_f32((a), (b))
#define vdiv(a, b)   vdivq_f32((a), (b))
#define vsqrt(a)     vsqrtq_f32((a))
#define vset(x)      vdupq_n_f32((x))
#endif

void vm_md_coulomb_tile_neon(vm_float_t *fx, vm_float_t *fy, vm_float_t *fz,
                             vm_float_t *energy, vm_float_t *virial,
                             const vm_float_t *x, const vm_float_t *y,
                             const vm_float_t *z, const vm_float_t *q,
                             const int i0, const int ni, const int j0, const int nj,
                             const vector3 box, const md_coulomb_prep *prep,
                             const int newton)
{
    const int half = md_coulomb_half(newton, i0, ni, j0, nj);
    const int i1 = i0 + ni;
    const int j1 = j0 + nj;
    vm_float_t sum = VM_F(0.0);
    vm_float_t comp = VM_F(0.0);
    const vec_t pref = vset(prep->prefactor);
    const vec_t rcut = vset(prep->rcut);

    for (int i = i0; i < i1; ++i) {
        const vec_t qi = vset(q[i]);
        int j = j0;
        for (; j + LANES <= j1; j += LANES) {
            vm_float_t dx[LANES], dy[LANES], dz[LANES];
            vm_float_t r2_store[LANES], r2_use[LANES];
            vm_float_t fix_s[LANES], fiy_s[LANES], fiz_s[LANES], u_s[LANES];
            int idx_skip[LANES];

            for (int lane = 0; lane < LANES; ++lane) {
                const int jj = j + lane;
                idx_skip[lane] = (i == jj) || (half && i > jj);
                dx[lane] = VECMAT_REMAINDER(x[i] - x[jj], box.x);
                dy[lane] = VECMAT_REMAINDER(y[i] - y[jj], box.y);
                dz[lane] = VECMAT_REMAINDER(z[i] - z[jj], box.z);
            }

            const vec_t vx = vload(dx);
            const vec_t vy = vload(dy);
            const vec_t vz = vload(dz);
            const vec_t r2 = vadd(vadd(vmul(vx, vx), vmul(vy, vy)), vmul(vz, vz));
            vstore(r2_store, r2);

            for (int lane = 0; lane < LANES; ++lane) {
                r2_use[lane] = idx_skip[lane] ? VM_F(1.0) : r2_store[lane];
            }

            const vec_t r2v = vload(r2_use);
            const vec_t scale = vmul(pref, vmul(qi, vload(q + j)));
            const vec_t r = vsqrt(r2v);
            const vec_t u = vdiv(scale, r);
            const vec_t u_c = vdiv(scale, rcut);
            const vec_t duc = vdiv(u_c, rcut);
            const vec_t c = vdiv(vdiv(scale, r2v), r);
            const vec_t u_sf = vadd(vsub(u, u_c), vmul(vsub(r, rcut), duc));
            const vec_t coeff = vsub(c, vdiv(duc, r));
            vstore(fix_s, vmul(coeff, vx));
            vstore(fiy_s, vmul(coeff, vy));
            vstore(fiz_s, vmul(coeff, vz));
            vstore(u_s, u_sf);

            for (int lane = 0; lane < LANES; ++lane) {
                if (idx_skip[lane] || r2_store[lane] >= prep->rcut2) {
                    continue;
                }
                md_coulomb_apply(fx, fy, fz, energy ? &sum : NULL, &comp, virial,
                                 i, j + lane, newton, dx[lane], dy[lane], dz[lane],
                                 fix_s[lane], fiy_s[lane], fiz_s[lane], u_s[lane]);
            }
        }

        for (; j < j1; ++j) {
            if (i == j || (half && i > j)) {
                continue;
            }
            const vm_float_t dx = VECMAT_REMAINDER(x[i] - x[j], box.x);
            const vm_float_t dy = VECMAT_REMAINDER(y[i] - y[j], box.y);
            const vm_float_t dz = VECMAT_REMAINDER(z[i] - z[j], box.z);
            vm_float_t fix, fiy, fiz, pair_energy;
            if (!md_coulomb_pair(prep, q[i], q[j], dx, dy, dz, &fix, &fiy, &fiz,
                                 &pair_energy)) {
                continue;
            }
            md_coulomb_apply(fx, fy, fz, energy ? &sum : NULL, &comp, virial,
                             i, j, newton, dx, dy, dz, fix, fiy, fiz, pair_energy);
        }
    }

    md_coulomb_commit_energy(energy, sum, comp);
}
