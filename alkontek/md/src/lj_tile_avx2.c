// SPDX-FileCopyrightText: 2025-2026 Igal Alkon
// SPDX-FileCopyrightText: 2026 ALKONTEK <git@alkontek.com>
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file
 * @brief AVX2 shifted-force Lennard-Jones tile.
 *
 * Minimum image stays on `remainder` / `remainderf`, one pair at a time.
 * The polynomial uses separate AVX2 mul, add, div, and sqrt in the scalar
 * order. Contraction is off. The cutoff test is the scalar `r2 >= rcut2`
 * on those `r2` bits. Lanes are applied in increasing `j`, then a scalar tail.
 */

#include <immintrin.h>

#include "lj_internal.h"

#if defined(VECMAT_USE_F64)
#define LANES 4
typedef __m256d vec_t;
#define vload(p)     _mm256_loadu_pd((p))
#define vstore(p, v) _mm256_storeu_pd((p), (v))
#define vadd(a, b)   _mm256_add_pd((a), (b))
#define vsub(a, b)   _mm256_sub_pd((a), (b))
#define vmul(a, b)   _mm256_mul_pd((a), (b))
#define vdiv(a, b)   _mm256_div_pd((a), (b))
#define vsqrt(a)     _mm256_sqrt_pd((a))
#define vset(x)      _mm256_set1_pd((x))
#else
#define LANES 8
typedef __m256 vec_t;
#define vload(p)     _mm256_loadu_ps((p))
#define vstore(p, v) _mm256_storeu_ps((p), (v))
#define vadd(a, b)   _mm256_add_ps((a), (b))
#define vsub(a, b)   _mm256_sub_ps((a), (b))
#define vmul(a, b)   _mm256_mul_ps((a), (b))
#define vdiv(a, b)   _mm256_div_ps((a), (b))
#define vsqrt(a)     _mm256_sqrt_ps((a))
#define vset(x)      _mm256_set1_ps((x))
#endif

void vm_md_lj_tile_avx2(vm_float_t *fx, vm_float_t *fy, vm_float_t *fz,
                        vm_float_t *energy, vm_float_t *virial,
                        const vm_float_t *x, const vm_float_t *y, const vm_float_t *z,
                        const int i0, const int ni, const int j0, const int nj,
                        const vector3 box, const md_lj_prep *prep, const int newton)
{
    const int half = md_lj_half(newton, i0, ni, j0, nj);
    const int i1 = i0 + ni;
    const int j1 = j0 + nj;

    vm_float_t sum = VM_F(0.0);
    vm_float_t comp = VM_F(0.0);

    const vec_t sig2 = vset(prep->sig2);
    const vec_t four_eps = vset(prep->four_eps);
    const vec_t twentyfour = vset(prep->twentyfour_eps);
    const vec_t two = vset(VM_F(2.0));
    const vec_t u_c = vset(prep->u_c);
    const vec_t duc = vset(prep->duc);
    const vec_t rcut = vset(prep->rcut);

    for (int i = i0; i < i1; ++i) {
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
            const vec_t s = vdiv(sig2, r2v);
            const vec_t s2 = vmul(s, s);
            const vec_t s6 = vmul(s2, s);
            const vec_t s12 = vmul(s6, s6);
            const vec_t u = vmul(four_eps, vsub(s12, s6));
            const vec_t c = vdiv(vmul(twentyfour, vsub(vmul(two, s12), s6)), r2v);
            const vec_t r = vsqrt(r2v);
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
                md_lj_apply(fx, fy, fz, energy ? &sum : NULL, &comp, virial,
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
            if (!md_lj_pair(prep, dx, dy, dz, &fix, &fiy, &fiz, &pair_energy)) {
                continue;
            }

            md_lj_apply(fx, fy, fz, energy ? &sum : NULL, &comp, virial,
                        i, j, newton, dx, dy, dz, fix, fiy, fiz, pair_energy);
        }
    }

    md_lj_commit_energy(energy, sum, comp);
}
