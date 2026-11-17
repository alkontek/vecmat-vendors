// SPDX-FileCopyrightText: 2025-2026 Igal Alkon
// SPDX-FileCopyrightText: 2026 ALKONTEK <git@alkontek.com>
// SPDX-License-Identifier: BSD-3-Clause

#include <math.h>
#include <string.h>
#include <vecmat.h>
#include <vecmat/feature/md.h>
#include "unitest.h"

#if !defined(_MSC_VER)
#pragma STDC FP_CONTRACT OFF
#endif

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC optimize("fp-contract=off")
#endif

/* Evaluation order matches alkontek/md/src/lj_internal.h. */
typedef struct md_ref_prep {
    vm_float_t rcut;
    vm_float_t rcut2;
    vm_float_t sig2;
    vm_float_t four_eps;
    vm_float_t twentyfour_eps;
    vm_float_t u_c;
    vm_float_t duc;
} md_ref_prep;

static void ref_prepare(md_ref_prep *prep, const vm_float_t epsilon,
                        const vm_float_t sigma, const vm_float_t rcut)
{
    prep->rcut = rcut;
    prep->rcut2 = rcut * rcut;
    prep->sig2 = sigma * sigma;
    prep->four_eps = VM_F(4.0) * epsilon;
    prep->twentyfour_eps = VM_F(24.0) * epsilon;
    const vm_float_t sc = prep->sig2 / prep->rcut2;
    const vm_float_t sc2 = sc * sc;
    const vm_float_t sc6 = sc2 * sc;
    const vm_float_t sc12 = sc6 * sc6;
    prep->u_c = prep->four_eps * (sc12 - sc6);
    const vm_float_t c_c =
        (prep->twentyfour_eps * ((VM_F(2.0) * sc12) - sc6)) / prep->rcut2;
    prep->duc = c_c * rcut;
}

static int ref_pair(const md_ref_prep *prep, const vm_float_t dx,
                    const vm_float_t dy, const vm_float_t dz,
                    vm_float_t *fix, vm_float_t *fiy, vm_float_t *fiz,
                    vm_float_t *energy)
{
    const vm_float_t r2 = (dx * dx + dy * dy) + dz * dz;
    if (r2 >= prep->rcut2) {
        return 0;
    }
    const vm_float_t s = prep->sig2 / r2;
    const vm_float_t s2 = s * s;
    const vm_float_t s6 = s2 * s;
    const vm_float_t s12 = s6 * s6;
    const vm_float_t u = prep->four_eps * (s12 - s6);
    const vm_float_t c =
        (prep->twentyfour_eps * ((VM_F(2.0) * s12) - s6)) / r2;
    const vm_float_t r = VECMAT_SQRT(r2);
    *energy = (u - prep->u_c) + ((r - prep->rcut) * prep->duc);
    const vm_float_t coeff = c - (prep->duc / r);
    *fix = coeff * dx;
    *fiy = coeff * dy;
    *fiz = coeff * dz;
    return 1;
}

static int ref_half(const int newton, const int i0, const int ni,
                    const int j0, const int nj)
{
    return newton && i0 < (j0 + nj) && j0 < (i0 + ni);
}

/* Closed form over vm_periodic_delta_ortho images. Pair energies are
 * appended for vm_sum_compensated. terms_cap == 0 discards them. */
static int ref_tile(vm_float_t *fx, vm_float_t *fy, vm_float_t *fz,
                    vm_float_t *energy, vm_float_t virial[6],
                    const vm_float_t *x, const vm_float_t *y, const vm_float_t *z,
                    const int i0, const int ni, const int j0, const int nj,
                    const vector3 box, const md_ref_prep *prep, const int newton,
                    vm_float_t *terms, const int terms_cap, int *nterms)
{
    const int half = ref_half(newton, i0, ni, j0, nj);
    if (nterms) {
        *nterms = 0;
    }
    for (int i = i0; i < i0 + ni; ++i) {
        for (int j = j0; j < j0 + nj; ++j) {
            if (i == j || (half && i > j)) {
                continue;
            }
            const vector3 a = vec3(x[i], y[i], z[i]);
            const vector3 b = vec3(x[j], y[j], z[j]);
            vector3 d;
            if (!vm_periodic_delta_ortho(&d, &a, &b, box, 1)) {
                return 0;
            }
            vm_float_t fix, fiy, fiz, pair_energy;
            if (!ref_pair(prep, d.x, d.y, d.z, &fix, &fiy, &fiz, &pair_energy)) {
                continue;
            }
            fx[i] += fix;
            fy[i] += fiy;
            fz[i] += fiz;
            if (newton) {
                fx[j] -= fix;
                fy[j] -= fiy;
                fz[j] -= fiz;
            }
            if (energy) {
                if (!terms || !nterms || *nterms >= terms_cap) {
                    return 0;
                }
                terms[*nterms] = pair_energy;
                *nterms += 1;
            }
            if (virial) {
                virial[0] += d.x * fix;
                virial[1] += d.y * fiy;
                virial[2] += d.z * fiz;
                virial[3] += d.x * fiy;
                virial[4] += d.x * fiz;
                virial[5] += d.y * fiz;
            }
        }
    }
    if (energy) {
        *energy += vm_sum_compensated(terms, *nterms);
    }
    return 1;
}

static int same_bits(const vm_float_t a, const vm_float_t b)
{
    return memcmp(&a, &b, sizeof a) == 0;
}

static void require_bits(const vm_float_t got, const vm_float_t exp, const char *what)
{
    if (!same_bits(got, exp)) {
        FAIL("%s got %.9g exp %.9g", what, (double)got, (double)exp);
    }
}

#define MD_N 19

static vector3 fixture_box(void)
{
    return vec3(VM_F(5.5), VM_F(4.25), VM_F(6.0));
}

static void fill_fixture(vm_float_t *x, vm_float_t *y, vm_float_t *z, const vector3 box)
{
    const vm_float_t rcut = VM_F(2.5);
    for (int i = 0; i < MD_N; ++i) {
        x[i] = VM_F(0.17) * (vm_float_t)((i * 3) % 11) + VM_F(0.05) * (vm_float_t)i;
        y[i] = VM_F(0.13) * (vm_float_t)((i * 5) % 9) - VM_F(0.2);
        z[i] = VM_F(0.11) * (vm_float_t)((i * 7) % 13) + VM_F(0.03) * (vm_float_t)(i % 4);
    }
    /* Exact cutoff along x, a ties-to-even image, and a periodic wrap. */
    x[0] = VM_F(0.0);
    y[0] = VM_F(0.0);
    z[0] = VM_F(0.0);
    x[1] = rcut;
    y[1] = VM_F(0.0);
    z[1] = VM_F(0.0);
    x[2] = box.x * VM_F(0.5);
    y[2] = VM_F(0.0);
    z[2] = VM_F(0.0);
    x[3] = box.x - VM_F(0.35);
    y[3] = VM_F(0.15);
    z[3] = VM_F(-0.2);
    x[4] = VM_F(0.4);
    y[4] = VM_F(0.2);
    z[4] = VM_F(-0.1);
}

static void zero_out(vm_float_t *fx, vm_float_t *fy, vm_float_t *fz,
                     vm_float_t *energy, vm_float_t virial[6])
{
    for (int i = 0; i < MD_N; ++i) {
        fx[i] = VM_F(0.0);
        fy[i] = VM_F(0.0);
        fz[i] = VM_F(0.0);
    }
    *energy = VM_F(0.0);
    for (int k = 0; k < 6; ++k) {
        virial[k] = VM_F(0.0);
    }
}

static void require_match(const vm_float_t *gfx, const vm_float_t *gfy, const vm_float_t *gfz,
                          const vm_float_t genergy, const vm_float_t *gvir,
                          const vm_float_t *rfx, const vm_float_t *rfy, const vm_float_t *rfz,
                          const vm_float_t renergy, const vm_float_t *rvir, const int n)
{
    for (int i = 0; i < n; ++i) {
        require_bits(gfx[i], rfx[i], "fx");
        require_bits(gfy[i], rfy[i], "fy");
        require_bits(gfz[i], rfz[i], "fz");
    }
    require_bits(genergy, renergy, "energy");
    for (int k = 0; k < 6; ++k) {
        require_bits(gvir[k], rvir[k], "virial");
    }
}

static void run_both(const vm_float_t *x, const vm_float_t *y, const vm_float_t *z,
                     const int n, const int i0, const int ni, const int j0, const int nj,
                     const vector3 box, const vm_float_t epsilon, const vm_float_t sigma,
                     const vm_float_t rcut, const int newton)
{
    vm_float_t gfx[MD_N], gfy[MD_N], gfz[MD_N], genergy, gvir[6];
    vm_float_t rfx[MD_N], rfy[MD_N], rfz[MD_N], renergy, rvir[6];
    vm_float_t terms[MD_N * MD_N];
    int nterms = 0;
    md_ref_prep prep;
    ref_prepare(&prep, epsilon, sigma, rcut);
    zero_out(gfx, gfy, gfz, &genergy, gvir);
    zero_out(rfx, rfy, rfz, &renergy, rvir);
    REQUIRE(vm_md_lj_tile(gfx, gfy, gfz, &genergy, gvir, x, y, z,
                          n, i0, ni, j0, nj, box, epsilon, sigma, rcut, newton));
    REQUIRE(ref_tile(rfx, rfy, rfz, &renergy, rvir, x, y, z,
                     i0, ni, j0, nj, box, &prep, newton, terms, MD_N * MD_N, &nterms));
    require_match(gfx, gfy, gfz, genergy, gvir, rfx, rfy, rfz, renergy, rvir, n);
}

TEST_CASE(md_one_pair_closed_form_and_minimum_image, "[md]") {
    const vector3 box = fixture_box();
    const vm_float_t epsilon = VM_F(0.8);
    const vm_float_t sigma = VM_F(1.15);
    const vm_float_t rcut = VM_F(2.5);
    vm_float_t x[MD_N], y[MD_N], z[MD_N];
    fill_fixture(x, y, z, box);

    /* Disjoint pair: particle 4 is inside the cutoff of particle 0, and the
     * image of particle 3 wraps. Particle 1 sits on the cutoff. */
    run_both(x, y, z, MD_N, 0, 1, 1, 4, box, epsilon, sigma, rcut, 1);

    const vector3 a = vec3(x[0], y[0], z[0]);
    const vector3 b = vec3(x[3], y[3], z[3]);
    vector3 d = vec3(VM_F(8.0), VM_F(8.0), VM_F(8.0));
    REQUIRE(vm_periodic_delta_ortho(&d, &a, &b, box, 1));
    REQUIRE(same_bits(d.x, VECMAT_REMAINDER(x[0] - x[3], box.x)));
    REQUIRE(same_bits(d.y, VECMAT_REMAINDER(y[0] - y[3], box.y)));
    REQUIRE(same_bits(d.z, VECMAT_REMAINDER(z[0] - z[3], box.z)));
    REQUIRE(d.x >= -box.x * VM_F(0.5) && d.x <= box.x * VM_F(0.5));
}

TEST_CASE(md_cutoff_excludes_exact_and_outside, "[md]") {
    const vector3 box = vec3(VM_F(20.0), VM_F(20.0), VM_F(20.0));
    const vm_float_t epsilon = VM_F(1.0);
    const vm_float_t sigma = VM_F(1.0);
    const vm_float_t rcut = VM_F(2.5);
    vm_float_t x[4] = { VM_F(0.0), rcut, VM_F(3.0), VM_F(1.0) };
    vm_float_t y[4] = { VM_F(0.0), VM_F(0.0), VM_F(0.0), VM_F(0.0) };
    vm_float_t z[4] = { VM_F(0.0), VM_F(0.0), VM_F(0.0), VM_F(0.0) };
    vm_float_t fx[4], fy[4], fz[4], energy, virial[6];
    for (int i = 0; i < 4; ++i) {
        fx[i] = VM_F(0.0);
        fy[i] = VM_F(0.0);
        fz[i] = VM_F(0.0);
    }
    energy = VM_F(0.0);
    for (int k = 0; k < 6; ++k) {
        virial[k] = VM_F(0.0);
    }

    /* j = 1 is r == rcut, j = 2 is outside, j = 3 is inside. */
    REQUIRE(vm_md_lj_tile(fx, fy, fz, &energy, virial, x, y, z,
                          4, 0, 1, 1, 3, box, epsilon, sigma, rcut, 1));
    REQUIRE(same_bits(fx[1], VM_F(0.0)));
    REQUIRE(same_bits(fx[2], VM_F(0.0)));
    REQUIRE(!same_bits(fx[0], VM_F(0.0)));
    REQUIRE(!same_bits(fx[3], VM_F(0.0)));
    REQUIRE(same_bits(fx[3], -fx[0]));

    vm_float_t zfx[4], zfy[4], zfz[4], zenergy = VM_F(4.0), zvir[6];
    for (int i = 0; i < 4; ++i) {
        zfx[i] = VM_F(4.0);
        zfy[i] = VM_F(4.0);
        zfz[i] = VM_F(4.0);
    }
    for (int k = 0; k < 6; ++k) {
        zvir[k] = VM_F(4.0);
    }
    x[1] = VM_F(9.0);
    REQUIRE(vm_md_lj_tile(zfx, zfy, zfz, &zenergy, zvir, x, y, z,
                          4, 0, 1, 1, 1, box, epsilon, sigma, rcut, 0));
    REQUIRE(same_bits(zfx[0], VM_F(4.0)));
    REQUIRE(same_bits(zenergy, VM_F(4.0)));
    REQUIRE(same_bits(zvir[0], VM_F(4.0)));
}

TEST_CASE(md_skips_self_pairs, "[md]") {
    const vector3 box = fixture_box();
    vm_float_t x[1] = { VM_F(1.0) };
    vm_float_t y[1] = { VM_F(2.0) };
    vm_float_t z[1] = { VM_F(3.0) };
    vm_float_t fx[1] = { VM_F(0.0) };
    vm_float_t fy[1] = { VM_F(0.0) };
    vm_float_t fz[1] = { VM_F(0.0) };
    vm_float_t energy = VM_F(0.0);
    vm_float_t virial[6];
    for (int k = 0; k < 6; ++k) {
        virial[k] = VM_F(0.0);
    }
    REQUIRE(vm_md_lj_tile(fx, fy, fz, &energy, virial, x, y, z,
                          1, 0, 1, 0, 1, box, VM_F(1.0), VM_F(1.0), VM_F(2.0), 1));
    REQUIRE(same_bits(fx[0], VM_F(0.0)));
    REQUIRE(same_bits(fy[0], VM_F(0.0)));
    REQUIRE(same_bits(fz[0], VM_F(0.0)));
    REQUIRE(same_bits(energy, VM_F(0.0)));
    REQUIRE(same_bits(virial[0], VM_F(0.0)));
}

TEST_CASE(md_newton_overlap_and_disjoint, "[md]") {
    const vector3 box = fixture_box();
    const vm_float_t epsilon = VM_F(0.8);
    const vm_float_t sigma = VM_F(1.15);
    const vm_float_t rcut = VM_F(2.5);
    vm_float_t x[MD_N], y[MD_N], z[MD_N];
    fill_fixture(x, y, z, box);

    /* Overlapping self-tile keeps i < j when newton is on. */
    run_both(x, y, z, MD_N, 0, 8, 0, 8, box, epsilon, sigma, rcut, 1);
    /* Partial overlap. */
    run_both(x, y, z, MD_N, 0, 6, 3, 6, box, epsilon, sigma, rcut, 1);
    /* Disjoint ranges apply every pair, including i > j, and the reaction. */
    run_both(x, y, z, MD_N, 0, 4, 8, 7, box, epsilon, sigma, rcut, 1);
    /* newton off on an overlap adds force on i only, both orders. */
    run_both(x, y, z, MD_N, 0, 6, 2, 6, box, epsilon, sigma, rcut, 0);
    /* newton off, disjoint: j is left untouched by the kernel. */
    run_both(x, y, z, MD_N, 1, 3, 10, 4, box, epsilon, sigma, rcut, 0);
}

TEST_CASE(md_empty_tile_writes_nothing, "[md]") {
    const vector3 box = fixture_box();
    vm_float_t fx = VM_F(7.0), fy = VM_F(7.0), fz = VM_F(7.0);
    vm_float_t energy = VM_F(7.0);
    vm_float_t virial[6];
    for (int k = 0; k < 6; ++k) {
        virial[k] = VM_F(7.0);
    }
    REQUIRE(vm_md_lj_tile(&fx, &fy, &fz, &energy, virial, NULL, NULL, NULL,
                          4, 0, 0, 0, 4, box, VM_F(1.0), VM_F(1.0), VM_F(2.0), 1));
    REQUIRE(vm_md_lj_tile(NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
                          0, 0, 0, 0, 0, box, VM_F(1.0), VM_F(1.0), VM_F(2.0), 0));
    REQUIRE(same_bits(fx, VM_F(7.0)));
    REQUIRE(same_bits(energy, VM_F(7.0)));
    REQUIRE(same_bits(virial[3], VM_F(7.0)));
}

TEST_CASE(md_rejects_bad_arguments_without_writes, "[md]") {
    const vector3 box = fixture_box();
    vm_float_t x[2] = { VM_F(0.0), VM_F(1.0) };
    vm_float_t y[2] = { VM_F(0.0), VM_F(0.2) };
    vm_float_t z[2] = { VM_F(0.0), VM_F(-0.2) };
    vm_float_t fx[2] = { VM_F(3.0), VM_F(3.0) };
    vm_float_t fy[2] = { VM_F(3.0), VM_F(3.0) };
    vm_float_t fz[2] = { VM_F(3.0), VM_F(3.0) };
    vm_float_t energy = VM_F(3.0);
    vm_float_t virial[6];
    for (int k = 0; k < 6; ++k) {
        virial[k] = VM_F(3.0);
    }
    const vector3 bad = vec3(VM_F(0.0), VM_F(1.0), VM_F(1.0));
    REQUIRE(!vm_md_lj_tile(fx, fy, fz, &energy, virial, x, y, z,
                           2, 0, 1, 1, 1, bad, VM_F(1.0), VM_F(1.0), VM_F(2.0), 1));
    REQUIRE(!vm_md_lj_tile(fx, fy, fz, &energy, virial, x, y, z,
                           2, 0, 1, 1, 1, box, VM_F(-1.0), VM_F(1.0), VM_F(2.0), 1));
    REQUIRE(!vm_md_lj_tile(fx, fy, fz, &energy, virial, x, y, z,
                           2, 0, 1, 1, 1, box, VM_F(1.0), NAN, VM_F(2.0), 1));
    REQUIRE(!vm_md_lj_tile(fx, fy, fz, &energy, virial, x, y, z,
                           2, 0, 1, 1, 1, box, VM_F(1.0), VM_F(1.0), VM_F(0.0), 1));
    REQUIRE(!vm_md_lj_tile(fx, fy, fz, &energy, virial, x, y, z,
                           -1, 0, 1, 0, 1, box, VM_F(1.0), VM_F(1.0), VM_F(2.0), 1));
    REQUIRE(!vm_md_lj_tile(fx, fy, fz, &energy, virial, x, y, z,
                           2, -1, 1, 0, 1, box, VM_F(1.0), VM_F(1.0), VM_F(2.0), 1));
    REQUIRE(!vm_md_lj_tile(fx, fy, fz, &energy, virial, x, y, z,
                           2, 0, 3, 0, 1, box, VM_F(1.0), VM_F(1.0), VM_F(2.0), 1));
    REQUIRE(!vm_md_lj_tile(NULL, fy, fz, &energy, virial, x, y, z,
                           2, 0, 1, 1, 1, box, VM_F(1.0), VM_F(1.0), VM_F(2.0), 1));
    REQUIRE(!vm_md_lj_tile(fx, fy, fz, &energy, virial, NULL, y, z,
                           2, 0, 1, 1, 1, box, VM_F(1.0), VM_F(1.0), VM_F(2.0), 1));
    REQUIRE(same_bits(fx[0], VM_F(3.0)));
    REQUIRE(same_bits(fy[1], VM_F(3.0)));
    REQUIRE(same_bits(energy, VM_F(3.0)));
    REQUIRE(same_bits(virial[5], VM_F(3.0)));

    /* NULL energy and virial are allowed. The end index may equal n. */
    REQUIRE(vm_md_lj_tile(fx, fy, fz, NULL, NULL, x, y, z,
                          2, 0, 1, 1, 1, box, VM_F(1.0), VM_F(1.0), VM_F(2.0), 0));
}

TEST_CASE(md_energy_matches_compensated_sum, "[md]") {
    const vector3 box = fixture_box();
    const vm_float_t epsilon = VM_F(0.8);
    const vm_float_t sigma = VM_F(1.15);
    const vm_float_t rcut = VM_F(2.5);
    vm_float_t x[MD_N], y[MD_N], z[MD_N];
    fill_fixture(x, y, z, box);
    vm_float_t fx[MD_N], fy[MD_N], fz[MD_N], energy, virial[6];
    vm_float_t terms[MD_N * MD_N];
    int nterms = 0;
    md_ref_prep prep;
    ref_prepare(&prep, epsilon, sigma, rcut);
    zero_out(fx, fy, fz, &energy, virial);
    vm_float_t rfx[MD_N], rfy[MD_N], rfz[MD_N], renergy = VM_F(0.0), rvir[6];
    zero_out(rfx, rfy, rfz, &renergy, rvir);
    REQUIRE(ref_tile(rfx, rfy, rfz, &renergy, rvir, x, y, z,
                     0, MD_N, 0, MD_N, box, &prep, 1, terms, MD_N * MD_N, &nterms));
    REQUIRE(nterms > 1);
    const vm_float_t oracle = vm_sum_compensated(terms, nterms);
    REQUIRE(same_bits(renergy, oracle));
    REQUIRE(vm_md_lj_tile(fx, fy, fz, &energy, virial, x, y, z,
                          MD_N, 0, MD_N, 0, MD_N, box, epsilon, sigma, rcut, 1));
    require_bits(energy, oracle, "compensated energy");
    require_match(fx, fy, fz, energy, virial, rfx, rfy, rfz, renergy, rvir, MD_N);
}

TEST_CASE(md_simd_matches_scalar_tile_on_fixture, "[md]") {
    const vector3 box = fixture_box();
    const vm_float_t epsilon = VM_F(0.8);
    const vm_float_t sigma = VM_F(1.15);
    const vm_float_t rcut = VM_F(2.5);
    vm_float_t x[MD_N], y[MD_N], z[MD_N];
    fill_fixture(x, y, z, box);

    /* Lengths 1..19 hit every AVX2 and NEON tail, including a full self-tile.
     * On this host the kernel takes the SIMD path when that ISA was compiled;
     * the reference is the scalar closed form, cutoff included. */
    for (int n = 1; n <= MD_N; ++n) {
        run_both(x, y, z, n, 0, n, 0, n, box, epsilon, sigma, rcut, 1);
        if (n >= 4) {
            run_both(x, y, z, n, 0, n / 2, n / 2, n - n / 2, box, epsilon, sigma, rcut, 1);
        }
    }
}

/* Evaluation order matches alkontek/md/src/coulomb_internal.h. */
typedef struct md_cref_prep {
    vm_float_t rcut;
    vm_float_t rcut2;
    vm_float_t prefactor;
} md_cref_prep;

static void cref_prepare(md_cref_prep *prep, const vm_float_t prefactor,
                         const vm_float_t rcut)
{
    prep->rcut = rcut;
    prep->rcut2 = rcut * rcut;
    prep->prefactor = prefactor;
}

static int cref_pair(const md_cref_prep *prep, const vm_float_t qi, const vm_float_t qj,
                     const vm_float_t dx, const vm_float_t dy, const vm_float_t dz,
                     vm_float_t *fix, vm_float_t *fiy, vm_float_t *fiz,
                     vm_float_t *energy)
{
    const vm_float_t r2 = (dx * dx + dy * dy) + dz * dz;
    if (r2 >= prep->rcut2) {
        return 0;
    }
    const vm_float_t scale = prep->prefactor * (qi * qj);
    const vm_float_t r = VECMAT_SQRT(r2);
    const vm_float_t u = scale / r;
    const vm_float_t u_c = scale / prep->rcut;
    const vm_float_t duc = u_c / prep->rcut;
    const vm_float_t c = (scale / r2) / r;
    *energy = (u - u_c) + ((r - prep->rcut) * duc);
    const vm_float_t coeff = c - (duc / r);
    *fix = coeff * dx;
    *fiy = coeff * dy;
    *fiz = coeff * dz;
    return 1;
}

/* Closed form over vm_periodic_delta_ortho images. Pair energies are
 * appended for vm_sum_compensated. terms_cap == 0 discards them. */
static int cref_tile(vm_float_t *fx, vm_float_t *fy, vm_float_t *fz,
                     vm_float_t *energy, vm_float_t virial[6],
                     const vm_float_t *x, const vm_float_t *y, const vm_float_t *z,
                     const vm_float_t *q, const int i0, const int ni, const int j0,
                     const int nj, const vector3 box, const md_cref_prep *prep,
                     const int newton, vm_float_t *terms, const int terms_cap,
                     int *nterms)
{
    const int half = ref_half(newton, i0, ni, j0, nj);
    if (nterms) {
        *nterms = 0;
    }
    for (int i = i0; i < i0 + ni; ++i) {
        for (int j = j0; j < j0 + nj; ++j) {
            if (i == j || (half && i > j)) {
                continue;
            }
            const vector3 a = vec3(x[i], y[i], z[i]);
            const vector3 b = vec3(x[j], y[j], z[j]);
            vector3 d;
            if (!vm_periodic_delta_ortho(&d, &a, &b, box, 1)) {
                return 0;
            }
            vm_float_t fix, fiy, fiz, pair_energy;
            if (!cref_pair(prep, q[i], q[j], d.x, d.y, d.z, &fix, &fiy, &fiz,
                           &pair_energy)) {
                continue;
            }
            fx[i] += fix;
            fy[i] += fiy;
            fz[i] += fiz;
            if (newton) {
                fx[j] -= fix;
                fy[j] -= fiy;
                fz[j] -= fiz;
            }
            if (energy) {
                if (!terms || !nterms || *nterms >= terms_cap) {
                    return 0;
                }
                terms[*nterms] = pair_energy;
                *nterms += 1;
            }
            if (virial) {
                virial[0] += d.x * fix;
                virial[1] += d.y * fiy;
                virial[2] += d.z * fiz;
                virial[3] += d.x * fiy;
                virial[4] += d.x * fiz;
                virial[5] += d.y * fiz;
            }
        }
    }
    if (energy) {
        *energy += vm_sum_compensated(terms, *nterms);
    }
    return 1;
}

static void fill_charges(vm_float_t *q)
{
    for (int i = 0; i < MD_N; ++i) {
        const int k = (i % 5) - 2;
        q[i] = VM_F(0.5) * (vm_float_t)k;
    }
    q[0] = VM_F(1.0);
    q[1] = VM_F(-0.5);
    q[2] = VM_F(0.8);
    q[3] = VM_F(-1.5);
    q[4] = VM_F(0.75);
    q[5] = VM_F(0.0);
}

static void run_coulomb(const vm_float_t *x, const vm_float_t *y, const vm_float_t *z,
                        const vm_float_t *q, const int n, const int i0, const int ni,
                        const int j0, const int nj, const vector3 box,
                        const vm_float_t prefactor, const vm_float_t rcut,
                        const int newton, const vm_float_t seed)
{
    vm_float_t gfx[MD_N], gfy[MD_N], gfz[MD_N], genergy, gvir[6];
    vm_float_t rfx[MD_N], rfy[MD_N], rfz[MD_N], renergy, rvir[6];
    vm_float_t terms[MD_N * MD_N];
    int nterms = 0;
    md_cref_prep prep;
    cref_prepare(&prep, prefactor, rcut);
    for (int i = 0; i < n; ++i) {
        gfx[i] = seed;
        gfy[i] = seed;
        gfz[i] = seed;
        rfx[i] = seed;
        rfy[i] = seed;
        rfz[i] = seed;
    }
    genergy = seed;
    renergy = seed;
    for (int k = 0; k < 6; ++k) {
        gvir[k] = seed;
        rvir[k] = seed;
    }
    REQUIRE(vm_md_coulomb_tile(gfx, gfy, gfz, &genergy, gvir, x, y, z, q,
                               n, i0, ni, j0, nj, box, prefactor, rcut, newton));
    REQUIRE(cref_tile(rfx, rfy, rfz, &renergy, rvir, x, y, z, q,
                      i0, ni, j0, nj, box, &prep, newton, terms, MD_N * MD_N, &nterms));
    require_match(gfx, gfy, gfz, genergy, gvir, rfx, rfy, rfz, renergy, rvir, n);
}

TEST_CASE(md_coulomb_one_pair_closed_form_and_minimum_image, "[md]") {
    const vector3 box = fixture_box();
    const vm_float_t prefactor = VM_F(2.3);
    const vm_float_t rcut = VM_F(2.5);
    vm_float_t x[MD_N], y[MD_N], z[MD_N], q[MD_N];
    fill_fixture(x, y, z, box);
    fill_charges(q);

    /* Disjoint pair: particle 4 repels 0, the image of 3 attracts 0, and
     * particle 1 sits on the cutoff. Seeded outputs must be added into. */
    run_coulomb(x, y, z, q, MD_N, 0, 1, 1, 4, box, prefactor, rcut, 1, VM_F(1.5));

    const vector3 a = vec3(x[0], y[0], z[0]);
    const vector3 b = vec3(x[3], y[3], z[3]);
    vector3 d = vec3(VM_F(8.0), VM_F(8.0), VM_F(8.0));
    REQUIRE(vm_periodic_delta_ortho(&d, &a, &b, box, 1));
    REQUIRE(same_bits(d.x, VECMAT_REMAINDER(x[0] - x[3], box.x)));
    REQUIRE(same_bits(d.y, VECMAT_REMAINDER(y[0] - y[3], box.y)));
    REQUIRE(same_bits(d.z, VECMAT_REMAINDER(z[0] - z[3], box.z)));
    REQUIRE(d.x >= -box.x * VM_F(0.5) && d.x <= box.x * VM_F(0.5));
}

TEST_CASE(md_coulomb_shift_sign_and_single_count, "[md]") {
    const vector3 box = vec3(VM_F(20.0), VM_F(20.0), VM_F(20.0));
    const vm_float_t prefactor = VM_F(2.3);
    const vm_float_t rcut = VM_F(2.5);
    vm_float_t x[2] = { VM_F(0.0), VM_F(1.0) };
    vm_float_t y[2] = { VM_F(0.0), VM_F(0.0) };
    vm_float_t z[2] = { VM_F(0.0), VM_F(0.0) };
    vm_float_t q[2] = { VM_F(1.0), VM_F(1.0) };
    vm_float_t fx[2], fy[2], fz[2], energy, virial[6];
    vm_float_t terms[4];
    int nterms = 0;
    md_cref_prep prep;
    cref_prepare(&prep, prefactor, rcut);

    for (int i = 0; i < 2; ++i) {
        fx[i] = VM_F(0.0);
        fy[i] = VM_F(0.0);
        fz[i] = VM_F(0.0);
    }
    energy = VM_F(0.0);
    for (int k = 0; k < 6; ++k) {
        virial[k] = VM_F(0.0);
    }
    REQUIRE(vm_md_coulomb_tile(fx, fy, fz, &energy, virial, x, y, z, q,
                               2, 0, 1, 1, 1, box, prefactor, rcut, 1));
    {
        vm_float_t rfx[2] = { VM_F(0.0), VM_F(0.0) };
        vm_float_t rfy[2] = { VM_F(0.0), VM_F(0.0) };
        vm_float_t rfz[2] = { VM_F(0.0), VM_F(0.0) };
        vm_float_t renergy = VM_F(0.0);
        vm_float_t rvir[6] = { VM_F(0.0), VM_F(0.0), VM_F(0.0),
                               VM_F(0.0), VM_F(0.0), VM_F(0.0) };
        REQUIRE(cref_tile(rfx, rfy, rfz, &renergy, rvir, x, y, z, q,
                          0, 1, 1, 1, box, &prep, 1, terms, 4, &nterms));
        REQUIRE(nterms == 1);
        REQUIRE(same_bits(renergy, terms[0]));
        REQUIRE(same_bits(renergy, vm_sum_compensated(terms, nterms)));
        require_match(fx, fy, fz, energy, virial, rfx, rfy, rfz, renergy, rvir, 2);
    }

    const vector3 a = vec3(x[0], y[0], z[0]);
    const vector3 b = vec3(x[1], y[1], z[1]);
    vector3 d;
    REQUIRE(vm_periodic_delta_ortho(&d, &a, &b, box, 1));
    /* Like charges repel: dx * Fx > 0, same virial sign as the LJ tile. */
    REQUIRE(d.x * fx[0] > VM_F(0.0));
    REQUIRE(virial[0] > VM_F(0.0));
    REQUIRE(same_bits(fx[1], -fx[0]));
    REQUIRE(same_bits(fy[0], VM_F(0.0)));
    REQUIRE(same_bits(fz[0], VM_F(0.0)));

    const vm_float_t scale = prefactor * (q[0] * q[1]);
    const vm_float_t r = VECMAT_SQRT((d.x * d.x + d.y * d.y) + d.z * d.z);
    const vm_float_t plain_u = scale / r;
    const vm_float_t plain_c = (scale / ((d.x * d.x + d.y * d.y) + d.z * d.z)) / r;
    REQUIRE(!same_bits(energy, plain_u));
    REQUIRE(!same_bits(fx[0], plain_c * d.x));

    q[1] = VM_F(-1.0);
    for (int i = 0; i < 2; ++i) {
        fx[i] = VM_F(0.0);
        fy[i] = VM_F(0.0);
        fz[i] = VM_F(0.0);
    }
    energy = VM_F(0.0);
    for (int k = 0; k < 6; ++k) {
        virial[k] = VM_F(0.0);
    }
    REQUIRE(vm_md_coulomb_tile(fx, fy, fz, &energy, virial, x, y, z, q,
                               2, 0, 1, 1, 1, box, prefactor, rcut, 1));
    REQUIRE(d.x * fx[0] < VM_F(0.0));
    REQUIRE(virial[0] < VM_F(0.0));
    REQUIRE(same_bits(fx[1], -fx[0]));
}

TEST_CASE(md_coulomb_cutoff_excludes_exact_and_outside, "[md]") {
    const vector3 box = vec3(VM_F(20.0), VM_F(20.0), VM_F(20.0));
    const vm_float_t prefactor = VM_F(2.3);
    const vm_float_t rcut = VM_F(2.5);
    vm_float_t x[4] = { VM_F(0.0), rcut, VM_F(3.0), VM_F(1.0) };
    vm_float_t y[4] = { VM_F(0.0), VM_F(0.0), VM_F(0.0), VM_F(0.0) };
    vm_float_t z[4] = { VM_F(0.0), VM_F(0.0), VM_F(0.0), VM_F(0.0) };
    vm_float_t q[4] = { VM_F(1.0), VM_F(1.0), VM_F(-1.0), VM_F(0.5) };
    vm_float_t fx[4], fy[4], fz[4], energy, virial[6];
    for (int i = 0; i < 4; ++i) {
        fx[i] = VM_F(0.0);
        fy[i] = VM_F(0.0);
        fz[i] = VM_F(0.0);
    }
    energy = VM_F(0.0);
    for (int k = 0; k < 6; ++k) {
        virial[k] = VM_F(0.0);
    }

    /* j = 1 is r == rcut, j = 2 is outside, j = 3 is inside. */
    REQUIRE(vm_md_coulomb_tile(fx, fy, fz, &energy, virial, x, y, z, q,
                               4, 0, 1, 1, 3, box, prefactor, rcut, 1));
    REQUIRE(same_bits(fx[1], VM_F(0.0)));
    REQUIRE(same_bits(fx[2], VM_F(0.0)));
    REQUIRE(!same_bits(fx[0], VM_F(0.0)));
    REQUIRE(!same_bits(fx[3], VM_F(0.0)));
    REQUIRE(same_bits(fx[3], -fx[0]));

    vm_float_t zfx[4], zfy[4], zfz[4], zenergy = VM_F(4.0), zvir[6];
    for (int i = 0; i < 4; ++i) {
        zfx[i] = VM_F(4.0);
        zfy[i] = VM_F(4.0);
        zfz[i] = VM_F(4.0);
    }
    for (int k = 0; k < 6; ++k) {
        zvir[k] = VM_F(4.0);
    }
    x[1] = VM_F(9.0);
    REQUIRE(vm_md_coulomb_tile(zfx, zfy, zfz, &zenergy, zvir, x, y, z, q,
                               4, 0, 1, 1, 1, box, prefactor, rcut, 0));
    REQUIRE(same_bits(zfx[0], VM_F(4.0)));
    REQUIRE(same_bits(zenergy, VM_F(4.0)));
    REQUIRE(same_bits(zvir[0], VM_F(4.0)));
}

TEST_CASE(md_coulomb_skips_self_pairs, "[md]") {
    const vector3 box = fixture_box();
    vm_float_t x[1] = { VM_F(1.0) };
    vm_float_t y[1] = { VM_F(2.0) };
    vm_float_t z[1] = { VM_F(3.0) };
    vm_float_t q[1] = { VM_F(1.0) };
    vm_float_t fx[1] = { VM_F(0.0) };
    vm_float_t fy[1] = { VM_F(0.0) };
    vm_float_t fz[1] = { VM_F(0.0) };
    vm_float_t energy = VM_F(0.0);
    vm_float_t virial[6];
    for (int k = 0; k < 6; ++k) {
        virial[k] = VM_F(0.0);
    }
    REQUIRE(vm_md_coulomb_tile(fx, fy, fz, &energy, virial, x, y, z, q,
                               1, 0, 1, 0, 1, box, VM_F(2.3), VM_F(2.0), 1));
    REQUIRE(same_bits(fx[0], VM_F(0.0)));
    REQUIRE(same_bits(fy[0], VM_F(0.0)));
    REQUIRE(same_bits(fz[0], VM_F(0.0)));
    REQUIRE(same_bits(energy, VM_F(0.0)));
    REQUIRE(same_bits(virial[0], VM_F(0.0)));
}

TEST_CASE(md_coulomb_newton_overlap_and_disjoint, "[md]") {
    const vector3 box = fixture_box();
    const vm_float_t prefactor = VM_F(2.3);
    const vm_float_t rcut = VM_F(2.5);
    vm_float_t x[MD_N], y[MD_N], z[MD_N], q[MD_N];
    fill_fixture(x, y, z, box);
    fill_charges(q);

    run_coulomb(x, y, z, q, MD_N, 0, 8, 0, 8, box, prefactor, rcut, 1, VM_F(0.0));
    run_coulomb(x, y, z, q, MD_N, 0, 6, 3, 6, box, prefactor, rcut, 1, VM_F(0.0));
    run_coulomb(x, y, z, q, MD_N, 0, 4, 8, 7, box, prefactor, rcut, 1, VM_F(0.0));
    run_coulomb(x, y, z, q, MD_N, 0, 6, 2, 6, box, prefactor, rcut, 0, VM_F(0.0));
    run_coulomb(x, y, z, q, MD_N, 1, 3, 10, 4, box, prefactor, rcut, 0, VM_F(0.0));
}

TEST_CASE(md_coulomb_empty_tile_writes_nothing, "[md]") {
    const vector3 box = fixture_box();
    vm_float_t fx = VM_F(7.0), fy = VM_F(7.0), fz = VM_F(7.0);
    vm_float_t energy = VM_F(7.0);
    vm_float_t virial[6];
    for (int k = 0; k < 6; ++k) {
        virial[k] = VM_F(7.0);
    }
    REQUIRE(vm_md_coulomb_tile(&fx, &fy, &fz, &energy, virial, NULL, NULL, NULL, NULL,
                               4, 0, 0, 0, 4, box, VM_F(2.3), VM_F(2.0), 1));
    REQUIRE(vm_md_coulomb_tile(NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
                               0, 0, 0, 0, 0, box, VM_F(2.3), VM_F(2.0), 0));
    REQUIRE(same_bits(fx, VM_F(7.0)));
    REQUIRE(same_bits(energy, VM_F(7.0)));
    REQUIRE(same_bits(virial[3], VM_F(7.0)));
}

TEST_CASE(md_coulomb_rejects_bad_arguments_without_writes, "[md]") {
    const vector3 box = fixture_box();
    vm_float_t x[2] = { VM_F(0.0), VM_F(1.0) };
    vm_float_t y[2] = { VM_F(0.0), VM_F(0.2) };
    vm_float_t z[2] = { VM_F(0.0), VM_F(-0.2) };
    vm_float_t q[2] = { VM_F(1.0), VM_F(-0.5) };
    vm_float_t fx[2] = { VM_F(3.0), VM_F(3.0) };
    vm_float_t fy[2] = { VM_F(3.0), VM_F(3.0) };
    vm_float_t fz[2] = { VM_F(3.0), VM_F(3.0) };
    vm_float_t energy = VM_F(3.0);
    vm_float_t virial[6];
    for (int k = 0; k < 6; ++k) {
        virial[k] = VM_F(3.0);
    }
    const vector3 bad = vec3(VM_F(0.0), VM_F(1.0), VM_F(1.0));
    REQUIRE(!vm_md_coulomb_tile(fx, fy, fz, &energy, virial, x, y, z, q,
                                2, 0, 1, 1, 1, bad, VM_F(2.3), VM_F(2.0), 1));
    REQUIRE(!vm_md_coulomb_tile(fx, fy, fz, &energy, virial, x, y, z, q,
                                2, 0, 1, 1, 1, box, VM_F(-1.0), VM_F(2.0), 1));
    REQUIRE(!vm_md_coulomb_tile(fx, fy, fz, &energy, virial, x, y, z, q,
                                2, 0, 1, 1, 1, box, NAN, VM_F(2.0), 1));
    REQUIRE(!vm_md_coulomb_tile(fx, fy, fz, &energy, virial, x, y, z, q,
                                2, 0, 1, 1, 1, box, VM_F(2.3), VM_F(0.0), 1));
    REQUIRE(!vm_md_coulomb_tile(fx, fy, fz, &energy, virial, x, y, z, q,
                                -1, 0, 1, 0, 1, box, VM_F(2.3), VM_F(2.0), 1));
    REQUIRE(!vm_md_coulomb_tile(fx, fy, fz, &energy, virial, x, y, z, q,
                                2, -1, 1, 0, 1, box, VM_F(2.3), VM_F(2.0), 1));
    REQUIRE(!vm_md_coulomb_tile(fx, fy, fz, &energy, virial, x, y, z, q,
                                2, 0, 3, 0, 1, box, VM_F(2.3), VM_F(2.0), 1));
    REQUIRE(!vm_md_coulomb_tile(NULL, fy, fz, &energy, virial, x, y, z, q,
                                2, 0, 1, 1, 1, box, VM_F(2.3), VM_F(2.0), 1));
    REQUIRE(!vm_md_coulomb_tile(fx, fy, fz, &energy, virial, NULL, y, z, q,
                                2, 0, 1, 1, 1, box, VM_F(2.3), VM_F(2.0), 1));
    REQUIRE(!vm_md_coulomb_tile(fx, fy, fz, &energy, virial, x, y, z, NULL,
                                2, 0, 1, 1, 1, box, VM_F(2.3), VM_F(2.0), 1));
    REQUIRE(same_bits(fx[0], VM_F(3.0)));
    REQUIRE(same_bits(fy[1], VM_F(3.0)));
    REQUIRE(same_bits(energy, VM_F(3.0)));
    REQUIRE(same_bits(virial[5], VM_F(3.0)));

    REQUIRE(vm_md_coulomb_tile(fx, fy, fz, NULL, NULL, x, y, z, q,
                               2, 0, 1, 1, 1, box, VM_F(2.3), VM_F(2.0), 0));
}

TEST_CASE(md_coulomb_null_outputs_match_forces, "[md]") {
    const vector3 box = fixture_box();
    const vm_float_t prefactor = VM_F(2.3);
    const vm_float_t rcut = VM_F(2.5);
    vm_float_t x[MD_N], y[MD_N], z[MD_N], q[MD_N];
    fill_fixture(x, y, z, box);
    fill_charges(q);
    vm_float_t fx[MD_N], fy[MD_N], fz[MD_N], energy, virial[6];
    vm_float_t nfx[MD_N], nfy[MD_N], nfz[MD_N];
    zero_out(fx, fy, fz, &energy, virial);
    for (int i = 0; i < MD_N; ++i) {
        nfx[i] = VM_F(0.0);
        nfy[i] = VM_F(0.0);
        nfz[i] = VM_F(0.0);
    }
    REQUIRE(vm_md_coulomb_tile(fx, fy, fz, &energy, virial, x, y, z, q,
                               MD_N, 0, MD_N, 0, MD_N, box, prefactor, rcut, 1));
    REQUIRE(vm_md_coulomb_tile(nfx, nfy, nfz, NULL, NULL, x, y, z, q,
                               MD_N, 0, MD_N, 0, MD_N, box, prefactor, rcut, 1));
    for (int i = 0; i < MD_N; ++i) {
        require_bits(nfx[i], fx[i], "null fx");
        require_bits(nfy[i], fy[i], "null fy");
        require_bits(nfz[i], fz[i], "null fz");
    }
}

TEST_CASE(md_coulomb_energy_matches_compensated_sum, "[md]") {
    const vector3 box = fixture_box();
    const vm_float_t prefactor = VM_F(2.3);
    const vm_float_t rcut = VM_F(2.5);
    vm_float_t x[MD_N], y[MD_N], z[MD_N], q[MD_N];
    fill_fixture(x, y, z, box);
    fill_charges(q);
    vm_float_t fx[MD_N], fy[MD_N], fz[MD_N], energy, virial[6];
    vm_float_t terms[MD_N * MD_N];
    int nterms = 0;
    md_cref_prep prep;
    cref_prepare(&prep, prefactor, rcut);
    zero_out(fx, fy, fz, &energy, virial);
    vm_float_t rfx[MD_N], rfy[MD_N], rfz[MD_N], renergy = VM_F(0.0), rvir[6];
    zero_out(rfx, rfy, rfz, &renergy, rvir);
    REQUIRE(cref_tile(rfx, rfy, rfz, &renergy, rvir, x, y, z, q,
                      0, MD_N, 0, MD_N, box, &prep, 1, terms, MD_N * MD_N, &nterms));
    REQUIRE(nterms > 1);
    const vm_float_t oracle = vm_sum_compensated(terms, nterms);
    REQUIRE(same_bits(renergy, oracle));
    REQUIRE(vm_md_coulomb_tile(fx, fy, fz, &energy, virial, x, y, z, q,
                               MD_N, 0, MD_N, 0, MD_N, box, prefactor, rcut, 1));
    require_bits(energy, oracle, "compensated energy");
    require_match(fx, fy, fz, energy, virial, rfx, rfy, rfz, renergy, rvir, MD_N);
}

TEST_CASE(md_coulomb_simd_matches_scalar_tile_on_fixture, "[md]") {
    const vector3 box = fixture_box();
    const vm_float_t prefactor = VM_F(2.3);
    const vm_float_t rcut = VM_F(2.5);
    vm_float_t x[MD_N], y[MD_N], z[MD_N], q[MD_N];
    fill_fixture(x, y, z, box);
    fill_charges(q);

    /* Lengths 1..19 hit every AVX2 and NEON tail, including a full self-tile.
     * On this host the kernel takes the SIMD path when that ISA was compiled;
     * the reference is the scalar closed form, cutoff included. */
    for (int n = 1; n <= MD_N; ++n) {
        run_coulomb(x, y, z, q, n, 0, n, 0, n, box, prefactor, rcut, 1, VM_F(0.0));
        if (n >= 4) {
            run_coulomb(x, y, z, q, n, 0, n / 2, n / 2, n - n / 2, box, prefactor, rcut, 1,
                        VM_F(0.0));
        }
    }
}
