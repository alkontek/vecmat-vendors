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

static int same_bits(const vm_float_t a, const vm_float_t b)
{
    return memcmp(&a, &b, sizeof a) == 0;
}

/* Same parenthesization as alkontek/md/src/kinetic.c. */
static vm_float_t kinetic_term(const vm_float_t vx, const vm_float_t vy,
                               const vm_float_t vz, const vm_float_t mass)
{
    const vm_float_t v2 = (vx * vx + vy * vy) + vz * vz;
    return (mass * v2) * VM_F(0.5);
}

TEST_CASE(md_kinetic_hand_sum, "[md]") {
    /* Span [1, 4) of a length-5 SoA. Outside particles are NaN and must
     * not enter the sum.
     *   i=1  m=4  v=(1, 2, 2)  |v|^2=9   term=(4*9)*0.5=18
     *   i=2  m=1  v=(0, 0, 2)  |v|^2=4   term=(1*4)*0.5=2
     *   i=3  m=2  v=(3, 0, 0)  |v|^2=9   term=(2*9)*0.5=9
     *   18+2+9 = 29
     */
    vm_float_t vx[5], vy[5], vz[5], mass[5];
    vm_float_t energy = VM_F(8.0);
    const vm_float_t t1 = (VM_F(4.0) * VM_F(9.0)) * VM_F(0.5);
    const vm_float_t t2 = (VM_F(1.0) * VM_F(4.0)) * VM_F(0.5);
    const vm_float_t t3 = (VM_F(2.0) * VM_F(9.0)) * VM_F(0.5);
    const vm_float_t hand = (t1 + t2) + t3;

    for (int i = 0; i < 5; ++i) {
        vx[i] = (vm_float_t)NAN;
        vy[i] = (vm_float_t)NAN;
        vz[i] = (vm_float_t)NAN;
        mass[i] = VM_F(1.0e30);
    }
    vx[1] = VM_F(1.0);
    vy[1] = VM_F(2.0);
    vz[1] = VM_F(2.0);
    mass[1] = VM_F(4.0);
    vx[2] = VM_F(0.0);
    vy[2] = VM_F(0.0);
    vz[2] = VM_F(2.0);
    mass[2] = VM_F(1.0);
    vx[3] = VM_F(3.0);
    vy[3] = VM_F(0.0);
    vz[3] = VM_F(0.0);
    mass[3] = VM_F(2.0);

    REQUIRE(same_bits(hand, VM_F(29.0)));
    REQUIRE(vm_md_kinetic(&energy, vx, vy, vz, mass, 5, 1, 3));
    REQUIRE(same_bits(energy, hand));
    REQUIRE(same_bits(energy, VM_F(29.0)));
}

TEST_CASE(md_kinetic_matches_compensated_sum, "[md]") {
    /* terms are 2^60, 1, and -2^60. A left-to-right sum is 0 in f32 and
     * f64. The compensated sum keeps 1. One extra incommensurate particle
     * locks the |v|^2 parenthesization. */
    vm_float_t vx[4], vy[4], vz[4], mass[4], terms[4];
    vm_float_t energy = VM_F(-3.0);
    vm_float_t naive3;

    vx[0] = VM_F(0x1p30);
    vy[0] = VM_F(0.0);
    vz[0] = VM_F(0.0);
    mass[0] = VM_F(2.0);
    vx[1] = VM_F(1.0);
    vy[1] = VM_F(0.0);
    vz[1] = VM_F(0.0);
    mass[1] = VM_F(2.0);
    vx[2] = VM_F(0x1p30);
    vy[2] = VM_F(0.0);
    vz[2] = VM_F(0.0);
    mass[2] = VM_F(-2.0);
    vx[3] = VM_F(0.3);
    vy[3] = VM_F(-0.7);
    vz[3] = VM_F(1.1);
    mass[3] = VM_F(1.25);

    for (int i = 0; i < 4; ++i) {
        terms[i] = kinetic_term(vx[i], vy[i], vz[i], mass[i]);
    }
    naive3 = (terms[0] + terms[1]) + terms[2];
    REQUIRE(same_bits(terms[0], VM_F(0x1p60)));
    REQUIRE(same_bits(terms[1], VM_F(1.0)));
    REQUIRE(same_bits(terms[2], VM_F(-0x1p60)));
    REQUIRE(same_bits(naive3, VM_F(0.0)));

    REQUIRE(vm_md_kinetic(&energy, vx, vy, vz, mass, 4, 0, 4));
    REQUIRE(same_bits(energy, vm_sum_compensated(terms, 4)));

    energy = VM_F(5.0);
    REQUIRE(vm_md_kinetic(&energy, vx, vy, vz, mass, 4, 0, 3));
    REQUIRE(same_bits(energy, vm_sum_compensated(terms, 3)));
    REQUIRE(same_bits(energy, VM_F(1.0)));
    REQUIRE(!same_bits(energy, naive3));

    energy = VM_F(5.0);
    REQUIRE(vm_md_kinetic(&energy, vx, vy, vz, mass, 4, 1, 3));
    REQUIRE(same_bits(energy, vm_sum_compensated(terms + 1, 3)));
}

TEST_CASE(md_kinetic_empty_span_is_zero, "[md]") {
    vm_float_t energy = VM_F(-0.0);
    vm_float_t vx = VM_F(9.0);
    vm_float_t vy = VM_F(9.0);
    vm_float_t vz = VM_F(9.0);
    vm_float_t mass = VM_F(9.0);

    REQUIRE(vm_md_kinetic(&energy, NULL, NULL, NULL, NULL, 4, 0, 0));
    REQUIRE(same_bits(energy, VM_F(0.0)));
    REQUIRE(!signbit(energy));

    energy = VM_F(4.0);
    REQUIRE(vm_md_kinetic(&energy, &vx, &vy, &vz, &mass, 4, 4, 0));
    REQUIRE(same_bits(energy, VM_F(0.0)));
    REQUIRE(same_bits(vx, VM_F(9.0)));

    energy = VM_F(4.0);
    REQUIRE(vm_md_kinetic(&energy, NULL, NULL, NULL, NULL, 0, 0, 0));
    REQUIRE(same_bits(energy, VM_F(0.0)));
}

TEST_CASE(md_kinetic_rejects_bad_arguments_without_writes, "[md]") {
    vm_float_t vx[2] = { VM_F(1.0), VM_F(0.0) };
    vm_float_t vy[2] = { VM_F(0.0), VM_F(1.0) };
    vm_float_t vz[2] = { VM_F(0.0), VM_F(0.0) };
    vm_float_t mass[2] = { VM_F(2.0), VM_F(2.0) };
    vm_float_t energy = VM_F(3.0);

    REQUIRE(!vm_md_kinetic(NULL, vx, vy, vz, mass, 2, 0, 2));
    REQUIRE(!vm_md_kinetic(&energy, NULL, vy, vz, mass, 2, 0, 1));
    REQUIRE(!vm_md_kinetic(&energy, vx, NULL, vz, mass, 2, 0, 1));
    REQUIRE(!vm_md_kinetic(&energy, vx, vy, NULL, mass, 2, 0, 1));
    REQUIRE(!vm_md_kinetic(&energy, vx, vy, vz, NULL, 2, 0, 1));
    REQUIRE(!vm_md_kinetic(&energy, vx, vy, vz, mass, -1, 0, 0));
    REQUIRE(!vm_md_kinetic(&energy, vx, vy, vz, mass, 2, -1, 1));
    REQUIRE(!vm_md_kinetic(&energy, vx, vy, vz, mass, 2, 0, -1));
    REQUIRE(!vm_md_kinetic(&energy, vx, vy, vz, mass, 2, 1, 2));
    REQUIRE(!vm_md_kinetic(&energy, NULL, NULL, NULL, NULL, 2, 3, 0));
    REQUIRE(!vm_md_kinetic(&energy, vx, vy, vz, mass, 0, 0, 1));
    REQUIRE(same_bits(energy, VM_F(3.0)));
    REQUIRE(same_bits(vx[0], VM_F(1.0)));
    REQUIRE(same_bits(mass[1], VM_F(2.0)));
}
