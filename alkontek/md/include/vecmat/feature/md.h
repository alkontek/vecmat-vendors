// SPDX-FileCopyrightText: 2025-2026 Igal Alkon
// SPDX-FileCopyrightText: 2026 ALKONTEK <git@alkontek.com>
// SPDX-License-Identifier: BSD-3-Clause

#ifndef VECMAT_FEATURE_MD_H
#define VECMAT_FEATURE_MD_H

/**
 * @file
 * @brief Cutoff neighbor lists, shifted-force pairs, and span kinetic energy.
 * @ingroup vecmat_advance_physics
 *
 * Available only when the `alkontek/md` feature module is loaded.
 * Needs the Vecmat `v0.4.x` surface (`vm_periodic_delta_ortho`,
 * `vm_sum_compensated`, `vm_cpu_has`).
 *
 * The library builds each entry point once per float ABI. The unsuffixed
 * names below are the ones callers write; the header maps them to the
 * `32` or `64` suffix.
 */

#include <vecmat.h>

#if defined(VECMAT_ABI_BITS) && (VECMAT_ABI_BITS == 64)
#  define VECMAT_MD_ABI_SUFFIX 64
#elif defined(VECMAT_ABI_BITS) && (VECMAT_ABI_BITS == 32)
#  define VECMAT_MD_ABI_SUFFIX 32
#elif defined(VECMAT_USE_F64)
#  define VECMAT_MD_ABI_SUFFIX 64
#else
#  define VECMAT_MD_ABI_SUFFIX 32
#endif

#if VECMAT_MD_ABI_SUFFIX == 64
#  define vm_md_lj_tile vm_md_lj_tile64
#  define vm_md_lj_list vm_md_lj_list64
#  define vm_md_lj_list_excl vm_md_lj_list_excl64
#  define vm_md_coulomb_tile vm_md_coulomb_tile64
#  define vm_md_coulomb_list vm_md_coulomb_list64
#  define vm_md_coulomb_list_excl vm_md_coulomb_list_excl64
#  define vm_md_neighbors vm_md_neighbors64
#  define vm_md_neighbors_scratch vm_md_neighbors_scratch64
#  define vm_md_kinetic vm_md_kinetic64
#else
#  define vm_md_lj_tile vm_md_lj_tile32
#  define vm_md_lj_list vm_md_lj_list32
#  define vm_md_lj_list_excl vm_md_lj_list_excl32
#  define vm_md_coulomb_tile vm_md_coulomb_tile32
#  define vm_md_coulomb_list vm_md_coulomb_list32
#  define vm_md_coulomb_list_excl vm_md_coulomb_list_excl32
#  define vm_md_neighbors vm_md_neighbors32
#  define vm_md_neighbors_scratch vm_md_neighbors_scratch32
#  define vm_md_kinetic vm_md_kinetic32
#endif

/**
 * @brief Add one shifted-force Lennard-Jones tile.
 *
 * Positions and forces are unit-stride SoA of length `n`. The tile is
 * the rectangle `[i0, i0+ni)` by `[j0, j0+nj)`. Scalar order is `i`, then
 * `j`. `i == j` is skipped.
 *
 * Each pair uses the orthorhombic minimum image `dx = remainder(xi - xj, Lx)`
 * (and the same for `y` and `z`): ties-to-even, result in `[-L/2, L/2]`,
 * the same rule as `vm_periodic_delta_ortho`. This function does not call
 * that routine. `r² >= rcut²` contributes nothing.
 *
 * Unshifted pair, with `r² = dx² + dy² + dz²` and `s = σ² / r²`,
 * `s6 = s³`, `s12 = s6²`:
 *
 *     U(r) = 4 ε (s12 - s6)
 *     C(r) = 24 ε (2 s12 - s6) / r²
 *     F_i  = C(r) (dx, dy, dz)
 *
 * Shifted-force subtracts the cutoff value and the cutoff derivative, so
 * both the pair potential and the pair force are zero at `rcut`. With
 * `U'(r) = -C(r) r`:
 *
 *     U_sf(r) = U(r) - U(rcut) - (r - rcut) U'(rcut)
 *             = U(r) - U(rcut) + (r - rcut) C(rcut) rcut
 *     C_sf(r) = C(r) - C(rcut) rcut / r
 *     F_i     = C_sf(r) (dx, dy, dz)
 *
 * Evaluation uses a real square root, never a reciprocal-square-root
 * estimate, and does not fuse multiply-adds:
 *
 *     r2 = (dx*dx + dy*dy) + dz*dz
 *     s = σ² / r2
 *     s6 = (s*s) * s
 *     s12 = s6 * s6
 *     U = (4ε) * (s12 - s6)
 *     C = ((24ε) * ((2*s12) - s6)) / r2
 *     r = sqrt(r2)
 *     U_sf = (U - U(rcut)) + ((r - rcut) * (C(rcut)*rcut))
 *     C_sf = C - ((C(rcut)*rcut) / r)
 *
 * `newton == 0` adds `F` on `i` only. Nonzero `newton` adds `+F` on `i`
 * and `-F` on `j`. If the two index ranges overlap, nonzero `newton` keeps
 * only `i < j`, so a self-tile is not applied twice.
 *
 * Energy is `U_sf` once per accepted pair. Accepted energies are summed
 * with the Neumaier recurrence in `(i, j)` order, then added to `*energy`.
 * `vm_sum_compensated` on those pair terms is the oracle (a caller-supplied
 * `+0` plus that sum).
 *
 * Virial is six components from the same minimum-image vector and the force
 * added to `i`, once per accepted pair. Repulsion (`F` along `d = ri - rj`)
 * makes `dx*Fx` positive:
 *
 *     [0] xx += dx*Fx
 *     [1] yy += dy*Fy
 *     [2] zz += dz*Fz
 *     [3] xy += dx*Fy
 *     [4] xz += dx*Fz
 *     [5] yz += dy*Fz
 *
 * The caller zeroes forces, energy, and virial before the first tile. The
 * tile adds. `energy` or `virial` may be NULL. `epsilon`, `sigma`, `rcut`,
 * and the three box lengths must be finite and positive.
 *
 * `ni == 0` or `nj == 0` with otherwise valid arguments returns true and
 * writes nothing (the arrays are not read). Out-of-range indexes, a
 * non-positive parameter, or a missing position or force pointer on a
 * non-empty tile returns false and writes nothing.
 *
 * @param fx,fy,fz Forces, length `n`. Required when the tile is non-empty.
 * @param energy   Pair-energy accumulator, or NULL.
 * @param virial   Six components `xx,yy,zz,xy,xz,yz`, or NULL.
 * @param x,y,z    Positions, length `n`. Required when the tile is non-empty.
 * @param n        SoA length.
 * @param i0,ni    Row range `[i0, i0+ni)`.
 * @param j0,nj    Column range `[j0, j0+nj)`.
 * @param box      Positive finite orthorhombic lengths.
 * @param epsilon  Positive finite well depth.
 * @param sigma    Positive finite diameter.
 * @param rcut     Positive finite cutoff.
 * @param newton   Nonzero: action-reaction, and `i < j` when the ranges overlap.
 * @return false when the arguments are rejected.
 */
VEC_API bool vm_md_lj_tile(
    vm_float_t *fx, vm_float_t *fy, vm_float_t *fz,
    vm_float_t *energy, vm_float_t virial[6],
    const vm_float_t *x, const vm_float_t *y, const vm_float_t *z,
    int n, int i0, int ni, int j0, int nj,
    vector3 box, vm_float_t epsilon, vm_float_t sigma, vm_float_t rcut,
    int newton);

/**
 * @brief Add shifted-force Lennard-Jones pairs from a neighbor list.
 *
 * Positions and forces are unit-stride SoA of length `n`. Row `k` is
 * particle `i0 + k`. `offsets` has length `ni + 1`, with `offsets[0] == 0`,
 * and `neighbors[offsets[k] .. offsets[k+1])` are absolute `j` indexes.
 * That is the CSR from `vm_md_neighbors`. Walk order is the row order, then
 * the stored neighbor order.
 *
 * Each stored neighbor uses the same minimum image, cutoff, shifted-force
 * energy, and virial as one pair of `vm_md_lj_tile`. The walk calls that
 * pair step. `r² >= rcut²` contributes nothing. `j == i` is skipped.
 *
 * The list stores both directions when both indexes lie in the rectangle,
 * so this function adds `F` on `i` only. It does not write the reaction
 * on `j`.
 *
 * Energy is `U_sf` once per accepted stored pair, summed with the Neumaier
 * recurrence in walk order, then added to `*energy`. Virial is the same
 * six components as `vm_md_lj_tile`, once per accepted stored pair. A list
 * that stores both directions therefore matches the forces of a Newton
 * self-tile, and the energy and virial of that rectangle with `newton == 0`.
 *
 * The caller zeroes forces, energy, and virial before the first call. The
 * call adds. `energy` or `virial` may be NULL. `epsilon`, `sigma`, `rcut`,
 * and the three box lengths must be finite and positive.
 *
 * `ni == 0`, or a CSR with no stored neighbors, returns true and writes
 * nothing. Offsets are not read when `ni == 0`. Neighbors, positions, and
 * forces are not read when no neighbor is stored. A decreasing offset, a
 * neighbor outside `[0, n)`, a non-positive parameter, a missing offsets
 * pointer, or a missing position or force pointer on a non-empty list
 * returns false and writes nothing.
 *
 * @param fx,fy,fz Forces, length `n`. Required when a neighbor is stored.
 * @param energy   Pair-energy accumulator, or NULL.
 * @param virial   Six components `xx,yy,zz,xy,xz,yz`, or NULL.
 * @param x,y,z    Positions, length `n`. Required when a neighbor is stored.
 * @param n        SoA length.
 * @param i0,ni    Row range `[i0, i0+ni)`.
 * @param offsets  CSR row pointers, length `ni + 1`. Ignored when `ni == 0`.
 * @param neighbors Packed `j` indexes. NULL only when no neighbor is stored.
 * @param box      Positive finite orthorhombic lengths.
 * @param epsilon  Positive finite well depth.
 * @param sigma    Positive finite diameter.
 * @param rcut     Positive finite cutoff.
 * @return false when the arguments are rejected.
 */
VEC_API bool vm_md_lj_list(
    vm_float_t *fx, vm_float_t *fy, vm_float_t *fz,
    vm_float_t *energy, vm_float_t virial[6],
    const vm_float_t *x, const vm_float_t *y, const vm_float_t *z,
    int n, int i0, int ni,
    const int *offsets, const int *neighbors,
    vector3 box, vm_float_t epsilon, vm_float_t sigma, vm_float_t rcut);

/**
 * @brief `vm_md_lj_list` with caller-supplied exclusions.
 *
 * An exclusion is an unordered pair of particle indexes, stored as a CSR
 * with the same shape as the neighbor list. `excl_offsets` has length
 * `ni + 1`, with `excl_offsets[0] == 0`, and
 * `excl_partners[excl_offsets[k] .. excl_offsets[k+1])` are the absolute
 * indexes excluded with particle `i0 + k`. Partner order does not matter.
 * The rows are not a topology graph.
 *
 * A stored neighbor `j` is skipped when `j` appears on that row. The
 * partner's row is not consulted, so a both-directions list skips each
 * direction only from its own row. The caller lists both endpoints.
 * A skipped pair adds no force, energy, or virial. `j == i` is still
 * skipped before the exclusion test. Pairs that remain are visited in
 * neighbor-list order, with the same pair step as `vm_md_lj_list`.
 *
 * `excl_offsets == NULL` means there are no exclusions, and
 * `excl_partners` is not read. An empty CSR is the same result. Either
 * one matches `vm_md_lj_list` on the same neighbor list. A decreasing
 * offset, a partner outside `[0, n)`, or a missing partner pointer when
 * a partner is stored returns false and writes nothing. `ni == 0` does
 * not read the exclusion CSR. An empty neighbor list still writes
 * nothing; a rejected exclusion CSR is rejected first.
 *
 * @param excl_offsets  Exclusion CSR, length `ni + 1`, or NULL.
 * @param excl_partners Packed partner indexes. NULL when no partner is stored.
 * @return false when the arguments are rejected.
 */
VEC_API bool vm_md_lj_list_excl(
    vm_float_t *fx, vm_float_t *fy, vm_float_t *fz,
    vm_float_t *energy, vm_float_t virial[6],
    const vm_float_t *x, const vm_float_t *y, const vm_float_t *z,
    int n, int i0, int ni,
    const int *offsets, const int *neighbors,
    vector3 box, vm_float_t epsilon, vm_float_t sigma, vm_float_t rcut,
    const int *excl_offsets, const int *excl_partners);

/**
 * @brief Add one shifted-force Coulomb tile.
 *
 * Positions, charges, and forces are unit-stride SoA of length `n`. The
 * tile is the rectangle `[i0, i0+ni)` by `[j0, j0+nj)`. Scalar order is
 * `i`, then `j`. `i == j` is skipped. `prefactor` is the Coulomb scale in
 * the caller's units (`U = prefactor * qi * qj / r` before the shift). It
 * is not a hardcoded constant.
 *
 * Each pair uses the orthorhombic minimum image `dx = remainder(xi - xj, Lx)`
 * (and the same for `y` and `z`): ties-to-even, result in `[-L/2, L/2]`,
 * the same rule as `vm_periodic_delta_ortho`. This function does not call
 * that routine. `r² >= rcut²` contributes nothing.
 *
 * Unshifted pair, with `r² = dx² + dy² + dz²` and `d = (dx, dy, dz)`:
 *
 *     scale = prefactor * (qi * qj)
 *     U(r)  = scale / r
 *     C(r)  = scale / r³
 *     F_i   = C(r) d
 *
 * Shifted-force subtracts the cutoff value and the cutoff derivative, so
 * both the pair potential and the pair force are zero at `rcut`. With
 * `U'(r) = -C(r) r`:
 *
 *     U_sf(r) = U(r) - U(rcut) - (r - rcut) U'(rcut)
 *             = U(r) - U(rcut) + (r - rcut) C(rcut) rcut
 *     C_sf(r) = C(r) - C(rcut) rcut / r
 *     F_i     = C_sf(r) d
 *
 * Evaluation uses a real square root, never a reciprocal-square-root
 * estimate, and does not fuse multiply-adds:
 *
 *     r2 = (dx*dx + dy*dy) + dz*dz
 *     scale = prefactor * (qi * qj)
 *     r = sqrt(r2)
 *     u = scale / r
 *     u_c = scale / rcut
 *     duc = u_c / rcut
 *     c = (scale / r2) / r
 *     U_sf = (u - u_c) + ((r - rcut) * duc)
 *     C_sf = c - (duc / r)
 *
 * `newton == 0` adds `F` on `i` only. Nonzero `newton` adds `+F` on `i`
 * and `-F` on `j`. If the two index ranges overlap, nonzero `newton` keeps
 * only `i < j`, so a self-tile is not applied twice.
 *
 * Energy is `U_sf` once per accepted pair. Accepted energies are summed
 * with the Neumaier recurrence in `(i, j)` order, then added to `*energy`.
 * `vm_sum_compensated` on those pair terms is the oracle (a caller-supplied
 * `+0` plus that sum). A zero charge product inside the cutoff is an
 * accepted pair, not a skip.
 *
 * Virial is six components from the same minimum-image vector and the force
 * added to `i`, once per accepted pair. The sign matches `vm_md_lj_tile`:
 * repulsion (`F` along `d = ri - rj`, like-signed charges) makes `dx*Fx`
 * positive:
 *
 *     [0] xx += dx*Fx
 *     [1] yy += dy*Fy
 *     [2] zz += dz*Fz
 *     [3] xy += dx*Fy
 *     [4] xz += dx*Fz
 *     [5] yz += dy*Fz
 *
 * The caller zeroes forces, energy, and virial before the first tile. The
 * tile adds. `energy` or `virial` may be NULL. `prefactor`, `rcut`, and the
 * three box lengths must be finite and positive. Charges may be negative
 * or zero.
 *
 * `ni == 0` or `nj == 0` with otherwise valid arguments returns true and
 * writes nothing (the arrays are not read). Out-of-range indexes, a
 * non-positive parameter, or a missing position, charge, or force pointer
 * on a non-empty tile returns false and writes nothing.
 *
 * @param fx,fy,fz Forces, length `n`. Required when the tile is non-empty.
 * @param energy   Pair-energy accumulator, or NULL.
 * @param virial   Six components `xx,yy,zz,xy,xz,yz`, or NULL.
 * @param x,y,z    Positions, length `n`. Required when the tile is non-empty.
 * @param q        Charges, length `n`. Required when the tile is non-empty.
 * @param n        SoA length.
 * @param i0,ni    Row range `[i0, i0+ni)`.
 * @param j0,nj    Column range `[j0, j0+nj)`.
 * @param box      Positive finite orthorhombic lengths.
 * @param prefactor Positive finite Coulomb scale, in the caller's units.
 * @param rcut     Positive finite cutoff.
 * @param newton   Nonzero: action-reaction, and `i < j` when the ranges overlap.
 * @return false when the arguments are rejected.
 */
VEC_API bool vm_md_coulomb_tile(
    vm_float_t *fx, vm_float_t *fy, vm_float_t *fz,
    vm_float_t *energy, vm_float_t virial[6],
    const vm_float_t *x, const vm_float_t *y, const vm_float_t *z,
    const vm_float_t *q,
    int n, int i0, int ni, int j0, int nj,
    vector3 box, vm_float_t prefactor, vm_float_t rcut,
    int newton);

/**
 * @brief Add shifted-force Coulomb pairs from a neighbor list.
 *
 * Positions, charges, and forces are unit-stride SoA of length `n`. Row
 * `k` is particle `i0 + k`. `offsets` has length `ni + 1`, with
 * `offsets[0] == 0`, and `neighbors[offsets[k] .. offsets[k+1])` are
 * absolute `j` indexes. That is the CSR from `vm_md_neighbors`. Walk
 * order is the row order, then the stored neighbor order.
 *
 * Each stored neighbor uses the same minimum image, cutoff, shifted-force
 * energy, and virial as one pair of `vm_md_coulomb_tile`. `prefactor` is
 * the caller's Coulomb scale. The walk calls that pair step.
 * `r² >= rcut²` contributes nothing. `j == i` is skipped. A zero charge
 * product inside the cutoff is an accepted pair, not a skip.
 *
 * The list stores both directions when both indexes lie in the rectangle,
 * so this function adds `F` on `i` only. It does not write the reaction
 * on `j`.
 *
 * Energy is `U_sf` once per accepted stored pair, summed with the Neumaier
 * recurrence in walk order, then added to `*energy`. Virial is the same
 * six components as `vm_md_coulomb_tile`, once per accepted stored pair. A
 * list that stores both directions therefore matches the forces of a
 * Newton self-tile, and the energy and virial of that rectangle with
 * `newton == 0`.
 *
 * The caller zeroes forces, energy, and virial before the first call. The
 * call adds. `energy` or `virial` may be NULL. `prefactor`, `rcut`, and
 * the three box lengths must be finite and positive. Charges may be
 * negative or zero.
 *
 * `ni == 0`, or a CSR with no stored neighbors, returns true and writes
 * nothing. Offsets are not read when `ni == 0`. Neighbors, positions,
 * charges, and forces are not read when no neighbor is stored. A
 * decreasing offset, a neighbor outside `[0, n)`, a non-positive
 * parameter, a missing offsets pointer, or a missing position, charge, or
 * force pointer on a non-empty list returns false and writes nothing.
 *
 * @param fx,fy,fz Forces, length `n`. Required when a neighbor is stored.
 * @param energy   Pair-energy accumulator, or NULL.
 * @param virial   Six components `xx,yy,zz,xy,xz,yz`, or NULL.
 * @param x,y,z    Positions, length `n`. Required when a neighbor is stored.
 * @param q        Charges, length `n`. Required when a neighbor is stored.
 * @param n        SoA length.
 * @param i0,ni    Row range `[i0, i0+ni)`.
 * @param offsets  CSR row pointers, length `ni + 1`. Ignored when `ni == 0`.
 * @param neighbors Packed `j` indexes. NULL only when no neighbor is stored.
 * @param box      Positive finite orthorhombic lengths.
 * @param prefactor Positive finite Coulomb scale, in the caller's units.
 * @param rcut     Positive finite cutoff.
 * @return false when the arguments are rejected.
 */
VEC_API bool vm_md_coulomb_list(
    vm_float_t *fx, vm_float_t *fy, vm_float_t *fz,
    vm_float_t *energy, vm_float_t virial[6],
    const vm_float_t *x, const vm_float_t *y, const vm_float_t *z,
    const vm_float_t *q,
    int n, int i0, int ni,
    const int *offsets, const int *neighbors,
    vector3 box, vm_float_t prefactor, vm_float_t rcut);

/**
 * @brief `vm_md_coulomb_list` with caller-supplied exclusions.
 *
 * The exclusion CSR is the same one `vm_md_lj_list_excl` uses: one row per
 * walked particle, listing the absolute indexes excluded with it. Partner
 * order does not matter. A stored neighbor `j` is skipped when `j` appears
 * on that row. The partner's row is not consulted. A skipped pair adds no
 * force, energy, or virial. `j == i` is still skipped first. Pairs that
 * remain stay in neighbor-list order and use the Coulomb pair step.
 *
 * `excl_offsets == NULL`, and an empty CSR, match `vm_md_coulomb_list`
 * on the same neighbor list. `excl_partners` is not read in the NULL case,
 * and it may be NULL when no partner is stored. A bad exclusion CSR
 * returns false and writes nothing. `ni == 0` does not read it.
 *
 * @param excl_offsets  Exclusion CSR, length `ni + 1`, or NULL.
 * @param excl_partners Packed partner indexes. NULL when no partner is stored.
 * @return false when the arguments are rejected.
 */
VEC_API bool vm_md_coulomb_list_excl(
    vm_float_t *fx, vm_float_t *fy, vm_float_t *fz,
    vm_float_t *energy, vm_float_t virial[6],
    const vm_float_t *x, const vm_float_t *y, const vm_float_t *z,
    const vm_float_t *q,
    int n, int i0, int ni,
    const int *offsets, const int *neighbors,
    vector3 box, vm_float_t prefactor, vm_float_t rcut,
    const int *excl_offsets, const int *excl_partners);

/**
 * @brief Scratch ints for one `vm_md_neighbors` call.
 *
 * Returns 0 when `ni == 0` or `nj == 0` and the other arguments are
 * accepted: the list call writes nothing and does not read scratch.
 * Returns -1 when `n < 0`, a range is out of bounds, or `rcut` or a box
 * length is not finite and positive. Also returns -1 when the cell grid
 * does not fit in `int`.
 *
 * @param n     SoA length.
 * @param i0,ni Row range `[i0, i0+ni)`.
 * @param j0,nj Column range `[j0, j0+nj)`.
 * @param box   Positive finite orthorhombic lengths.
 * @param rcut  Positive finite cutoff.
 * @return Scratch length in ints, 0 for an empty range, or -1.
 */
VEC_API int vm_md_neighbors_scratch(
    int n, int i0, int ni, int j0, int nj,
    vector3 box, vm_float_t rcut);

/**
 * @brief Cutoff neighbor list on an orthorhombic box.
 *
 * Positions are unit-stride SoA of length `n`. The list is the rectangle
 * `[i0, i0+ni)` by `[j0, j0+nj)`. Row `k` is particle `i0 + k`. `offsets`
 * has length `ni + 1`, and `neighbors[offsets[k] .. offsets[k+1])` are the
 * absolute indexes `j`.
 *
 * Binning folds each coordinate with the same rule as `vm_wrap_ortho`
 * (`remainder`, then into `[0, L)`, with `±0` and `L` stored as `+0`).
 * Positions are not written. Each axis is split into equal cells that tile
 * the box. The cell count is the largest integer whose edge `L/count`
 * is at least `rcut`, or one cell when `L < rcut`. The edge is the
 * cutoff when the cutoff divides the axis. A pair inside the cutoff
 * shares a cell or a periodic neighbor of it, so the search is that
 * 27-cell stencil, and each cell is visited once.
 *
 * A stored pair has `j` in the column range, `j != i`, and minimum-image
 * `r² < rcut²`. Each axis is `remainder`, ties-to-even, in `[-L/2, L/2]`,
 * the same rule as `vm_periodic_delta_ortho` and the pair tiles. This
 * function does not call that routine. `r² >= rcut²` is omitted, including
 * an exact cutoff. Both directions are stored when both indexes lie in
 * both ranges. This is not a pair-force kernel.
 *
 * Rows follow increasing `i`. Inside a row, cells run over `dz`, then `dy`,
 * then `dx`, each from -1 to 1. Particles in a cell are in increasing
 * index order.
 *
 * The caller owns every buffer. `scratch` is opaque work space of length
 * `vm_md_neighbors_scratch`, or longer. `capacity` is the length of
 * `neighbors`.
 *
 * `ni == 0` or `nj == 0` with otherwise valid arguments returns true and
 * writes nothing. Positions, scratch, offsets, and neighbors are not read.
 * `n < 0`, an out-of-range index, a non-positive box or cutoff, a missing
 * offsets or position pointer on a non-empty list, or a short scratch
 * buffer returns false and writes nothing. A neighbor buffer shorter than
 * the pair count returns false and does not write `offsets` or `neighbors`.
 * Scratch may already have been used in that case. `neighbors` may be NULL
 * when the pair count is zero.
 *
 * @param offsets    CSR row pointers, length `ni + 1`.
 * @param neighbors  Packed column indexes. NULL only when no pair is stored.
 * @param capacity   Length of `neighbors`.
 * @param scratch    Work ints. Ignored for an empty range.
 * @param scratch_n  Length of `scratch`.
 * @param x,y,z      Positions, length `n`.
 * @param n          SoA length.
 * @param i0,ni      Row range `[i0, i0+ni)`.
 * @param j0,nj      Column range `[j0, j0+nj)`.
 * @param box        Positive finite orthorhombic lengths.
 * @param rcut       Positive finite cutoff.
 * @return false when the arguments are rejected or `neighbors` is short.
 */
VEC_API bool vm_md_neighbors(
    int *offsets, int *neighbors, int capacity,
    int *scratch, int scratch_n,
    const vm_float_t *x, const vm_float_t *y, const vm_float_t *z,
    int n, int i0, int ni, int j0, int nj,
    vector3 box, vm_float_t rcut);

/**
 * @brief Kinetic energy of one particle span.
 *
 * Velocities and masses are unit-stride SoA of length `n`. The span is
 * `[i0, i0+ni)`. Particle `i` contributes `m |v|² / 2`. The sum is not
 * added to the previous value of `*energy`.
 *
 * Each term is formed without a fused multiply-add, then the `ni` terms
 * are reduced with `vm_sum_compensated` (index order, particle `i0` first):
 *
 *     v2   = (vx*vx + vy*vy) + vz*vz
 *     term = (m * v2) * 0.5
 *
 * There is no local Neumaier step and no SIMD path. The core sum is the
 * reduction. The term buffer is allocated for the call and freed before
 * return. Masses and velocities are not checked for sign or finiteness.
 *
 * `ni == 0` with an in-range span writes `+0` and does not read velocities
 * or masses. `n == 0` is that empty span when `i0 == 0`. A negative count,
 * an out-of-range index, a missing `energy` pointer, a missing velocity or
 * mass pointer on a non-empty span, or a term buffer that cannot be
 * allocated returns false and writes nothing.
 *
 * @param energy Result. Required. Unchanged when the call is rejected.
 * @param vx,vy,vz Velocities, length `n`. Ignored when `ni == 0`.
 * @param mass   Per-particle masses, length `n`. Ignored when `ni == 0`.
 * @param n      SoA length.
 * @param i0,ni  Span `[i0, i0+ni)`.
 * @return false when the arguments are rejected or the terms cannot be stored.
 */
VEC_API bool vm_md_kinetic(
    vm_float_t *energy,
    const vm_float_t *vx, const vm_float_t *vy, const vm_float_t *vz,
    const vm_float_t *mass,
    int n, int i0, int ni);

#endif /* VECMAT_FEATURE_MD_H */
