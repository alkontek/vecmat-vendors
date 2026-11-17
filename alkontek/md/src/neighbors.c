// SPDX-FileCopyrightText: 2025-2026 Igal Alkon
// SPDX-FileCopyrightText: 2026 ALKONTEK <git@alkontek.com>
// SPDX-License-Identifier: BSD-3-Clause

#include <limits.h>
#include <math.h>
#include <stddef.h>

#include <vecmat/feature/md.h>

#if !defined(_MSC_VER)
#pragma STDC FP_CONTRACT OFF
#endif

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC optimize("fp-contract=off")
#endif

/**
 * @brief Uniform cell counts for one accepted box.
 *
 * `ncells` is `ncx * ncy * ncz`.
 */
typedef struct md_grid {
    int ncx;
    int ncy;
    int ncz;
    int ncells;
} md_grid;

static int positive(const vm_float_t v)
{
    return isfinite(v) && v > VM_F(0.0);
}

static int range_ok(const int n, const int i0, const int count)
{
    if (n < 0 || i0 < 0 || count < 0 || i0 > n) {
        return 0;
    }
    return count <= n - i0;
}

/**
 * @brief Largest equal-cell count whose edge is at least `rcut`.
 *
 * One cell when the axis is shorter than the cutoff. `-1` when the count
 * does not fit in `int` or is not an exact `vm_float_t`.
 */
static int axis_cells(const vm_float_t length, const vm_float_t rcut)
{
    const vm_float_t q = length / rcut;
    int n;

    if (!(q > VM_F(0.0))) {
        return -1;
    }
    if (q < VM_F(1.0)) {
        n = 1;
    } else if (q > (vm_float_t)INT_MAX) {
        return -1;
    } else {
        n = (int)VECMAT_FLOOR(q);
        if (n < 1) {
            n = 1;
        }
    }

    while (n > 1 && (length / (vm_float_t)n) < rcut) {
        n -= 1;
    }
    if ((int)(vm_float_t)n != n) {
        return -1;
    }
    return n;
}

static int make_grid(md_grid *g, const vector3 box, const vm_float_t rcut)
{
    const int ncx = axis_cells(box.x, rcut);
    const int ncy = axis_cells(box.y, rcut);
    const int ncz = axis_cells(box.z, rcut);
    int nxy;

    if (ncx < 1 || ncy < 1 || ncz < 1) {
        return -1;
    }
    if (ncy > INT_MAX / ncx) {
        return -1;
    }
    nxy = ncx * ncy;
    if (ncz > INT_MAX / nxy) {
        return -1;
    }

    g->ncx = ncx;
    g->ncy = ncy;
    g->ncz = ncz;
    g->ncells = nxy * ncz;
    return g->ncells;
}

/**
 * @brief `2*n + ncells + 1 + ni` ints: cell id, cell particles, cell CSR, row counts.
 */
static int scratch_count(const int n, const int ni, const int ncells)
{
    int two_n;
    int cells;

    if (n < 0 || ni < 0 || ncells < 1) {
        return -1;
    }
    if (n > INT_MAX / 2) {
        return -1;
    }
    two_n = n * 2;
    if (ncells > INT_MAX - 1 - two_n) {
        return -1;
    }
    cells = two_n + ncells + 1;
    if (ni > INT_MAX - cells) {
        return -1;
    }
    return cells + ni;
}

/**
 * @brief Validate, then the scratch length. `0` is an empty range.
 */
static int prepare(md_grid *g, const int n, const int i0, const int ni,
                   const int j0, const int nj, const vector3 box,
                   const vm_float_t rcut)
{
    if (!range_ok(n, i0, ni) || !range_ok(n, j0, nj)) {
        return -1;
    }
    if (!positive(rcut) || !positive(box.x) || !positive(box.y) || !positive(box.z)) {
        return -1;
    }
    if (ni == 0 || nj == 0) {
        return 0;
    }
    if (make_grid(g, box, rcut) < 0) {
        return -1;
    }
    return scratch_count(n, ni, g->ncells);
}

/**
 * @brief Fold one coordinate into `[0, L)`, matching `vm_wrap_ortho`.
 *
 * Core wraps an AoS buffer in place. There is no SoA form, and the caller's
 * positions stay unchanged.
 */
static vm_float_t wrap_axis(const vm_float_t v, const vm_float_t length)
{
    const vm_float_t r0 = VECMAT_REMAINDER(v, length);
    vm_float_t r;

    if (isnan(r0)) {
        return r0;
    }
    r = r0;
    if (r < VM_F(0.0)) {
        r += length;
    }
    if (!(r > VM_F(0.0)) || !(r < length)) {
        return VM_F(0.0);
    }
    return r;
}

static int axis_bin(const vm_float_t wrapped, const vm_float_t length, const int naxis)
{
    vm_float_t scaled;
    int c;

    if (naxis <= 1) {
        return 0;
    }
    scaled = (wrapped / length) * (vm_float_t)naxis;
    c = (int)VECMAT_FLOOR(scaled);
    if (c < 0) {
        c = 0;
    }
    if (c >= naxis) {
        c = naxis - 1;
    }
    return c;
}

static int locate(const vm_float_t *x, const vm_float_t *y, const vm_float_t *z,
                  const int i, const vector3 box, const md_grid *g)
{
    const vm_float_t wx = wrap_axis(x[i], box.x);
    const vm_float_t wy = wrap_axis(y[i], box.y);
    const vm_float_t wz = wrap_axis(z[i], box.z);
    int cx, cy, cz;

    if (!isfinite(wx) || !isfinite(wy) || !isfinite(wz)) {
        return -1;
    }
    cx = axis_bin(wx, box.x, g->ncx);
    cy = axis_bin(wy, box.y, g->ncy);
    cz = axis_bin(wz, box.z, g->ncz);
    return (cz * g->ncy + cy) * g->ncx + cx;
}

static int wrap_cell(const int cell, const int delta, const int naxis)
{
    int c = cell + delta;
    if (c >= naxis) {
        c -= naxis;
    } else if (c < 0) {
        c += naxis;
    }
    return c;
}

/**
 * @brief Neighbors of `i` in `[j0, j1)`. `dst` NULL counts without storing.
 */
static int row_neighbors(int *dst, const int dst_n, const int i, const int j0,
                         const int j1, const int cell, const md_grid *g,
                         const int *cell_offsets, const int *cell_particles,
                         const vm_float_t *x, const vm_float_t *y, const vm_float_t *z,
                         const vector3 box, const vm_float_t rcut2)
{
    const int cx = cell % g->ncx;
    const int cy = (cell / g->ncx) % g->ncy;
    const int cz = cell / (g->ncx * g->ncy);
    int seen[27];
    int nseen = 0;
    int count = 0;
    int sz;

    for (sz = -1; sz <= 1; ++sz) {
        const int gz = wrap_cell(cz, sz, g->ncz);
        int sy;
        for (sy = -1; sy <= 1; ++sy) {
            const int gy = wrap_cell(cy, sy, g->ncy);
            int sx;
            for (sx = -1; sx <= 1; ++sx) {
                const int gx = wrap_cell(cx, sx, g->ncx);
                const int id = (gz * g->ncy + gy) * g->ncx + gx;
                int dup = 0;
                int s;
                int p;

                for (s = 0; s < nseen; ++s) {
                    if (seen[s] == id) {
                        dup = 1;
                        break;
                    }
                }
                if (dup) {
                    continue;
                }
                seen[nseen++] = id;

                for (p = cell_offsets[id]; p < cell_offsets[id + 1]; ++p) {
                    const int j = cell_particles[p];
                    vm_float_t dx, dy, dz, r2;

                    if (j == i || j < j0 || j >= j1) {
                        continue;
                    }
                    dx = VECMAT_REMAINDER(x[i] - x[j], box.x);
                    dy = VECMAT_REMAINDER(y[i] - y[j], box.y);
                    dz = VECMAT_REMAINDER(z[i] - z[j], box.z);
                    r2 = (dx * dx + dy * dy) + dz * dz;
                    if (!(r2 < rcut2)) {
                        continue;
                    }
                    if (dst && count < dst_n) {
                        dst[count] = j;
                    }
                    count += 1;
                }
            }
        }
    }
    return count;
}

static void bin_particles(int *cell_of, int *cell_particles, int *cell_offsets,
                          const vm_float_t *x, const vm_float_t *y, const vm_float_t *z,
                          const int n, const vector3 box, const md_grid *g)
{
    const int ncells = g->ncells;
    int sum = 0;
    int i;
    int c;

    for (c = 0; c <= ncells; ++c) {
        cell_offsets[c] = 0;
    }
    for (i = 0; i < n; ++i) {
        const int id = locate(x, y, z, i, box, g);
        cell_of[i] = id;
        if (id >= 0) {
            cell_offsets[id] += 1;
        }
    }
    for (c = 0; c < ncells; ++c) {
        const int cnt = cell_offsets[c];
        cell_offsets[c] = sum;
        sum += cnt;
    }
    cell_offsets[ncells] = sum;

    for (i = 0; i < n; ++i) {
        const int id = cell_of[i];
        if (id < 0) {
            continue;
        }
        cell_particles[cell_offsets[id]] = i;
        cell_offsets[id] += 1;
    }
    for (c = ncells; c > 0; --c) {
        cell_offsets[c] = cell_offsets[c - 1];
    }
    cell_offsets[0] = 0;
}

int vm_md_neighbors_scratch(
    const int n, const int i0, const int ni, const int j0, const int nj,
    const vector3 box, const vm_float_t rcut)
{
    md_grid g;
    return prepare(&g, n, i0, ni, j0, nj, box, rcut);
}

bool vm_md_neighbors(
    int *offsets, int *neighbors, const int capacity,
    int *scratch, const int scratch_n,
    const vm_float_t *x, const vm_float_t *y, const vm_float_t *z,
    const int n, const int i0, const int ni, const int j0, const int nj,
    const vector3 box, const vm_float_t rcut)
{
    md_grid g;
    const int need = prepare(&g, n, i0, ni, j0, nj, box, rcut);
    int j1;
    vm_float_t rcut2;
    int *cell_of;
    int *cell_particles;
    int *cell_offsets;
    int *row_counts;
    int total = 0;
    int row;

    if (need < 0) {
        return false;
    }
    if (need == 0) {
        return true;
    }
    if (!scratch || scratch_n < need || capacity < 0 || !offsets || !x || !y || !z) {
        return false;
    }
    /* The range check passed, so these sums fit in int. */
    j1 = j0 + nj;
    rcut2 = rcut * rcut;

    cell_of = scratch;
    cell_particles = scratch + n;
    cell_offsets = scratch + (2 * n);
    row_counts = scratch + (2 * n) + g.ncells + 1;
    bin_particles(cell_of, cell_particles, cell_offsets, x, y, z, n, box, &g);

    for (row = 0; row < ni; ++row) {
        const int i = i0 + row;
        const int cell = cell_of[i];
        int cnt = 0;

        if (cell >= 0) {
            cnt = row_neighbors(NULL, 0, i, j0, j1, cell, &g, cell_offsets,
                                cell_particles, x, y, z, box, rcut2);
        }
        if (cnt > INT_MAX - total) {
            return false;
        }
        row_counts[row] = cnt;
        total += cnt;
    }
    if (total > capacity || (total > 0 && !neighbors)) {
        return false;
    }

    offsets[0] = 0;
    for (row = 0; row < ni; ++row) {
        offsets[row + 1] = offsets[row] + row_counts[row];
        if (row_counts[row] == 0) {
            continue;
        }
        row_neighbors(neighbors + offsets[row], row_counts[row], i0 + row, j0, j1,
                      cell_of[i0 + row], &g, cell_offsets, cell_particles,
                      x, y, z, box, rcut2);
    }
    return true;
}
