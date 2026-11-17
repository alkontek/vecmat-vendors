# Supporting modules

Plan for the molecular-dynamics work that stays outside
[`alkontek/md`](md.md). Three feature modules live beside it, under
`alkontek/`, and call the entry points `md` already ships. A fourth
module, `alkontek/fft`, exists because particle-mesh Ewald needs a
transform and Vecmat has none.

| Module | Path | Role |
| --- | --- | --- |
| Topology | `alkontek/topology` | Bond graph and the exclusion CSR |
| Force field | `alkontek/force-field` | Parameter file, bonded forces |
| FFT | `alkontek/fft` | Complex 3-D transform on a regular grid |
| PME | `alkontek/pme` | Smooth particle-mesh Ewald |

This document is the plan. It does not add sources.

`alkontek/md` keeps the cutoff neighbor list, the shifted-force
Lennard-Jones and Coulomb tiles and list walks, the caller-supplied
exclusion CSR, and `vm_md_kinetic`. Those signatures stay as they are.
The new modules do not register a backend or a width. They use the same
selected-width header pattern as `vecmat/feature/md.h`: one body per
float ABI, unsuffixed names in the text, `32` and `64` on the symbols.
Build them against `alkavan/vecmat` `v0.4.x`. Do not edit that tree.
Do not change `vm_verlet`.

Each module is one change, in the order below. A module's tests run in
f32 and in f64. A rejected call returns false and writes nothing. An
empty count that is otherwise valid returns success and does not read
the arrays it would walk.

## Shared rules

Layout, copied from `alkontek/md` and `alkontek/spectral-graph`:

```text
alkontek/<name>/
├── doc/<name>.md
├── include/vecmat/feature/<name>.h
├── module.cmake          # VECMAT_MODULE_KIND feature
├── src/
└── test/test_<name>.c
```

`module.cmake` appends `VECMAT_MODULE_HOOKS`,
`VECMAT_MODULE_TEST_HOOKS`, and `VECMAT_MODULE_TEST_SOURCES`. It does
not append a backend or width register list. Sources that evaluate a
real expression compile with `-ffp-contract=off`. Products and sums
stay unfused. A sum of many terms is a materialized buffer passed to
`vm_sum_compensated`. There is no running Neumaier helper and no SIMD
in the first cut of any of these modules.

The box is orthorhombic, the same `vector3` of positive finite lengths
the neighbor list takes. The minimum image is
`vm_periodic_delta_ortho` (ties-to-even, `[-L/2, L/2]`). Bonded terms
and PME call that routine. They do not keep a private `remainder`.

The caller owns every buffer. A short output buffer is rejected before
any write. `energy` and `virial` on a force kernel may be NULL. The
caller zeroes forces, energy, and virial; the kernel adds. Virial is
the six components `xx, yy, zz, xy, xz, yz` from the minimum-image
vector and the force added to atom `i`, the same slots as
`vm_md_lj_tile`.

CI follows the `md` job: checkout `alkavan/vecmat` at `v0.4.x`, Ninja,
`VECMAT_BUILD_TESTS=ON`, `VECMAT_INSTALL=OFF`, then `vecmat_tests` once
with `VECMAT_USE_F64` off and once on. The existing fp16, bf16,
spectral-graph, and `md` jobs stay on their current module lists. A new
job names only the modules that change.

## 1. Topology

`alkontek/topology`. Header `vecmat/feature/topology.h`. Prefix
`vm_top_`.

The exclusion CSR in `md` is a skip list. This module is the bond graph
that produces that list. It stores no positions, no parameters, and no
forces.

Inputs are caller-owned index arrays of length `n` for the particle
count and length `nbonds` for the bonds. A bond is two absolute indexes
`bi[p]`, `bj[p]`. Order inside a pair does not matter. An angle is
`(i, j, k)` with `j` the vertex. A dihedral is `(i, j, k, l)` with the
axis `j–k`. Impropers are a separate list of the same shape. The first
cut stores impropers and checks their indexes. It attaches no potential
to them.

`vm_top_bonds_ok`, `vm_top_angles_ok`, and `vm_top_dihedrals_ok` return
false when `n < 0`, a count is negative, an index is outside `[0, n)`,
a bond is a self pair, or a bond is repeated. They write nothing.

Exclusions come from the bond graph, not from the angle list, so a
missing angle still excludes the 1–3 pair.

```c
int vm_top_excl_scratch(int n, int nbonds);

bool vm_top_excl(
    int *offsets, int *partners, int capacity,
    int *scratch, int scratch_n,
    int n, int i0, int ni,
    const int *bi, const int *bj, int nbonds,
    int max_bonds);
```

`max_bonds` is 1, 2, or 3. The value 1 emits 1–2 pairs, 2 emits 1–2 and
1–3, and 3 also emits 1–4. A 1–4 scale factor is a force-field choice.
Topology either puts the pair in the skip list or leaves it out.

The CSR is the one `vm_md_lj_list_excl` and `vm_md_coulomb_list_excl`
already consume. `offsets` has length `ni + 1` and `offsets[0] == 0`.
Row `k` belongs to particle `i0 + k`. Packed partners are absolute
indexes. Both endpoints are stored, because the list walk consults only
the row of `i`. Partner order does not matter. A partner may lie
outside the row span and must still lie in `[0, n)`.

`vm_top_excl_scratch` returns the int count, 0 for `ni == 0` or
`nbonds == 0` when the other arguments are valid, and -1 when they are
not. `ni == 0` makes `vm_top_excl` return true and write nothing.
Scratch, offsets, and partners are not read in that case. A short
partner buffer returns false and does not write `offsets` or
`partners`.

Tests, with `alkontek/md` loaded beside this module:

- A four-atom chain. `max_bonds` 1, 2, and 3 produce the right partner
  sets, and each pair is stored on both rows.
- A self bond, a repeated bond, and an index past `n` are rejected.
- `ni == 0` leaves a sentinel CSR untouched.
- The CSR from the chain is accepted by `vm_md_lj_list_excl`. The
  excluded pair adds no force, energy, or virial. A pair outside the
  skip set still matches `vm_md_lj_list`.

## 2. Force field

`alkontek/force-field`. Header `vecmat/feature/force_field.h`. Prefix
`vm_ff_`.

The file format belongs to this module. It is a line-oriented text
buffer, not a path, so the tests never touch the disk. It is not a
GROMACS, Amber, or CHARMM file. A later reader can fill the same tables.

`n` is the first record. Then one record per line. `#` starts a
comment. An unknown key is a hard error.

```text
n <count>
mass <n floats>
charge <n floats>
box <Lx> <Ly> <Lz>
lj <epsilon> <sigma> <rcut>
coulomb <prefactor> <rcut>
excl <max_bonds>
bond <i> <j> <fc> <r0>
angle <i> <j> <k> <fc> <theta0>
dihedral <i> <j> <k> <l> <fc> <n> <phi0>
```

`mass` and `charge` are the SoA arrays `vm_md_kinetic` and
`vm_md_coulomb_*` already take. `lj` and `coulomb` are the scalar
arguments `vm_md_lj_*` and `vm_md_coulomb_*` already take: one
epsilon, one sigma, one cutoff, one prefactor. Per-type Lennard-Jones
and combination rules need a new `md` kernel. They are not part of this
file. `excl` is the `max_bonds` argument of `vm_top_excl`. `theta0` and
`phi0` are radians. The dihedral multiplicity `n` is an integer from 1
to 6. One Fourier term per dihedral line.

Positions and velocities are not in the file. The caller owns those SoA
arrays.

```c
typedef struct vm_ff_counts {
    int n;
    int nbonds;
    int nangles;
    int ndihedrals;
} vm_ff_counts;

bool vm_ff_counts(vm_ff_counts *counts, const char *text, int text_n);

bool vm_ff_parse(
    vm_float_t *mass, vm_float_t *charge,
    vm_float_t *epsilon, vm_float_t *sigma, vm_float_t *lj_rcut,
    vm_float_t *prefactor, vm_float_t *coulomb_rcut,
    vector3 *box, int *max_bonds,
    int *bond_i, int *bond_j, vm_float_t *bond_k, vm_float_t *bond_r0,
    int *angle_i, int *angle_j, int *angle_k,
    vm_float_t *angle_fc, vm_float_t *angle_theta0,
    int *dihedral_i, int *dihedral_j, int *dihedral_k, int *dihedral_l,
    vm_float_t *dihedral_fc, int *dihedral_n, vm_float_t *dihedral_phi0,
    const char *text, int text_n,
    const vm_ff_counts *cap);
```

`vm_ff_counts` writes the struct on success. A bad buffer returns false
and leaves `*counts` unchanged. `cap` is the capacity the caller
allocated, including `cap->n`. A short array returns false and writes
nothing. Numbers are scanned with `strtof` or `strtod` for the ABI in
force. The test fixtures use decimals that are exact in both f32 and
f64, and compare with `memcmp`.

Bonded kernels live here. `md` has no bonds.

Harmonic bond, `r` the minimum-image length:

```text
U = (1/2) k (r - r0)^2
```

The force on `i` is the analytic derivative. With `d = ri - rj`, a
stretched bond (`r > r0`) pulls `i` back along `-d`, so `dx * Fx` is
negative. That is tension. The Lennard-Jones virial comment is the
opposite case (repulsion, `dx * Fx` positive). The six slots are the
same ones.

Harmonic angle, `theta` from the two minimum-image legs at the vertex:

```text
U = (1/2) k (theta - theta0)^2
```

One Fourier term, `phi` the torsion in `(-π, π]` from `atan2` of the
two planes around `j–k`:

```text
U = k (1 + cos(n phi - phi0))
```

Bond forces and energies match that closed form bit for bit, including
a bond that crosses a periodic face. Angle and dihedral energies match
the closed form bit for bit. Their forces are the analytic derivative,
checked on the fixtures by a central difference of that energy. The
tolerance is recorded in the test. Accepted bond energies are
materialized and passed to `vm_sum_compensated`. The same for angles and
for dihedrals. `k` is finite and `>= 0`. A zero spring is an accepted
term and adds nothing. `r0` is finite and positive. `theta0` and `phi0`
are finite. The box lengths are finite and positive.

```c
bool vm_ff_bonds(
    vm_float_t *fx, vm_float_t *fy, vm_float_t *fz,
    vm_float_t *energy, vm_float_t virial[6],
    const vm_float_t *x, const vm_float_t *y, const vm_float_t *z,
    int n,
    const int *bi, const int *bj,
    const vm_float_t *k, const vm_float_t *r0, int nbonds,
    vector3 box);
```

`vm_ff_angles` and `vm_ff_dihedrals` take the matching index and
parameter arrays. `nbonds == 0` returns true and writes nothing.

This module does not call `vm_md_*`. It builds alone. The caller, after
a successful parse, runs:

1. `vm_top_excl` with the bond arrays and `max_bonds`.
2. `vm_md_neighbors` for the pair list.
3. `vm_md_lj_list_excl` with the file's `epsilon`, `sigma`, and `lj_rcut`.
4. `vm_md_coulomb_list_excl` with the file's `prefactor`, charges, and
   `coulomb_rcut`.
5. `vm_ff_bonds`, `vm_ff_angles`, `vm_ff_dihedrals`.
6. `vm_md_kinetic` when a kinetic energy is wanted.

Tests:

- A short buffer round-trips `n`, masses, charges, the LJ scalars, one
  bond, one angle, and one dihedral.
- A bond at a known separation matches `(1/2) k (r - r0)^2` and the
  force magnitude `k |r - r0|`.
- The same bond, placed across a periodic face, matches
  `vm_periodic_delta_ortho`.
- An empty bond list leaves a sentinel force and energy in place.
- An unknown key, a truncated `mass` line, and a short capacity leave
  every output untouched.

## 3. FFT

`alkontek/fft`. Header `vecmat/feature/fft.h`. Prefix `vm_fft_`.

Vecmat's spectral header is a Jacobi eigensolver (`vm_syev`,
`vm_syev_topk`) and a dense graph Laplacian. It is not a Fourier
transform. PME stops until this module exists.

The first cut is a complex 3-D radix-2 transform. Each axis length is a
power of two, including `1`. The data are two unit-stride SoA arrays,
real and imaginary. The index is `x + nx * (y + ny * z)`, with `x`
contiguous. The transform is out of place. The caller owns both sides.

```c
bool vm_fft3(
    vm_float_t *out_re, vm_float_t *out_im,
    const vm_float_t *in_re, const vm_float_t *in_im,
    int nx, int ny, int nz,
    int sign);
```

`sign == +1` is the forward transform. `sign == -1` is the inverse and
divides by `nx * ny * nz`, so a round trip returns the input. Any other
`sign`, a non-power-of-two axis, a negative axis, or a missing pointer
when the grid is non-empty returns false and writes nothing. A grid
with a zero axis is rejected. There is no in-place mode and no real
packed layout in the first cut.

Tests: an impulse round-trips in f32 and f64 within a small absolute
tolerance; a single sine mode along `x` lands in the expected bin; a
rejected size leaves the output sentinel in place.

## 4. PME

`alkontek/pme`. Header `vecmat/feature/pme.h`. Prefix `vm_pme_`. It
includes `vecmat/feature/fft.h` and calls `vm_fft3`. The CI job lists
`alkontek/fft` and `alkontek/pme` together.

Smooth particle-mesh Ewald on the orthorhombic box, cardinal B-spline
of order 4, conducting boundary. The `k = 0` term is omitted. The
caller passes `beta` and `rcut`. Both are finite and positive. The
Coulomb scale is the same `prefactor` as `vm_md_coulomb_*`, finite and
positive. The module does not pick a default `beta`.

The direct sum below is the test reference. `k = 2π (mx/Lx, my/Ly, mz/Lz)`.
`V` is `Lx * Ly * Lz`. The pair sum counts each unordered pair once.

```text
U = sum_{i < j} prefactor * qi * qj * erfc(beta * r_ij) / r_ij
  + prefactor * (2π / V) * sum_{k != 0}
        exp(-|k|² / (4 beta²)) / |k|² * |sum_j qj exp(i k · rj)|²
  - prefactor * (beta / sqrt(π)) * sum_i qi²
```

The vacuum surface term in `(sum qi)²` is absent. A non-neutral system
is accepted.

`vm_md_coulomb_*` stays the shifted-force cutoff kernel. PME does not
call it. The real-space pair is a hard cutoff of the `erfc` term:
`r² >= rcut²` adds nothing. A shifted-force `erfc` is a later function.

### Real space

```c
bool vm_pme_real(
    vm_float_t *fx, vm_float_t *fy, vm_float_t *fz,
    vm_float_t *energy, vm_float_t virial[6],
    const vm_float_t *x, const vm_float_t *y, const vm_float_t *z,
    const vm_float_t *q,
    int n, int i0, int ni,
    const int *offsets, const int *neighbors,
    const int *excl_offsets, const int *excl_partners,
    vector3 box, vm_float_t prefactor, vm_float_t beta, vm_float_t rcut);
```

The neighbor CSR and the exclusion CSR have the shapes `md` already
documents. `excl_offsets == NULL` means there are no exclusions. The
walk adds the force on `i` only, which is what `vm_md_coulomb_list`
does, so a both-directions list still matches a full force. Energy is
added only when `i < j`, so that list does not double the unordered
pair sum. The header states this energy rule next to the Coulomb list
rule.

Unfused evaluation, `d` the minimum image, `s = beta * r`:

```text
r2 = (dx*dx + dy*dy) + dz*dz
r = sqrt(r2)
er = erfc(s)
ex = exp(-(s*s))
inv_r = 1 / r
inv_r2 = inv_r * inv_r
scale = prefactor * (qi * qj)
U = scale * (er * inv_r)
coeff = scale * (inv_r2 * ((er * inv_r) + ((2 * beta / sqrt(π)) * ex)))
F_i = coeff * d
```

Like-signed charges make `dx * Fx` positive. `erfc` and `exp` are the
libm functions. `i == j` is skipped. An excluded partner is skipped
here and handled by the correction below. `ni == 0`, or a CSR with no
stored neighbors, returns true and writes nothing.

### Self energy

```c
bool vm_pme_self(
    vm_float_t *energy,
    const vm_float_t *q, int n,
    vm_float_t prefactor, vm_float_t beta);
```

Materialize `qi * qi`, call `vm_sum_compensated`, then scale by
`prefactor * (-beta / sqrt(π))`. `n == 0` writes `+0`. This replaces
`*energy`. It does not add.

### Exclusion correction

The mesh spreads every charge, including bonded neighbors. A pair in
the exclusion CSR must lose the reciprocal piece the direct sum would
have given it:

```text
U_corr = prefactor * qi * qj * erf(beta * r) / r
```

```c
bool vm_pme_excl(
    vm_float_t *fx, vm_float_t *fy, vm_float_t *fz,
    vm_float_t *energy, vm_float_t virial[6],
    const vm_float_t *x, const vm_float_t *y, const vm_float_t *z,
    const vm_float_t *q,
    int n, int i0, int ni,
    const int *excl_offsets, const int *excl_partners,
    vector3 box, vm_float_t prefactor, vm_float_t beta);
```

Walk the exclusion CSR. When `i < j`, add `-U_corr` to the energy and
add the analytic force of that contribution on `i` only. Same virial
slots.
This is the direct-Ewald `erf` term. The remaining gap to the mesh is
the usual spline error, which the reciprocal test measures.

### Reciprocal space

```c
int vm_pme_scratch(int n, int nx, int ny, int nz);

bool vm_pme_recip(
    vm_float_t *fx, vm_float_t *fy, vm_float_t *fz,
    vm_float_t *energy,
    vm_float_t *grid_re, vm_float_t *grid_im,
    const vm_float_t *x, const vm_float_t *y, const vm_float_t *z,
    const vm_float_t *q,
    int n, int nx, int ny, int nz,
    vector3 box, vm_float_t prefactor, vm_float_t beta);
```

`nx`, `ny`, and `nz` are powers of two and at least 4, so an order-4
spline has a cell to stand on. `vm_pme_scratch` returns the float count
of work space, or -1. `grid_re` and `grid_im` are caller-owned, length
`nx * ny * nz`, the layout `vm_fft3` uses.

Steps: fold each charge into `[0, L)` with the same `remainder` rule as
`vm_md_neighbors`, spread it with the order-4 cardinal B-spline, forward
FFT, multiply by the influence function, inverse FFT, interpolate the
field with the same weights, and add `q E` to the particle. Positions
are not written. The influence function is the conducting-boundary
smooth PME factor of Essmann, Darden, York, and Pedersen, J. Chem. Phys.
103, 8577 (1995), including the B-spline Fourier correction. The header
writes the expression the code evaluates. Reciprocal virial is not in
this function. `energy` is the reciprocal piece only. The caller adds
real space, the exclusion correction, and the self term.

`n == 0` with a valid grid writes `+0` and does not read coordinates or
charges. A bad grid returns false and writes nothing.

Tests:

- One real-space pair matches the closed form and
  `vm_periodic_delta_ortho`. An exact cutoff adds nothing.
- The self term matches `vm_sum_compensated` of `qi²`, then the scale.
- Two charges marked excluded: real space skips them, and
  `vm_pme_excl` matches a hand `erf` evaluation.
- One charge spread onto the grid: the spline weights sum to `q`.
- Two opposite charges on a small power-of-two grid. The sum of real
  space, reciprocal, exclusion correction, and self lies within an
  absolute tolerance of the direct sum truncated to the wavevectors on
  that grid. The tolerance is whatever the fixture needs, and the test
  records the number. This comparison is not bitwise.
- One coordinate step. The change in that total energy agrees with the
  reported force to a stated tolerance.
- A rejected grid leaves the force and energy sentinels in place.

An integration test that also loads `alkontek/md` and
`alkontek/topology` builds the neighbor list and the exclusion CSR, then
calls `vm_pme_real`. That job is separate from the `md` job.

## Core gaps

These are why the work is split this way.

- No FFT. `vm_syev` does not fill that role. `alkontek/fft` does.
- No running Neumaier step. Bond energies, the self term, and any other
  reduction materialize their terms and call `vm_sum_compensated`.
- No per-type Lennard-Jones parameters on `vm_md_lj_*`. The force-field
  file carries one `epsilon` and one `sigma`.
- `vm_periodic_delta_ortho` is AoS `vector3`. Bonded terms and PME call
  it with one pair, or a short array. The `md` tiles keep their own
  remainder and stay as they are.
- `vm_wrap_ortho` writes an AoS array in place. Charge spreading folds a
  copy of the coordinate. Particle positions stay put.
- `vm_soa3_distance_squared` has no periodic image.

## Out of scope

Triclinic boxes. SHAKE, LINCS, and rigid waters. GROMACS, Amber, and
CHARMM files. 1–4 scale factors. Combination rules. Reciprocal virial.
A shifted-force `erfc` cutoff. SIMD. An integrator, including any change
to `vm_verlet`.

## Order

1. `alkontek/topology`, including the `vm_md_lj_list_excl` check.
2. `alkontek/force-field`, parser and bonded kernels.
3. `alkontek/fft`.
4. `alkontek/pme`, real space, self, and the exclusion correction.
5. `alkontek/pme` reciprocal space, on top of `vm_fft3`.

Each of those is one pull request into `master`.
