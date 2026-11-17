# Molecular dynamics neighbors and pairs

Feature module for [Vecmat](https://github.com/alkavan/vecmat) `v0.4.x`.
Loaded as `alkontek/md`. It does not register a backend or a width.

`vm_md_neighbors` is a cutoff neighbor list. `vm_md_lj_tile` and
`vm_md_coulomb_tile` are rectangular shifted-force tiles. `vm_md_lj_list`
and `vm_md_coulomb_list` evaluate those same pairs on a caller-supplied
CSR. `vm_md_lj_list_excl` and `vm_md_coulomb_list_excl` are those walks
with a caller-supplied exclusion CSR. `vm_md_kinetic` is the kinetic
energy of one particle span. The Coulomb scale is the `prefactor`
argument, in the caller's units. Charges are a per-particle SoA array.
The minimum image is the orthorhombic `remainder` rule from
`vm_periodic_delta_ortho` (ties-to-even, `[-L/2, L/2]`). Pair energies
use the Neumaier recurrence, so `vm_sum_compensated` on the accepted pair
terms is the oracle. PME, topology, and force-field files are not in
this module. The plan for those modules is
[supporting-modules.md](supporting-modules.md).

```bash
cmake -S vecmat -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DVECMAT_MODULES=alkontek/md \
  -DVECMAT_MODULES_PATH=../vecmat-vendors \
  -DVECMAT_BUILD_TESTS=ON \
  -DVECMAT_INSTALL=OFF
cmake --build build --parallel --target vecmat_tests
./build/test/vecmat_tests
```

Point `vecmat` at a checkout of `alkavan/vecmat` ref `v0.4.x`. The vendors
CI job `md` does that. The fp16, bf16, and spectral-graph jobs stay on
`alkontek/vecmat` `master`.

AVX2 (`-mavx2 -mfma`, contraction off) runs only when
`vm_cpu_has(VM_LEVEL_AVX2)`. NEON runs only when
`vm_cpu_has(VM_LEVEL_NEON)`. Both are compiled on the same hosts as Vecmat's
own SIMD sources. The tile tests compare them to the scalar evaluation
bit for bit, including the `r² >= rcut²` decision. That absolute bound is
zero on the fixtures below. Both tiles ship the AVX2 and NEON paths.
The neighbor list and the list walk are scalar.

## Cutoff neighbor list

`vm_md_neighbors` writes CSR indexes for one rectangle of particles.
`offsets` has length `ni + 1`. Row `k` is particle `i0 + k`, and its
neighbors are absolute `j` indexes. The caller owns `offsets`,
`neighbors`, and an opaque scratch buffer. `vm_md_neighbors_scratch`
returns that length in ints, `0` for an empty range, or `-1` when the
arguments are rejected.

Cells are equal and tile the orthorhombic box. On an axis the cell count
is the largest integer with edge `L/count >= rcut`, or one cell when the
axis is shorter than the cutoff. The edge is the cutoff when the cutoff
divides the axis. Each particle searches the 27 periodic neighbor cells,
and a repeated cell is visited once. A pair is stored when `j` is in the
column range, `j != i`, and the minimum-image `r^2` is strictly below
`rcut^2`. An exact cutoff is omitted. Both directions are stored when
both indexes lie in both ranges.

`n < 0`, a bad range, a non-positive box or cutoff, a short scratch
buffer, or a neighbor buffer shorter than the pair count returns false
and does not write the CSR. `n == 0`, or `ni == 0`, or `nj == 0`, returns
true and writes nothing.

The neighbor tests use `vm_periodic_delta_ortho` and
`vm_soa3_length_squared` as the pair oracle. They cover a five-particle
cluster plus one non-finite coordinate (the directed pair count is 6),
cutoff exclusion, self exclusion, an empty range, a particle just outside
a periodic face, and a one-cell box that must not list the same pair 27
times.

Core still has no SoA or one-point `vm_periodic_delta_ortho`, and no
running Neumaier step. The neighbor builder does not add a local Neumaier
copy. `vm_sum_compensated` remains the batch sum the tiles already use.
`vm_soa3_distance_squared` does not take a periodic image, and
`vm_wrap_ortho` is in-place AoS, so the bin fold stays local.

## Neighbor-list pairs

`vm_md_lj_list` and `vm_md_coulomb_list` walk a CSR. `offsets` has length
`ni + 1` and starts at 0. `neighbors` holds absolute `j` indexes. The
walk calls the rectangular tile's pair step, so the minimum image, the
cutoff test, and the shifted-force energy and virial are the same. There
is no second pair kernel and no SIMD path for the walk. The tile SIMD
paths are unchanged.

The list stores both directions when both indexes lie in the rectangle,
so the walk adds the force on `i` only. A self index is skipped.
`r^2 >= rcut^2` adds nothing. Energy and virial are counted once per
accepted stored pair, in list order. A both-directions list therefore
matches the forces of a Newton self-tile, and the energy and virial of
that rectangle with `newton == 0`. A one-direction list, such as a
disjoint pair of blocks, matches `newton == 0` on that rectangle,
including energy and virial.

A false return writes nothing. `ni == 0`, or a CSR that stores no
neighbors, returns true and does not touch the buffers.

The walk uses the tile's file-local remainder and Neumaier step. It does
not add another helper. The core gaps above are unchanged.

## List exclusions

`vm_md_lj_list_excl` and `vm_md_coulomb_list_excl` take the same arguments
as the list walks, plus an exclusion CSR. The rectangular tile APIs are
unchanged. `vm_md_lj_list` and `vm_md_coulomb_list` are the walks with no
exclusions.

An exclusion is an unordered pair of particle indexes. The set is a CSR
with the same shape as the neighbor list, which is less machinery than a
second search structure. `excl_offsets` has length `ni + 1` and starts at
0. Row `k` belongs to particle `i0 + k`, and its packed entries are the
absolute partner indexes excluded with that particle. The row does not
have to be sorted. This is not a topology graph.

A stored neighbor `j` is skipped when `j` appears on that row. The
partner's row is not read, so a both-directions list skips each direction
only from its own row. The caller lists both endpoints. A skipped pair
adds no force, energy, or virial. A self index is still skipped before
that test. Pairs that remain are visited in neighbor-list order and use
the same pair step as the list kernel.

`excl_offsets == NULL` means there are no exclusions, and `excl_partners`
is not read. An empty CSR is the same result. Either one matches
`vm_md_lj_list` or `vm_md_coulomb_list` bit for bit. A decreasing offset,
a partner outside `[0, n)`, or a missing partner pointer when a partner
is stored returns false and writes nothing, including when the neighbor
CSR itself is empty. `ni == 0` does not read the exclusion CSR.

Both walks call one file-local membership test. There is no SIMD path.

## Lennard-Jones fixture

Nineteen particles in a box `(5.5, 4.25, 6)`. `ε = 0.8`, `σ = 1.15`,
`rcut = 2.5`. The coordinates mix an exact cutoff pair, a ties-to-even
image at `Lx/2`, a periodic wrap, and incommensurate offsets. Self-tiles
and disjoint tiles are both checked, with `newton` on and off. The bound
is bitwise equality with the scalar tile (`memcmp` on every force, the
energy, and the six virial components).

## Coulomb fixture

The same nineteen coordinates and box, with `prefactor = 2.3` and
`rcut = 2.5`. Charges mix positive, negative, and zero, including an
attractive wrap, a repulsive interior pair, and a particle on the cutoff.
The shift is part of the closed form: an interior pair does not match the
plain `prefactor * qi * qj / r` truncate. Like-signed charges make
`dx * Fx` positive, the same virial sign as the Lennard-Jones tile.
Self-tiles and disjoint tiles are both checked, with `newton` on and off.
The bound is the same bitwise comparison.

## Kinetic energy

`vm_md_kinetic` writes `sum m |v|^2 / 2` for the span `[i0, i0+ni)` of
unit-stride velocities `vx, vy, vz` and per-particle masses. The sum
replaces `*energy`. It is not added to the previous value.

Each term is `(m * v2) * 0.5` with `v2 = (vx*vx + vy*vy) + vz*vz`,
unfused. The `ni` terms are materialized and passed to
`vm_sum_compensated`. There is no running Neumaier helper and no SIMD
path. `vm_sum_compensated` of those terms is the oracle.

`ni == 0` on an in-range span writes `+0` and does not read the SoA.
A rejected call returns false and writes nothing. Masses are not checked
for sign. The term buffer is allocated for the call.

Core still has no running Neumaier step, and no weighted SoA reduction
that would form this sum without a temporary. `vm_soa3_length_squared`
writes `|v|^2` and does not apply a mass or a compensated sum.
`vm_dot_compensated` still needs those `|v|^2` terms materialized first.
`vm_sum_compensated` remains the batch sum. The kinetic entry point does
not add a local Neumaier copy.
