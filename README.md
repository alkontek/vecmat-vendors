# Vendors of Vecmat 

Optional modules for [Vecmat](https://github.com/alkavan/vecmat) a simple math linear algebra library in C.

Modules live under `<vendor>/<name>/` and are loaded when their path appears in `VECMAT_MODULES`.
Each module provides a `module.cmake` that sets `VECMAT_MODULE_KIND` to `backend|width|feature`.

## Using the official repository

After cloning vecmat, add this repository as the `vendors/` submodule:

```bash
cd vecmat
git submodule add git@github.com:alkontek/vecmat-vendors.git vendors
git submodule update --init
```

Then configure with the modules you want:

```bash
cmake -S . -B build \
  -DVECMAT_MODULES="arm/cortex-a5x;arm/fp16"
```

Paths are relative to `vendors/`. You can also keep the repository beside the tree and pass
`-DVECMAT_MODULES_PATH=../vecmat-vendors` instead of using a submodule.

## Your own vendors repository

Anyone can create a `vendors/` repository. The only requirement is the same layout: a directory tree of
`<vendor>/<name>/module.cmake` (plus the sources the module lists).

Point vecmat at it the same way:

```bash
# as a submodule
git submodule add git@github.com:you/my-vecmat-vendors.git vendors

# or as an external root
cmake -S . -B build \
  -DVECMAT_MODULES_PATH=/path/to/my-vecmat-vendors \
  -DVECMAT_MODULES="acme/fast-gemm"
```

---

**Official modules currently include:**

- `arm/fp16` — Arm fp16 GEMM override (width)
- `intel/fp16` — Intel fp16 GEMM override (width)
- `arm/cortex-a5x` — Cortex-A53 / A55 schedule (backend; internal)  
  fully tested on [NXP FRDM-IMX95](https://www.nxp.com/design/design-center/development-boards-and-designs/FRDM-IMX95) development board.
