# Spectral graph

Feature module for [Vecmat](https://github.com/alkavan/vecmat). Loaded as
`alkontek/spectral-graph`.

Bitset primitives and combinatorial unrank. The dense Laplacian builder turns
row bitsets into a dense matrix the in-tree spectral kernels can use.

```bash
cmake -S vecmat -B build \
  -DVECMAT_MODULES="alkontek/spectral-graph" \
  -DVECMAT_MODULES_PATH=../vecmat-vendors
```
