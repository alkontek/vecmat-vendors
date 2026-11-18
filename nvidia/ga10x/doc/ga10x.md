# nvidia/ga10x

GeForce 30xx device module (Ampere GA102, GA104, GA106, GA107). Compute capability 8.6 only. The first matching CUDA
device is registered as `nvidia-ga10x`. Anything else, including a missing driver, registers nothing.

```c
int rc = vm_nvidia_ga10x_register();
```

`VM_DEVICE_OK` means the device is in the table. A second call with the same module is idempotent.
`VM_DEVICE_ERR_UNSUPPORTED` means there is no GA10x, and the test returns without failing.

The module is a `feature` so `-DVECMAT_MODULES=nvidia/ga10x` is a legal loader entry. It does not append
`VECMAT_MODULE_HOOKS`. CUDA sources are not float objects. They build the static library `vecmat_nvidia_ga10x` for
`sm_86`. Configure fails if the CUDA toolkit is missing, or if the Vecmat build has no ABI 32. The default Vecmat build
does not name this module.

| Field | Value |
|-------|--------|
| kind | `VM_DEVICE_GPU` |
| name | `nvidia-ga10x` |
| dtypes | f32 |
| caps | `VM_DEVICE_CAP_ASYNC` |
| memory | that device's global memory |

`vm_buffer_wrap` is unsupported. There is no host-memory capability bit.

Ops, f32 only:

- `array_add`, `array_mul`, and `axpy` are elementwise kernels on the queue's stream.
- GEMM is row-major, packed, and not transposed. The body is a 16×16×8 WMMA TF32 kernel on the queue's stream. f32
  inputs are rounded to TF32 inside the product, so the result is not bit-exact with `vm_gemm32`. `k == 0` scales C by
  `beta` and does not read A or B. Edges that are not a multiple of the tile are zero-padded.

f64 returns unsupported from `vm_device_supports`. Callers use the CPU device.

f64, fp16, bf16, and TF32 as device dtypes stay out, as do Ada, Hopper, Blackwell, and GA100. Xeon AMX stays a CPU
feature. `vm_device_ops` and the public device calls are unchanged. Vecmat core is unchanged.
