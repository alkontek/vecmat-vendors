// SPDX-FileCopyrightText: 2025-2026 Igal Alkon
// SPDX-FileCopyrightText: 2026 ALKONTEK <git@alkontek.com>
// SPDX-License-Identifier: BSD-3-Clause

/**
 * CUDA side of the GA10x device. Compiled once for sm_86, not as a float
 * object. f32 GEMM runs on the GA10x tensor cores (WMMA TF32).
 */

#include <stddef.h>
#include <stdlib.h>

#include <cuda_runtime.h>
#include <mma.h>

/**
 * @brief 16x16x8 TF32 WMMA GEMM kernel (C = alpha * A*B + beta * C).
 *
 * Inputs remain f32; each product is rounded to TF32. Short edges are
 * zero-padded. When @p k is 0, A and B are not read.
 *
 * @param C Output matrix (m x n), row-major; may be scaled by beta in-place.
 * @param A Input matrix A (m x k), row-major; may be null if unused.
 * @param B Input matrix B (k x n), row-major; may be null if unused.
 * @param m Number of rows of A and C.
 * @param n Number of columns of B and C.
 * @param k Shared inner dimension of A and B.
 * @param alpha Scale factor for the A*B product.
 * @param beta Scale factor for the existing C values (0 skips reading C).
 */
__global__ void ga10x_gemm_tf32(float *C, const float *A, const float *B,
                                int m, int n, int k, float alpha, float beta)
{
    const int TM = 16;
    const int TN = 16;
    const int TK = 8;
    const int warp_in_block = (int)threadIdx.x >> 5;
    const int lane = (int)threadIdx.x & 31;
    const int warps = (int)blockDim.x >> 5;
    const int tiles_n = (n + TN - 1) / TN;
    const int tile = (int)blockIdx.x * warps + warp_in_block;
    const int tile_count = ((m + TM - 1) / TM) * tiles_n;
    int tile_m, tile_n, stride;
    extern __shared__ float smem[];
    float *as, *bs, *cs;

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 8,
                           nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major> af;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 8,
                           nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major> bf;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 8, float> cf;

    if (tile >= tile_count) {
        return;
    }
    tile_m = tile / tiles_n;
    tile_n = tile - tile_m * tiles_n;
    stride = TM * TK + TK * TN + TM * TN;
    as = smem + warp_in_block * stride;
    bs = as + TM * TK;
    cs = bs + TK * TN;

    nvcuda::wmma::fill_fragment(cf, 0.0f);
    for (int kk = 0; kk < k; kk += TK) {
        for (int i = lane; i < TM * TK; i += 32) {
            const int r = i / TK;
            const int c = i - r * TK;
            const int gr = tile_m * TM + r;
            const int gc = kk + c;
            as[i] = (A && gr < m && gc < k)
                ? A[(size_t)gr * (size_t)k + (size_t)gc] : 0.0f;
        }
        for (int i = lane; i < TK * TN; i += 32) {
            const int r = i / TN;
            const int c = i - r * TN;
            const int gr = kk + r;
            const int gc = tile_n * TN + c;
            bs[i] = (B && gr < k && gc < n)
                ? B[(size_t)gr * (size_t)n + (size_t)gc] : 0.0f;
        }
        __syncwarp();
        nvcuda::wmma::load_matrix_sync(af, as, TK);
        nvcuda::wmma::load_matrix_sync(bf, bs, TN);
        nvcuda::wmma::mma_sync(cf, af, bf, cf);
        __syncwarp();
    }

    nvcuda::wmma::store_matrix_sync(cs, cf, TN, nvcuda::wmma::mem_row_major);
    __syncwarp();
    for (int i = lane; i < TM * TN; i += 32) {
        const int r = i / TN;
        const int c = i - r * TN;
        const int gr = tile_m * TM + r;
        const int gc = tile_n * TN + c;
        if (gr < m && gc < n) {
            const float acc = cs[r * TN + c];
            const float prev = (beta != 0.0f)
                ? C[(size_t)gr * (size_t)n + (size_t)gc] : 0.0f;
            C[(size_t)gr * (size_t)n + (size_t)gc] = alpha * acc + beta * prev;
        }
    }
}

extern "C" {

#include <vecmat.h>

struct ga10x_pending {
    void *ptr;
    int device_mem; /* non-zero: cudaFree. Zero: cudaFreeHost. */
    struct ga10x_pending *next;
};

struct ga10x_queue {
    cudaStream_t stream;
    struct ga10x_pending *pending;
};

/**
 * @brief Map a CUDA error onto a device status.
 *
 * @param err CUDA status from the call just made.
 * @return `VM_DEVICE_OK`, `VM_DEVICE_ERR_NOMEM` for an allocation failure,
 *         or `VM_DEVICE_ERR_DEVICE` for any other error.
 */
static int ga10x_map(cudaError_t err)
{
    if (err == cudaSuccess) {
        return VM_DEVICE_OK;
    }
    if (err == cudaErrorMemoryAllocation) {
        return VM_DEVICE_ERR_NOMEM;
    }
    return VM_DEVICE_ERR_DEVICE;
}

/**
 * @brief Make @p ordinal the current CUDA device.
 *
 * @param ordinal Device index from `ga10x_probe`.
 * @return Mapped `cudaSetDevice` status.
 */
static int ga10x_set(int ordinal)
{
    return ga10x_map(cudaSetDevice(ordinal));
}

/**
 * @brief Keep @p ptr alive until the queue is synced.
 *
 * A null pointer is ignored. @p device_mem selects `cudaFree`; otherwise the
 * pointer came from `cudaMallocHost`. If the tracker cannot be allocated,
 * the stream is synchronized and @p ptr is freed immediately.
 *
 * @param q Queue that will outlive the queued copy.
 * @param ptr Allocation to free later, or NULL.
 * @param device_mem Non-zero for device memory, zero for pinned host memory.
 * @return `VM_DEVICE_OK`, or `VM_DEVICE_ERR_NOMEM` if it had to free @p ptr now.
 */
static int ga10x_retain(ga10x_queue *q, void *ptr, int device_mem)
{
    ga10x_pending *node;

    if (!ptr) {
        return VM_DEVICE_OK;
    }
    node = (ga10x_pending *)malloc(sizeof *node);
    if (!node) {
        cudaStreamSynchronize(q->stream);
        if (device_mem) {
            cudaFree(ptr);
        } else {
            cudaFreeHost(ptr);
        }
        return VM_DEVICE_ERR_NOMEM;
    }
    node->ptr = ptr;
    node->device_mem = device_mem;
    node->next = q->pending;
    q->pending = node;
    return VM_DEVICE_OK;
}

/**
 * @brief Free every pointer retained on @p q.
 *
 * Does not synchronize. Callers sync first so queued copies are finished.
 *
 * @param q Queue whose pending list is cleared.
 */
static void ga10x_release(ga10x_queue *q)
{
    ga10x_pending *node = q->pending;
    q->pending = NULL;
    while (node) {
        ga10x_pending *next = node->next;
        if (node->device_mem) {
            cudaFree(node->ptr);
        } else {
            cudaFreeHost(node->ptr);
        }
        free(node);
        node = next;
    }
}

/**
 * @brief Find the first compute-capability 8.6 device.
 *
 * A failed property query skips that device. No match, or a failed device
 * count, registers nothing.
 *
 * @param ordinal Receives the CUDA device index.
 * @param mem_bytes Receives `totalGlobalMem` for that device.
 * @return `VM_DEVICE_OK`, or `VM_DEVICE_ERR_UNSUPPORTED`.
 */
int ga10x_probe(int *ordinal, size_t *mem_bytes)
{
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);

    if (err != cudaSuccess || count <= 0) {
        return VM_DEVICE_ERR_UNSUPPORTED;
    }

    for (int i = 0; i < count; ++i) {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, i);
        if (err != cudaSuccess) {
            continue;
        }
        if (prop.major == 8 && prop.minor == 6) {
            *ordinal = i;
            *mem_bytes = (size_t)prop.totalGlobalMem;
            return VM_DEVICE_OK;
        }
    }
    return VM_DEVICE_ERR_UNSUPPORTED;
}

/**
 * @brief Allocate device memory.
 *
 * @param ordinal CUDA device index.
 * @param bytes Number of bytes. The caller has already rejected zero.
 * @param native Receives the device pointer.
 * @return Mapped `cudaMalloc` status.
 */
int ga10x_buffer_create(int ordinal, size_t bytes, void **native)
{
    int rc = ga10x_set(ordinal);
    if (rc != VM_DEVICE_OK) {
        return rc;
    }
    return ga10x_map(cudaMalloc(native, bytes));
}

/**
 * @brief Free a device allocation.
 *
 * Does not free when the device cannot be selected.
 *
 * @param ordinal CUDA device index.
 * @param native Device pointer from `ga10x_buffer_create`.
 */
void ga10x_buffer_destroy(int ordinal, void *native)
{
    if (ga10x_set(ordinal) != VM_DEVICE_OK) {
        return;
    }
    cudaFree(native);
}

int ga10x_queue_sync(int ordinal, void *native);

/**
 * @brief Create an in-order stream.
 *
 * @param ordinal CUDA device index.
 * @param native Receives the queue object. Freed if the stream cannot be created.
 * @return `VM_DEVICE_OK`, `VM_DEVICE_ERR_NOMEM`, or a mapped CUDA status.
 */
int ga10x_queue_create(int ordinal, void **native)
{
    ga10x_queue *q;
    int rc = ga10x_set(ordinal);
    cudaStream_t stream = NULL;

    if (rc != VM_DEVICE_OK) {
        return rc;
    }
    q = (ga10x_queue *)calloc(1, sizeof *q);
    if (!q) {
        return VM_DEVICE_ERR_NOMEM;
    }
    rc = ga10x_map(cudaStreamCreate(&stream));
    if (rc != VM_DEVICE_OK) {
        free(q);
        return rc;
    }
    q->stream = stream;
    *native = q;
    return VM_DEVICE_OK;
}

/**
 * @brief Sync a queue, destroy its stream, and free it.
 *
 * A null queue does nothing. The stream is left in place when the device
 * cannot be selected; the queue object is still freed.
 *
 * @param ordinal CUDA device index.
 * @param native Queue from `ga10x_queue_create`, or NULL.
 */
void ga10x_queue_destroy(int ordinal, void *native)
{
    ga10x_queue *q = (ga10x_queue *)native;

    if (!q) {
        return;
    }
    (void)ga10x_queue_sync(ordinal, q);
    if (ga10x_set(ordinal) == VM_DEVICE_OK) {
        cudaStreamDestroy(q->stream);
    }
    free(q);
}

/**
 * @brief Wait for the stream, then free retained memory.
 *
 * Pending memory is released even when the wait fails.
 *
 * @param ordinal CUDA device index.
 * @param native Queue from `ga10x_queue_create`.
 * @return Mapped `cudaStreamSynchronize` status.
 */
int ga10x_queue_sync(int ordinal, void *native)
{
    ga10x_queue *q = (ga10x_queue *)native;
    int rc = ga10x_set(ordinal);

    if (rc != VM_DEVICE_OK) {
        return rc;
    }
    rc = ga10x_map(cudaStreamSynchronize(q->stream));
    ga10x_release(q);
    return rc;
}

/**
 * @brief Queue a host-to-device copy.
 *
 * @param ordinal CUDA device index.
 * @param native Queue whose stream receives the copy.
 * @param dst Device buffer.
 * @param offset Byte offset into @p dst.
 * @param src Host pointer. Must stay valid until the queue is synced.
 * @param bytes Number of bytes.
 * @return Mapped `cudaMemcpyAsync` status.
 */
int ga10x_write(int ordinal, void *native, void *dst, size_t offset,
                const void *src, size_t bytes)
{
    ga10x_queue *q = (ga10x_queue *)native;
    int rc = ga10x_set(ordinal);
    char *dev = (char *)dst + offset;

    if (rc != VM_DEVICE_OK) {
        return rc;
    }
    return ga10x_map(cudaMemcpyAsync(dev, src, bytes, cudaMemcpyHostToDevice, q->stream));
}

/**
 * @brief Queue a device-to-host copy.
 *
 * @param ordinal CUDA device index.
 * @param native Queue whose stream receives the copy.
 * @param src Device buffer.
 * @param offset Byte offset into @p src.
 * @param dst Host pointer. Must stay valid until the queue is synced.
 * @param bytes Number of bytes.
 * @return Mapped `cudaMemcpyAsync` status.
 */
int ga10x_read(int ordinal, void *native, const void *src, size_t offset,
               void *dst, size_t bytes)
{
    ga10x_queue *q = (ga10x_queue *)native;
    int rc = ga10x_set(ordinal);
    const char *dev = (const char *)src + offset;

    if (rc != VM_DEVICE_OK) {
        return rc;
    }
    return ga10x_map(cudaMemcpyAsync((void *)dst, dev, bytes, cudaMemcpyDeviceToHost, q->stream));
}

/**
 * @brief Queue a device-to-device copy.
 *
 * Copies through a temporary so the two ranges may overlap. The temporary
 * is retained until sync. A zero byte count does nothing.
 *
 * @param ordinal CUDA device index.
 * @param native Queue whose stream receives the copy.
 * @param dst Destination device buffer.
 * @param dst_offset Byte offset into @p dst.
 * @param src Source device buffer.
 * @param src_offset Byte offset into @p src.
 * @param bytes Number of bytes.
 * @return `VM_DEVICE_OK`, or a mapped CUDA status.
 */
int ga10x_copy(int ordinal, void *native, void *dst, size_t dst_offset,
               const void *src, size_t src_offset, size_t bytes)
{
    ga10x_queue *q = (ga10x_queue *)native;
    int rc = ga10x_set(ordinal);
    void *tmp = NULL;
    char *d = (char *)dst + dst_offset;
    const char *s = (const char *)src + src_offset;

    if (rc != VM_DEVICE_OK) {
        return rc;
    }
    if (bytes == 0) {
        return VM_DEVICE_OK;
    }
    /* A temporary makes overlapping device ranges well defined. */
    rc = ga10x_map(cudaMalloc(&tmp, bytes));
    if (rc != VM_DEVICE_OK) {
        return rc;
    }
    rc = ga10x_map(cudaMemcpyAsync(tmp, s, bytes, cudaMemcpyDeviceToDevice, q->stream));
    if (rc == VM_DEVICE_OK) {
        rc = ga10x_map(cudaMemcpyAsync(d, tmp, bytes, cudaMemcpyDeviceToDevice, q->stream));
    }
    if (rc != VM_DEVICE_OK) {
        cudaFree(tmp);
        return rc;
    }
    /* cudaFree is not ordered with the stream, so the temporary stays until sync. */
    return ga10x_retain(q, tmp, 1);
}

/**
 * @brief `dst[i] = a[i] + b[i]` for `i` in `[0, n)`.
 *
 * @param dst Device output.
 * @param a Device addend.
 * @param b Device addend.
 * @param n Element count.
 */
__global__ void ga10x_add_k(float *dst, const float *a, const float *b, int n)
{
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) {
        dst[i] = a[i] + b[i];
    }
}

/**
 * @brief `dst[i] = a[i] * b[i]` for `i` in `[0, n)`.
 *
 * @param dst Device output.
 * @param a Device factor.
 * @param b Device factor.
 * @param n Element count.
 */
__global__ void ga10x_mul_k(float *dst, const float *a, const float *b, int n)
{
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) {
        dst[i] = a[i] * b[i];
    }
}

/**
 * @brief `y[i] += alpha * x[i]` for `i` in `[0, n)`.
 *
 * @param y Device vector, updated in place.
 * @param alpha Scale applied to @p x.
 * @param x Device vector.
 * @param n Element count.
 */
__global__ void ga10x_axpy_k(float *y, float alpha, const float *x, int n)
{
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) {
        y[i] += alpha * x[i];
    }
}

/**
 * @brief Launch `ga10x_add_k` on the queue stream.
 *
 * @param ordinal CUDA device index.
 * @param native Queue whose stream runs the kernel.
 * @param dst Device output.
 * @param a Device addend.
 * @param b Device addend.
 * @param n Element count.
 * @return Mapped launch status.
 */
int ga10x_add(int ordinal, void *native, float *dst, const float *a,
              const float *b, int n)
{
    ga10x_queue *q = (ga10x_queue *)native;
    int threads = 128;
    int blocks = (n + threads - 1) / threads;
    int rc = ga10x_set(ordinal);

    if (rc != VM_DEVICE_OK) {
        return rc;
    }
    ga10x_add_k<<<blocks, threads, 0, q->stream>>>(dst, a, b, n);
    return ga10x_map(cudaGetLastError());
}

/**
 * @brief Launch `ga10x_mul_k` on the queue stream.
 *
 * @param ordinal CUDA device index.
 * @param native Queue whose stream runs the kernel.
 * @param dst Device output.
 * @param a Device factor.
 * @param b Device factor.
 * @param n Element count.
 * @return Mapped launch status.
 */
int ga10x_mul(int ordinal, void *native, float *dst, const float *a,
              const float *b, int n)
{
    ga10x_queue *q = (ga10x_queue *)native;
    int threads = 128;
    int blocks = (n + threads - 1) / threads;
    int rc = ga10x_set(ordinal);

    if (rc != VM_DEVICE_OK) {
        return rc;
    }
    ga10x_mul_k<<<blocks, threads, 0, q->stream>>>(dst, a, b, n);
    return ga10x_map(cudaGetLastError());
}

/**
 * @brief Launch `ga10x_axpy_k` on the queue stream.
 *
 * @param ordinal CUDA device index.
 * @param native Queue whose stream runs the kernel.
 * @param y Device vector, updated in place.
 * @param alpha Scale applied to @p x.
 * @param x Device vector.
 * @param n Element count.
 * @return Mapped launch status.
 */
int ga10x_axpy(int ordinal, void *native, float *y, float alpha,
               const float *x, int n)
{
    ga10x_queue *q = (ga10x_queue *)native;
    int threads = 128;
    int blocks = (n + threads - 1) / threads;
    int rc = ga10x_set(ordinal);

    if (rc != VM_DEVICE_OK) {
        return rc;
    }
    ga10x_axpy_k<<<blocks, threads, 0, q->stream>>>(y, alpha, x, n);
    return ga10x_map(cudaGetLastError());
}

/**
 * @brief Launch `ga10x_gemm_tf32` on the queue stream.
 *
 * An empty tile count (m or n is zero) returns success without a launch.
 * `k == 0` is left to the kernel, which does not read A or B.
 *
 * @param ordinal CUDA device index.
 * @param native Queue whose stream runs the kernel.
 * @param C Device output, m by n, row-major.
 * @param A Device matrix, m by k, row-major.
 * @param B Device matrix, k by n, row-major.
 * @param m Rows of A and C.
 * @param n Columns of B and C.
 * @param k Inner dimension.
 * @param alpha Scale for the product.
 * @param beta Scale for the existing C.
 * @return Mapped launch status, or `VM_DEVICE_OK` when nothing is launched.
 */
int ga10x_gemm(int ordinal, void *native, float *C, const float *A,
               const float *B, int m, int n, int k, float alpha, float beta)
{
    ga10x_queue *q = (ga10x_queue *)native;
    const int tiles_m = (m + 15) / 16;
    const int tiles_n = (n + 15) / 16;
    const int tiles = tiles_m * tiles_n;
    const int warps = 4;
    const int blocks = (tiles + warps - 1) / warps;
    const size_t smem = (size_t)warps * (16 * 8 + 8 * 16 + 16 * 16) * sizeof(float);
    int rc = ga10x_set(ordinal);

    if (rc != VM_DEVICE_OK) {
        return rc;
    }
    if (tiles == 0) {
        return VM_DEVICE_OK;
    }
    ga10x_gemm_tf32<<<blocks, warps * 32, smem, q->stream>>>(
        C, A, B, m, n, k, alpha, beta);
    return ga10x_map(cudaGetLastError());
}

} /* extern "C" */
