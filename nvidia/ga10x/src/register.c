// SPDX-FileCopyrightText: 2025-2026 Igal Alkon
// SPDX-FileCopyrightText: 2026 ALKONTEK <git@alkontek.com>
// SPDX-License-Identifier: BSD-3-Clause

#include <stddef.h>

#include <vecmat_nvidia_ga10x.h>

/**
 * CUDA entry points implemented in ga10x.cu. This file stays C and does not
 * include a CUDA header. The native queue pointer is owned by ga10x.cu.
 */
int ga10x_probe(int *ordinal, size_t *mem_bytes);
int ga10x_buffer_create(int ordinal, size_t bytes, void **native);
void ga10x_buffer_destroy(int ordinal, void *native);
int ga10x_queue_create(int ordinal, void **native);
void ga10x_queue_destroy(int ordinal, void *native);
int ga10x_queue_sync(int ordinal, void *native);
int ga10x_write(int ordinal, void *native, void *dst, size_t offset,
                const void *src, size_t bytes);
int ga10x_read(int ordinal, void *native, const void *src, size_t offset,
               void *dst, size_t bytes);
int ga10x_copy(int ordinal, void *native, void *dst, size_t dst_offset,
               const void *src, size_t src_offset, size_t bytes);
int ga10x_add(int ordinal, void *native, float *dst, const float *a,
              const float *b, int n);
int ga10x_mul(int ordinal, void *native, float *dst, const float *a,
              const float *b, int n);
int ga10x_axpy(int ordinal, void *native, float *y, float alpha,
               const float *x, int n);
int ga10x_gemm(int ordinal, void *native, float *C, const float *A,
               const float *B, int m, int n, int k, float alpha, float beta);

typedef struct ga10x_ctx {
    int ordinal;
    size_t mem_bytes;
} ga10x_ctx;

static ga10x_ctx ga10x_state;
static int ga10x_have;

static void ga10x_info(void *ctx, vm_device_info *out)
{
    const ga10x_ctx *g = (const ga10x_ctx *)ctx;

    out->kind = VM_DEVICE_GPU;
    out->name = "nvidia-ga10x";
    out->dtypes = VM_DTYPE_BIT(VM_DTYPE_F32);
    out->caps = VM_DEVICE_CAP_ASYNC;
    out->mem_bytes = g->mem_bytes;
}

static int ga10x_supports(void *ctx, vm_device_op op, vm_dtype dt)
{
    (void)ctx;
    if (dt != VM_DTYPE_F32) {
        return 0;
    }
    return op == VM_DEVICE_OP_ARRAY_ADD || op == VM_DEVICE_OP_ARRAY_MUL
        || op == VM_DEVICE_OP_AXPY || op == VM_DEVICE_OP_GEMM;
}

static int ga10x_buf_create(void *ctx, size_t bytes, void **native)
{
    const ga10x_ctx *g = (const ga10x_ctx *)ctx;
    return ga10x_buffer_create(g->ordinal, bytes, native);
}

static void ga10x_buf_destroy(void *ctx, void *native, int wrapped)
{
    const ga10x_ctx *g = (const ga10x_ctx *)ctx;
    if (!wrapped) {
        ga10x_buffer_destroy(g->ordinal, native);
    }
}

static int ga10x_q_create(void *ctx, void **native)
{
    const ga10x_ctx *g = (const ga10x_ctx *)ctx;
    return ga10x_queue_create(g->ordinal, native);
}

static void ga10x_q_destroy(void *ctx, void *native)
{
    const ga10x_ctx *g = (const ga10x_ctx *)ctx;
    ga10x_queue_destroy(g->ordinal, native);
}

static int ga10x_q_write(void *ctx, void *q, void *dst, size_t offset,
                         const void *src, size_t bytes)
{
    const ga10x_ctx *g = (const ga10x_ctx *)ctx;
    return ga10x_write(g->ordinal, q, dst, offset, src, bytes);
}

static int ga10x_q_read(void *ctx, void *q, const void *src, size_t offset,
                        void *dst, size_t bytes)
{
    const ga10x_ctx *g = (const ga10x_ctx *)ctx;
    return ga10x_read(g->ordinal, q, src, offset, dst, bytes);
}

static int ga10x_q_copy(void *ctx, void *q, void *dst, size_t dst_offset,
                        const void *src, size_t src_offset, size_t bytes)
{
    const ga10x_ctx *g = (const ga10x_ctx *)ctx;
    return ga10x_copy(g->ordinal, q, dst, dst_offset, src, src_offset, bytes);
}

static int ga10x_submit(void *ctx, void *q, const vm_device_cmd *cmd)
{
    const ga10x_ctx *g = (const ga10x_ctx *)ctx;
    float *dst = (float *)cmd->dst;
    const float *a = (const float *)cmd->a;
    const float *b = (const float *)cmd->b;

    switch (cmd->op) {
    case VM_DEVICE_OP_ARRAY_ADD:
        return ga10x_add(g->ordinal, q, dst, a, b, cmd->n);
    case VM_DEVICE_OP_ARRAY_MUL:
        return ga10x_mul(g->ordinal, q, dst, a, b, cmd->n);
    case VM_DEVICE_OP_AXPY:
        return ga10x_axpy(g->ordinal, q, dst, (float)cmd->alpha, a, cmd->n);
    case VM_DEVICE_OP_GEMM:
        return ga10x_gemm(g->ordinal, q, dst, a, b, cmd->m, cmd->n, cmd->k,
                          (float)cmd->alpha, (float)cmd->beta);
    default:
        return VM_DEVICE_ERR_UNSUPPORTED;
    }
}

static int ga10x_finish(void *ctx, void *q)
{
    const ga10x_ctx *g = (const ga10x_ctx *)ctx;
    return ga10x_queue_sync(g->ordinal, q);
}

static const vm_device_ops ga10x_ops = {
    sizeof(vm_device_ops), VM_DEVICE_OPS_VERSION,
    ga10x_info, ga10x_supports,
    ga10x_buf_create, NULL, ga10x_buf_destroy,
    ga10x_q_create, ga10x_q_destroy,
    ga10x_q_write, ga10x_q_read, ga10x_q_copy, ga10x_submit, ga10x_finish
};

int vm_nvidia_ga10x_register(void)
{
    int ordinal = 0;
    size_t mem_bytes = 0;

    if (ga10x_have) {
        return vm_device_register(&ga10x_ops, &ga10x_state);
    }

    int rc = ga10x_probe(&ordinal, &mem_bytes);
    if (rc != VM_DEVICE_OK) {
        return rc;
    }

    ga10x_state.ordinal = ordinal;
    ga10x_state.mem_bytes = mem_bytes;
    ga10x_have = 1;
    return vm_device_register(&ga10x_ops, &ga10x_state);
}
