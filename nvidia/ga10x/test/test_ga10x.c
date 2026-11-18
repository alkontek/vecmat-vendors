// SPDX-FileCopyrightText: 2025-2026 Igal Alkon
// SPDX-FileCopyrightText: 2026 ALKONTEK <git@alkontek.com>
// SPDX-License-Identifier: BSD-3-Clause

#include <string.h>

#include <vecmat.h>
#include <vecmat_nvidia_ga10x.h>

#include "unitest.h"

/* Take a device slot before other tests fill the table. A miss registers nothing. */
static void ga10x_register_early(void) __attribute__((constructor));
static void ga10x_register_early(void)
{
    (void)vm_nvidia_ga10x_register();
}

static int f32_near(float got, float exp)
{
    float d = got - exp;
    if (d < 0.0f)
        d = -d;
    return d <= 1.0e-5f;
}

/* TF32 tensor-core products are not bit-exact with vm_gemm32. */
static int f32_tf32(float got, float exp)
{
    float scale = exp < 0.0f ? -exp : exp;
    float d = got - exp;
    if (scale < 1.0f)
        scale = 1.0f;
    if (d < 0.0f)
        d = -d;
    return d <= 1.0e-2f * scale;
}

static vm_device *find_named(const char *name)
{
    const int n = vm_device_count();
    for (int i = 0; i < n; ++i) {
        vm_device_info info;
        vm_device *dev = vm_device_get(i);
        if (vm_device_info_get(dev, &info) == VM_DEVICE_OK
            && info.name && strcmp(info.name, name) == 0)
            return dev;
    }
    return NULL;
}

static void check_info(vm_device *dev)
{
    vm_device_info info;
    float host = 0.0f;
    vm_buffer *wrapped = NULL;

    REQUIRE(vm_device_info_get(dev, &info) == VM_DEVICE_OK);
    REQUIRE(info.kind == VM_DEVICE_GPU);
    REQUIRE(info.name && strcmp(info.name, "nvidia-ga10x") == 0);
    REQUIRE(info.dtypes == VM_DTYPE_BIT(VM_DTYPE_F32));
    REQUIRE((info.caps & VM_DEVICE_CAP_ASYNC) != 0);
    REQUIRE((info.caps & VM_DEVICE_CAP_HOST_MEMORY) == 0);
    REQUIRE(info.mem_bytes > 0);
    REQUIRE(vm_device_supports(dev, VM_DEVICE_OP_ARRAY_ADD, VM_DTYPE_F32) == 1);
    REQUIRE(vm_device_supports(dev, VM_DEVICE_OP_ARRAY_MUL, VM_DTYPE_F32) == 1);
    REQUIRE(vm_device_supports(dev, VM_DEVICE_OP_AXPY, VM_DTYPE_F32) == 1);
    REQUIRE(vm_device_supports(dev, VM_DEVICE_OP_GEMM, VM_DTYPE_F32) == 1);
    REQUIRE(vm_device_supports(dev, VM_DEVICE_OP_GEMM, VM_DTYPE_F64) == 0);
    REQUIRE(vm_device_supports(dev, VM_DEVICE_OP_ARRAY_ADD, VM_DTYPE_F64) == 0);
    REQUIRE(vm_buffer_wrap(dev, &host, sizeof host, &wrapped) == VM_DEVICE_ERR_UNSUPPORTED);
}

static void check_against_cpu(vm_device *gpu)
{
    vm_device *cpu = vm_device_get(0);
    vm_queue *gq = NULL;
    vm_queue *cq = NULL;
    vm_buffer *ga = NULL, *gb = NULL, *gc = NULL, *gy = NULL;
    vm_buffer *ca = NULL, *cb = NULL, *cc = NULL, *cy = NULL;
    float a[8], b[8], y[8], got[8], exp[8];
    float A[12], B[15], C[10], Cg[10], Cc[10];
    float gout[10], eout[10];
    const int m = 2, n = 5, k = 3;

    REQUIRE(cpu != NULL);
    for (int i = 0; i < 8; ++i) {
        a[i] = (float)i - 1.5f;
        b[i] = 0.25f * (float)(i + 1);
        y[i] = 1.0f;
    }
    REQUIRE(vm_queue_create(gpu, &gq) == VM_DEVICE_OK);
    REQUIRE(vm_queue_create(cpu, &cq) == VM_DEVICE_OK);
    REQUIRE(vm_buffer_create(gpu, sizeof a, &ga) == VM_DEVICE_OK);
    REQUIRE(vm_buffer_create(gpu, sizeof b, &gb) == VM_DEVICE_OK);
    REQUIRE(vm_buffer_create(gpu, sizeof y, &gy) == VM_DEVICE_OK);
    REQUIRE(vm_buffer_create(cpu, sizeof a, &ca) == VM_DEVICE_OK);
    REQUIRE(vm_buffer_create(cpu, sizeof b, &cb) == VM_DEVICE_OK);
    REQUIRE(vm_buffer_create(cpu, sizeof y, &cy) == VM_DEVICE_OK);

    REQUIRE(vm_queue_write(gq, ga, 0, a, sizeof a) == VM_DEVICE_OK);
    REQUIRE(vm_queue_write(gq, gb, 0, b, sizeof b) == VM_DEVICE_OK);
    REQUIRE(vm_queue_write(gq, gy, 0, y, sizeof y) == VM_DEVICE_OK);
    REQUIRE(vm_queue_finish(gq) == VM_DEVICE_OK);
    REQUIRE(vm_queue_read(gq, ga, 0, got, sizeof got) == VM_DEVICE_OK);
    REQUIRE(vm_queue_finish(gq) == VM_DEVICE_OK);
    for (int i = 0; i < 8; ++i)
        REQUIRE(f32_near(got[i], a[i]));

    REQUIRE(vm_queue_write(cq, ca, 0, a, sizeof a) == VM_DEVICE_OK);
    REQUIRE(vm_queue_write(cq, cb, 0, b, sizeof b) == VM_DEVICE_OK);
    REQUIRE(vm_queue_write(cq, cy, 0, y, sizeof y) == VM_DEVICE_OK);
    REQUIRE(vm_queue_array_add(gq, VM_DTYPE_F32, gy, ga, gb, 8) == VM_DEVICE_OK);
    REQUIRE(vm_queue_array_add(cq, VM_DTYPE_F32, cy, ca, cb, 8) == VM_DEVICE_OK);
    REQUIRE(vm_queue_read(gq, gy, 0, got, sizeof got) == VM_DEVICE_OK);
    REQUIRE(vm_queue_read(cq, cy, 0, exp, sizeof exp) == VM_DEVICE_OK);
    REQUIRE(vm_queue_finish(gq) == VM_DEVICE_OK);
    REQUIRE(vm_queue_finish(cq) == VM_DEVICE_OK);
    for (int i = 0; i < 8; ++i)
        REQUIRE(f32_near(got[i], exp[i]));

    for (int i = 0; i < 12; ++i)
        A[i] = 0.5f * (float)(i - 3);
    for (int i = 0; i < 15; ++i)
        B[i] = 0.25f * (float)(i + 1);
    for (int i = 0; i < 10; ++i)
        C[i] = 2.0f;
    memcpy(Cg, C, sizeof C);
    memcpy(Cc, C, sizeof C);

    vm_buffer_destroy(ga);
    vm_buffer_destroy(gb);
    vm_buffer_destroy(gy);
    vm_buffer_destroy(ca);
    vm_buffer_destroy(cb);
    vm_buffer_destroy(cy);
    ga = gb = gy = ca = cb = cy = NULL;
    REQUIRE(vm_buffer_create(gpu, sizeof A, &ga) == VM_DEVICE_OK);
    REQUIRE(vm_buffer_create(gpu, sizeof B, &gb) == VM_DEVICE_OK);
    REQUIRE(vm_buffer_create(gpu, sizeof C, &gc) == VM_DEVICE_OK);
    REQUIRE(vm_buffer_create(cpu, sizeof A, &ca) == VM_DEVICE_OK);
    REQUIRE(vm_buffer_create(cpu, sizeof B, &cb) == VM_DEVICE_OK);
    REQUIRE(vm_buffer_create(cpu, sizeof C, &cc) == VM_DEVICE_OK);
    REQUIRE(vm_queue_write(gq, ga, 0, A, sizeof A) == VM_DEVICE_OK);
    REQUIRE(vm_queue_write(gq, gb, 0, B, sizeof B) == VM_DEVICE_OK);
    REQUIRE(vm_queue_write(gq, gc, 0, Cg, sizeof Cg) == VM_DEVICE_OK);
    REQUIRE(vm_queue_write(cq, ca, 0, A, sizeof A) == VM_DEVICE_OK);
    REQUIRE(vm_queue_write(cq, cb, 0, B, sizeof B) == VM_DEVICE_OK);
    REQUIRE(vm_queue_write(cq, cc, 0, Cc, sizeof Cc) == VM_DEVICE_OK);
    REQUIRE(vm_queue_gemm(gq, VM_DTYPE_F32, m, n, k, 1.5, ga, gb, 0.5, gc) == VM_DEVICE_OK);
    REQUIRE(vm_queue_gemm(cq, VM_DTYPE_F32, m, n, k, 1.5, ca, cb, 0.5, cc) == VM_DEVICE_OK);
    REQUIRE(vm_queue_read(gq, gc, 0, gout, sizeof gout) == VM_DEVICE_OK);
    REQUIRE(vm_queue_read(cq, cc, 0, eout, sizeof eout) == VM_DEVICE_OK);
    REQUIRE(vm_queue_finish(gq) == VM_DEVICE_OK);
    REQUIRE(vm_queue_finish(cq) == VM_DEVICE_OK);
    for (int i = 0; i < 10; ++i)
        REQUIRE(f32_tf32(gout[i], eout[i]));

    /* k == 0 scales C and does not read A or B. */
    for (int i = 0; i < 10; ++i)
        Cg[i] = 4.0f;
    REQUIRE(vm_queue_write(gq, gc, 0, Cg, sizeof Cg) == VM_DEVICE_OK);
    REQUIRE(vm_queue_gemm(gq, VM_DTYPE_F32, m, n, 0, 1.0, ga, gb, 0.25, gc) == VM_DEVICE_OK);
    REQUIRE(vm_queue_read(gq, gc, 0, gout, sizeof gout) == VM_DEVICE_OK);
    REQUIRE(vm_queue_finish(gq) == VM_DEVICE_OK);
    for (int i = 0; i < 10; ++i)
        REQUIRE(f32_near(gout[i], 1.0f));

    vm_buffer_destroy(ga);
    vm_buffer_destroy(gb);
    vm_buffer_destroy(gc);
    vm_buffer_destroy(ca);
    vm_buffer_destroy(cb);
    vm_buffer_destroy(cc);
    vm_buffer_destroy(cy);
    vm_queue_destroy(gq);
    vm_queue_destroy(cq);
}

TEST_CASE(nvidia_ga10x_registers_or_skips, "[ga10x]")
{
    vm_device *dev;
    const int rc = vm_nvidia_ga10x_register();

    if (rc == VM_DEVICE_ERR_UNSUPPORTED)
        return;

    REQUIRE(rc == VM_DEVICE_OK);
    REQUIRE(vm_nvidia_ga10x_register() == VM_DEVICE_OK);
    dev = find_named("nvidia-ga10x");
    REQUIRE(dev != NULL);
    check_info(dev);
    check_against_cpu(dev);
}
