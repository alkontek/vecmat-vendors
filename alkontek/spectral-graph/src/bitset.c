// SPDX-FileCopyrightText: 2025-2026 Igal Alkon
// SPDX-FileCopyrightText: 2026 ALKONTEK <git@alkontek.com>
// SPDX-License-Identifier: BSD-3-Clause

#include <stdlib.h>
#include <string.h>
#include <vecmat/feature/spectral_graph.h>

static int words_for(const int nbits)
{
    return (nbits + 63) / 64;
}

bool vm_bitset_alloc(vm_bitset *bs, const int nbits)
{
    if (!bs || nbits <= 0) {
        return false;
    }

    const int nwords = words_for(nbits);
    uint64_t *words = calloc((size_t)nwords, sizeof(uint64_t));
    if (!words) {
        return false;
    }

    bs->words = words;
    bs->nbits = nbits;
    bs->nwords = nwords;

    return true;
}

void vm_bitset_free(vm_bitset *bs)
{
    if (!bs) {
        return;
    }

    free(bs->words);
    bs->words = NULL;
    bs->nbits = 0;
    bs->nwords = 0;
}

void vm_bitset_set(vm_bitset *bs, const int i)
{
    if (!bs || !bs->words || i < 0 || i >= bs->nbits) {
        return;
    }

    bs->words[i >> 6] |= (uint64_t)1 << (i & 63);
}

void vm_bitset_clear(vm_bitset *bs, const int i)
{
    if (!bs || !bs->words || i < 0 || i >= bs->nbits) {
        return;
    }

    bs->words[i >> 6] &= ~((uint64_t)1 << (i & 63));
}

bool vm_bitset_test(const vm_bitset *bs, const int i)
{
    if (!bs || !bs->words || i < 0 || i >= bs->nbits) {
        return false;
    }

    return (bs->words[i >> 6] >> (i & 63)) & 1;
}

static void apply(vm_bitset *dst, const vm_bitset *a, const vm_bitset *b, const int op)
{
    if (!dst || !a || !b || !dst->words || !a->words || !b->words) {
        return;
    }

    const int n = dst->nwords < a->nwords ? dst->nwords : a->nwords;
    for (int i = 0; i < n; ++i)
    {
        const uint64_t bv = (i < b->nwords) ? b->words[i] : 0;
        if (op == 0) {
            dst->words[i] = a->words[i] & bv;
        } else if (op == 1) {
            dst->words[i] = a->words[i] | bv;
        } else {
            dst->words[i] = a->words[i] ^ bv;
        }
    }
}

void vm_bitset_and(vm_bitset *dst, const vm_bitset *a, const vm_bitset *b)
{
    apply(dst, a, b, 0);
}

void vm_bitset_or(vm_bitset *dst, const vm_bitset *a, const vm_bitset *b)
{
    apply(dst, a, b, 1);
}

void vm_bitset_xor(vm_bitset *dst, const vm_bitset *a, const vm_bitset *b)
{
    apply(dst, a, b, 2);
}

int vm_bitset_popcount(const vm_bitset *bs)
{
    if (!bs || !bs->words) {
        return 0;
    }

    int count = 0;
    for (int i = 0; i < bs->nwords; ++i) {
        uint64_t w = bs->words[i];
        while (w) {
            w &= w - 1;
            ++count;
        }
    }

    return count;
}
