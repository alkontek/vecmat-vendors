// SPDX-FileCopyrightText: 2025-2026 Igal Alkon
// SPDX-FileCopyrightText: 2026 ALKONTEK <git@alkontek.com>
// SPDX-License-Identifier: BSD-3-Clause

#ifndef VECMAT_MD_LIST_EXCL_H
#define VECMAT_MD_LIST_EXCL_H

/**
 * @brief True when neighbor `j` is listed on this exclusion row.
 *
 * `excl_offsets == NULL` means the caller supplied no exclusions. An empty
 * row does not read `excl_partners`. The partner's own row is not searched.
 */
static inline int md_list_excluded(const int *excl_offsets, const int *excl_partners,
                                   const int row, const int j)
{
    if (!excl_offsets) {
        return 0;
    }

    for (int p = excl_offsets[row]; p < excl_offsets[row + 1]; ++p) {
        if (excl_partners[p] == j) {
            return 1;
        }
    }

    return 0;
}

#endif /* VECMAT_MD_LIST_EXCL_H */
