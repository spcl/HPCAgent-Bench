# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Boolean SpGEMM (OR, AND) in Triton.

Row i of C is the sorted set of columns reachable through row i of A. The phases follow
upstream's structure, all on the GPU:

1. ``_bound_bin_kernel``: per row the raw product count ``sum_j nnz(B[j])``, and the row is
   appended to the bin of the smallest power-of-two table (32, 64, ...) that holds it, with an
   atomic counter like upstream (order inside a bin does not affect the result);
2. ``_row_kernel`` (symbolic pass, one launch per non-empty bin): the row's product columns
   are gathered into a register vector of the bin's size (padded with a sentinel), sorted with
   ``tl.sort`` (a bitonic network, as upstream sorts its tables), and the first occurrence of
   each value is counted;
3. exclusive scan of the counts into ``C_indptr``;
4. ``_row_kernel`` (numeric pass): the same sort, and the unique values are compacted with a
   cumulative sum into ``C_indices`` -- each row comes out sorted ascending.

Comparing a sorted value with its left neighbour goes through a small per-program scratch
buffer (a register tensor cannot be shifted).

NOTE: written without a GPU. To be tested.
"""

import torch
import triton
import triton.language as tl

_NCLASS = 20  # tables 32 .. 32 << 19


@triton.jit
def _bound_bin_kernel(a_ptr, a_idx, b_ptr, raw, cls_cnt, bins, M, NCLASS: tl.constexpr):
    i = tl.program_id(0)
    s = tl.load(a_ptr + i)
    e = tl.load(a_ptr + i + 1)
    tot = tl.zeros((), dtype=tl.int64)
    for k in range(s, e):
        j = tl.load(a_idx + k)
        tot += (tl.load(b_ptr + j + 1) - tl.load(b_ptr + j)).to(tl.int64)
    tl.store(raw + i, tot)
    # smallest c with tot <= 32 << c
    c = tl.zeros((), dtype=tl.int32)
    cap = tl.full((), 32, tl.int64)
    for _ in tl.static_range(NCLASS - 1):
        grow = tot > cap
        c += grow.to(tl.int32)
        cap = tl.where(grow, cap * 2, cap)
    has = tot > 0
    slot = tl.atomic_add(cls_cnt + c, 1, mask=has)
    tl.store(bins + c.to(tl.int64) * M + slot, i, mask=has)


@triton.jit
def _row_kernel(a_ptr, a_idx, b_ptr, b_idx, bins_c, scratch, cnt, c_ptr, c_idx,
                TS: tl.constexpr, NUMERIC: tl.constexpr):
    pid = tl.program_id(0)
    row = tl.load(bins_c + pid)
    t = tl.arange(0, TS)
    cand = tl.full((TS,), 2**31 - 1, tl.int32)
    s = tl.load(a_ptr + row)
    e = tl.load(a_ptr + row + 1)
    off = tl.zeros((), dtype=tl.int64)
    for k in range(s, e):
        j = tl.load(a_idx + k)
        bs = tl.load(b_ptr + j)
        bl = tl.load(b_ptr + j + 1) - bs
        inr = (t >= off) & (t < off + bl)
        col = tl.load(b_idx + bs + (t - off), mask=inr, other=0)
        cand = tl.where(inr, col.to(tl.int32), cand)
        off += bl
    v = tl.sort(cand)
    sc = scratch + pid.to(tl.int64) * TS
    tl.store(sc + t, v)
    tl.debug_barrier()
    prev = tl.load(sc + t - 1, mask=t > 0, other=-1)
    uniq = (v != 2**31 - 1) & ((t == 0) | (v != prev))
    if NUMERIC:
        base = tl.load(c_ptr + row)
        rank = tl.cumsum(uniq.to(tl.int32), axis=0) - 1
        tl.store(c_idx + base + rank, v.to(c_idx.dtype.element_ty), mask=uniq)
    else:
        tl.store(cnt + row, tl.sum(uniq.to(tl.int64), axis=0))


@triton.jit
def _excl_scan_kernel(cnt_ptr, out_ptr, n, BLOCK: tl.constexpr):
    carry = tl.zeros((), dtype=tl.int64)
    tl.store(out_ptr, carry.to(out_ptr.dtype.element_ty))
    for s in range(0, n, BLOCK):
        o = s + tl.arange(0, BLOCK)
        c = tl.load(cnt_ptr + o, mask=o < n, other=0).to(tl.int64)
        cs = tl.cumsum(c, axis=0) + carry
        tl.store(out_ptr + o + 1, cs.to(out_ptr.dtype.element_ty), mask=o < n)
        carry += tl.sum(c, axis=0)


def spgemm_hash(A_indices, A_indptr, B_indices, B_indptr, N, M, C_indices, C_indptr):
    M = int(M)
    _ = int(N)
    dev = C_indptr.device

    def t(x):
        return torch.as_tensor(x, device=dev).contiguous()

    a_idx, a_ptr, b_idx, b_ptr = t(A_indices), t(A_indptr), t(B_indices), t(B_indptr)
    raw = torch.empty(M, dtype=torch.int64, device=dev)
    cls_cnt = torch.zeros(_NCLASS, dtype=torch.int32, device=dev)
    bins = torch.empty(_NCLASS * M, dtype=torch.int32, device=dev)
    _bound_bin_kernel[(M,)](a_ptr, a_idx, b_ptr, raw, cls_cnt, bins, M, NCLASS=_NCLASS)
    counts = cls_cnt.tolist()

    row_nnz = torch.zeros(M, dtype=torch.int64, device=dev)
    for numeric in (False, True):
        if numeric:
            _excl_scan_kernel[(1,)](row_nnz, C_indptr, M, BLOCK=1024)
        for c, n_rows in enumerate(counts):
            if n_rows == 0:
                continue
            TS = 32 << c
            scratch = torch.empty(n_rows * TS, dtype=torch.int32, device=dev)
            nw = max(1, min(8, TS // 256))
            _row_kernel[(n_rows,)](
                a_ptr, a_idx, b_ptr, b_idx, bins[c * M : (c + 1) * M], scratch, row_nnz,
                C_indptr, C_indices, TS=TS, NUMERIC=numeric, num_warps=nw,
            )
