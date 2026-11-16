import itertools

import torch
import triton
import triton.language as tl


def get_configs():
    # One warp. Extra warps re-execute this serial frontier and fault on the longer streams.
    return [triton.Config({}, num_warps=1)]


@triton.autotune(
    configs=get_configs(),
    key=["T", "NS"],
    cache_results=True,
    restore_value=["activation_counts", "report_counts"],
)
@triton.jit(
    do_not_specialize=[
        "comp_ptr",
        "row_ptr",
        "col_idx",
        "symbol_cols",
        "is_report",
        "start_ptr",
        "start_idx",
        "start_sod",
        "stream",
        "activation_counts",
        "report_counts",
        "C",
        "NS",
        "T",
        "NE",
        "NSTART",
    ]
)
def _simulate_component(
    comp_ptr,
    row_ptr,
    col_idx,
    symbol_cols,
    is_report,
    start_ptr,
    start_idx,
    start_sod,
    stream,
    enabled,
    frontier,
    matched,
    activation_counts,
    report_counts,
    C,
    NS,
    T,
    NE,
    NSTART,
):
    c = tl.program_id(0)
    if c >= C:
        return

    c_ok = (c >= 0) & (c < C)
    base = tl.load(comp_ptr + tl.where(c_ok, c, 0), mask=c_ok, other=0)
    comp_end = tl.load(comp_ptr + tl.where(c_ok, c + 1, 0), mask=c_ok & (c + 1 <= C), other=0)
    first_start = tl.load(start_ptr + tl.where(c_ok, c, 0), mask=c_ok, other=0)
    last_start = tl.load(
        start_ptr + tl.where(c_ok, c + 1, 0),
        mask=c_ok & (c + 1 <= C),
        other=0,
    )
    width = tl.maximum(comp_end - base, 0)

    n_front = 0
    for k in tl.range(first_start, last_start):
        k_ok = (k >= 0) & (k < NSTART) & (k < last_start)
        k_i = tl.where(k_ok, k, 0)
        s = tl.load(start_idx + k_i, mask=k_ok, other=0)
        s_ok = k_ok & (s >= 0) & (s < NS)
        s_i = tl.where(s_ok, s, 0)
        if s_ok & (tl.load(enabled + s_i, mask=s_ok, other=1) == 0):
            tl.store(enabled + s_i, 1, mask=s_ok)
            widx = base + n_front
            w_ok = s_ok & (widx >= 0) & (widx < NS) & (widx < comp_end)
            w_i = tl.where(w_ok, widx, 0)
            tl.store(frontier + w_i, s, mask=w_ok)
            n_front = n_front + tl.where(w_ok, 1, 0)

    reports = 0
    for t in tl.range(T):
        t_ok = (t >= 0) & (t < T)
        sym = tl.load(stream + tl.where(t_ok, t, 0), mask=t_ok, other=0)
        eod = 0
        if t == T - 1:
            eod = 1
        elif sym == 10:
            eod = 1

        n_match = 0
        n_front_cap = tl.minimum(n_front, width)
        for k in tl.range(n_front_cap):
            fk = base + k
            fk_ok = (k >= 0) & (k < n_front) & (fk >= 0) & (fk < NS) & (fk < comp_end)
            fk_i = tl.where(fk_ok, fk, 0)
            s = tl.load(frontier + fk_i, mask=fk_ok, other=0)
            s_ok = fk_ok & (s >= 0) & (s < NS)
            s_i = tl.where(s_ok, s, 0)
            sym_ok = (sym >= 0) & (sym < 256)
            sym_i = tl.where(sym_ok, sym, 0)
            sym_hit = tl.load(
                symbol_cols + s_i * 256 + sym_i,
                mask=s_ok & sym_ok,
                other=0,
            )
            if s_ok & sym_ok & (sym_hit != 0):
                midx = base + n_match
                m_ok = s_ok & (midx >= 0) & (midx < NS) & (midx < comp_end)
                m_i = tl.where(m_ok, midx, 0)
                tl.store(matched + m_i, s, mask=m_ok)
                n_match = n_match + tl.where(m_ok, 1, 0)
                tl.atomic_add(activation_counts + s_i, 1, mask=s_ok, sem="relaxed")
                if tl.load(is_report + s_i, mask=s_ok, other=0) != 0:
                    reports += 1
            tl.store(enabled + s_i, 0, mask=s_ok)

        n_front = 0
        n_match_cap = tl.minimum(n_match, width)
        for k in tl.range(n_match_cap):
            mk = base + k
            mk_ok = (k >= 0) & (k < n_match) & (mk >= 0) & (mk < NS) & (mk < comp_end)
            mk_i = tl.where(mk_ok, mk, 0)
            s = tl.load(matched + mk_i, mask=mk_ok, other=0)
            s_ok = mk_ok & (s >= 0) & (s < NS)
            s_i = tl.where(s_ok, s, 0)
            row_b = tl.load(row_ptr + s_i, mask=s_ok, other=0)
            row_e = tl.load(row_ptr + s_i + 1, mask=s_ok & (s_i + 1 <= NS), other=0)
            row_b = tl.maximum(row_b, 0)
            row_e = tl.minimum(tl.maximum(row_e, row_b), NE)
            for e in tl.range(row_b, row_e):
                e_ok = (e >= 0) & (e < NE) & (e >= row_b) & (e < row_e)
                e_i = tl.where(e_ok, e, 0)
                child = tl.load(col_idx + e_i, mask=e_ok, other=0)
                child_ok = e_ok & (child >= 0) & (child < NS)
                child_i = tl.where(child_ok, child, 0)
                if child_ok & (tl.load(enabled + child_i, mask=child_ok, other=1) == 0):
                    tl.store(enabled + child_i, 1, mask=child_ok)
                    widx = base + n_front
                    w_ok = child_ok & (widx >= 0) & (widx < NS) & (widx < comp_end)
                    w_i = tl.where(w_ok, widx, 0)
                    tl.store(frontier + w_i, child, mask=w_ok)
                    n_front = n_front + tl.where(w_ok, 1, 0)

        for k in tl.range(first_start, last_start):
            k_ok = (k >= 0) & (k < NSTART) & (k < last_start)
            k_i = tl.where(k_ok, k, 0)
            sod = tl.load(start_sod + k_i, mask=k_ok, other=0)
            if k_ok & ((sod == 0) | (eod == 1)):
                s = tl.load(start_idx + k_i, mask=k_ok, other=0)
                s_ok = k_ok & (s >= 0) & (s < NS)
                s_i = tl.where(s_ok, s, 0)
                if s_ok & (tl.load(enabled + s_i, mask=s_ok, other=1) == 0):
                    tl.store(enabled + s_i, 1, mask=s_ok)
                    widx = base + n_front
                    w_ok = s_ok & (widx >= 0) & (widx < NS) & (widx < comp_end)
                    w_i = tl.where(w_ok, widx, 0)
                    tl.store(frontier + w_i, s, mask=w_ok)
                    n_front = n_front + tl.where(w_ok, 1, 0)

    tl.store(report_counts + tl.where(c_ok, c, 0), reports, mask=c_ok)


def nfa_frontier(
    comp_ptr,
    row_ptr,
    col_idx,
    symbol_cols,
    is_report,
    start_ptr,
    start_idx,
    start_sod,
    stream,
    activation_counts,
    report_counts,
    C,
    NS,
    T,
):
    comp_ptr = comp_ptr.contiguous()
    row_ptr = row_ptr.contiguous()
    col_idx = col_idx.contiguous()
    symbol_cols = symbol_cols.contiguous()
    is_report = is_report.contiguous()
    start_ptr = start_ptr.contiguous()
    start_idx = start_idx.contiguous()
    start_sod = start_sod.contiguous()
    stream = stream.contiguous()

    n_c = int(C)
    n_ns = int(NS)
    n_t = int(T)
    n_ne = int(col_idx.numel())
    n_start = int(start_idx.numel())

    enabled = torch.zeros(n_ns, dtype=torch.int64, device=comp_ptr.device)
    frontier = torch.zeros(n_ns, dtype=torch.int64, device=comp_ptr.device)
    matched = torch.zeros(n_ns, dtype=torch.int64, device=comp_ptr.device)

    _simulate_component[(n_c,)](
        comp_ptr,
        row_ptr,
        col_idx,
        symbol_cols,
        is_report,
        start_ptr,
        start_idx,
        start_sod,
        stream,
        enabled,
        frontier,
        matched,
        activation_counts,
        report_counts,
        n_c,
        n_ns,
        n_t,
        n_ne,
        n_start,
    )
