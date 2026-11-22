import math

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


_AX = 0.9847450218426965
_GAMMA, _B1, _B2 = -0.1423, 1.0529, 0.3334
_A, _B, _C, _D = 0.0311, -0.0480, 0.0020, -0.0116

# Python floats folded the way NumPy sees them, then stored in the run dtype (NEP 50).
_CONSTS = (
    -0.75 * _AX,
    -_AX,
    _GAMMA,
    _B1,
    _B2,
    _A,
    _B,
    _C,
    _D,
    math.pi,
    1.0e-12,
    (7.0 / 6.0) * _B1,
    (4.0 / 3.0) * _B2,
    _B - _A / 3.0,
    (2.0 / 3.0) * _C,
    (2.0 * _D - _C) / 3.0,
    1.0 / 3.0,
)

# Partials are sized for this tile. A larger autotune block writes a shorter prefix.
_MIN_BLOCK = 128


def _map_configs():
    return [
        triton.Config({"BLOCK": 1024}, num_warps=8),
        triton.Config({"BLOCK": 512}, num_warps=4),
        triton.Config({"BLOCK": 256}, num_warps=4),
        triton.Config({"BLOCK": 128}, num_warps=2),
        triton.Config({"BLOCK": 1024}, num_warps=4),
        triton.Config({"BLOCK": 256}, num_warps=8),
    ]


def _tl_float(dtype: torch.dtype):
    if dtype == torch.float64:
        return tl.float64
    if dtype == torch.float32:
        return tl.float32
    raise TypeError(f"lda_xc_potential expects float32 or float64, got {dtype}")


# restore_value: partial is overwritten per trial; the tail past a large block must stay zero.
@triton.autotune(configs=_map_configs(), key=["n"], cache_results=True, restore_value=["partial"])
@triton.jit
def _map(rho, vxc, partial, consts, n, DTYPE: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    rho_v = tl.load(rho + offs, mask=mask, other=0)

    eps_scale = tl.load(consts + 0)
    vx_scale = tl.load(consts + 1)
    gamma = tl.load(consts + 2)
    b1 = tl.load(consts + 3)
    b2 = tl.load(consts + 4)
    a = tl.load(consts + 5)
    b = tl.load(consts + 6)
    c = tl.load(consts + 7)
    d = tl.load(consts + 8)
    pi = tl.load(consts + 9)
    floor = tl.load(consts + 10)
    seven_six_b1 = tl.load(consts + 11)
    four_three_b2 = tl.load(consts + 12)
    b_minus = tl.load(consts + 13)
    two_third_c = tl.load(consts + 14)
    two_d_c = tl.load(consts + 15)
    third = tl.load(consts + 16)
    one = tl.full((), 1, DTYPE)
    three = tl.full((), 3, DTYPE)
    four = tl.full((), 4, DTYPE)

    dens = tl.where(rho_v > floor, rho_v, floor)
    rs = libdevice.pow(three / (four * pi * dens), third)
    n13 = libdevice.pow(dens, third)
    eps_x = eps_scale * n13
    v_x = vx_scale * n13

    sqrt_rs = libdevice.sqrt(rs)
    ln_rs = libdevice.log(rs)
    denom = one + b1 * sqrt_rs + b2 * rs
    eps_c_ge1 = gamma / denom
    inner = one + seven_six_b1 * sqrt_rs + four_three_b2 * rs
    v_c_ge1 = eps_c_ge1 * inner / denom
    eps_c_lt1 = a * ln_rs + b + c * rs * ln_rs + d * rs
    v_c_lt1 = a * ln_rs + b_minus + two_third_c * rs * ln_rs + two_d_c * rs
    high = rs < one
    eps_c = tl.where(high, eps_c_lt1, eps_c_ge1)
    v_c = tl.where(high, v_c_lt1, v_c_ge1)

    tl.store(vxc + offs, v_x + v_c, mask=mask)
    contrib = tl.where(mask, (eps_x + eps_c) * dens, tl.zeros((), dtype=DTYPE))
    tl.store(partial + pid, tl.sum(contrib))


@triton.jit
def _reduce_partial(src, dst, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(dst + pid, tl.sum(tl.load(src + offs, mask=mask, other=0)))


def _tree_sum(buf: torch.Tensor) -> torch.Tensor:
    block = 1024
    cur = buf
    while cur.numel() > 1:
        n = cur.numel()
        nout = triton.cdiv(n, block)
        nxt = torch.empty(nout, dtype=cur.dtype, device=cur.device)
        _reduce_partial[(nout,)](cur, nxt, n, BLOCK=block)
        cur = nxt
    return cur


def kernel(dvol, rho, vxc, exc):
    rho_c = rho.contiguous()
    vxc_c = vxc if vxc.is_contiguous() else vxc.contiguous()
    if rho_c.dtype != vxc_c.dtype:
        rho_c = rho_c.to(vxc_c.dtype)
    n = rho_c.numel()
    dtype = vxc_c.dtype
    consts = torch.tensor(_CONSTS, dtype=dtype, device=rho_c.device)
    partial = torch.zeros(triton.cdiv(n, _MIN_BLOCK), dtype=dtype, device=rho_c.device)

    grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)
    _map[grid](rho_c, vxc_c, partial, consts, n, _tl_float(dtype), enable_fp_fusion=False)
    if vxc_c.data_ptr() != vxc.data_ptr():
        vxc.copy_(vxc_c)

    total = _tree_sum(partial)
    dvol_t = torch.tensor(float(dvol), dtype=dtype, device=rho_c.device)
    exc.copy_((total * dvol_t).to(dtype=exc.dtype))
