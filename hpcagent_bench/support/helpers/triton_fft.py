# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Mixed-radix Stockham FFT in Triton, for any length and any axis of a contiguous array.

Used by the scientific Triton baselines that used to call ``torch.fft`` (cuFFT).

Algorithm: N = p_1 * p_2 * ... (radix 4 first, then 2, 3, 5, 7, then any remaining prime).
Stage s with radix P and Ns = product of the radices before it maps group j in [0, N/P) as

    y[(j // Ns) * Ns * P + (j % Ns) + u * Ns] = sum_t x[j + t * N/P] * w^(t * ((j % Ns) + u * Ns)),
    w = exp(-+2 pi i / (Ns * P)),

which folds the inter-stage twiddle and the P-point DFT into one exponent and leaves the output
in natural order (Stockham autosort). Every stage is out of place (ping-pong buffers). Twiddles
and accumulation are float64 whatever the data dtype. A large prime factor P costs O(N * P),
like a naive DFT of that factor.

``div`` divides the result by an integer: 1 for an unnormalised transform, N for NumPy's
``ifft``, and so on. It is an integer so the scale stays exact in float64.

NOTE: written without a GPU. To be tested.
"""

import math

import torch
import triton
import triton.language as tl

__all__ = ["fft", "fftn", "ifft", "ifftn"]

_BLOCK = 256


def _factor(n):
    out = []
    while n % 4 == 0:
        out.append(4)
        n //= 4
    for p in (2, 3, 5, 7):
        while n % p == 0:
            out.append(p)
            n //= p
    p = 11
    while n > 1 and p * p <= n:
        while n % p == 0:
            out.append(p)
            n //= p
        p += 2
    if n > 1:
        out.append(n)
    return out


@triton.jit
def _stage_kernel(src_ptr, dst_ptr, total, N, INNER, Ns, NP, div,
                  P: tl.constexpr, INV: tl.constexpr, APPLY_DIV: tl.constexpr, BLOCK: tl.constexpr):
    # src/dst are real views of complex arrays: element e has its real part at 2e, imag at 2e+1.
    g = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    m = g < total
    b = g // NP
    j = g % NP
    base = (b // INNER) * N * INNER + (b % INNER)
    k = j % Ns
    out0 = (j // Ns) * Ns * P + k
    NsP = Ns * P
    NsPf = (k * 0 + NsP).to(tl.float64)  # Ns == 1 arrives as a constexpr; go through a tensor
    two_pi = 6.283185307179586
    if INV:
        sgn = 1.0
    else:
        sgn = -1.0
    for u in range(P):
        acc_r = tl.zeros((BLOCK,), dtype=tl.float64)
        acc_i = tl.zeros((BLOCK,), dtype=tl.float64)
        e = k + u * Ns
        for t in range(P):
            src = base + (j + t * NP) * INNER
            xr = tl.load(src_ptr + 2 * src, mask=m, other=0.0).to(tl.float64)
            xi = tl.load(src_ptr + 2 * src + 1, mask=m, other=0.0).to(tl.float64)
            num = (t * e) % NsP
            theta = sgn * two_pi * (num.to(tl.float64) / NsPf)
            c = tl.cos(theta)
            s = tl.sin(theta)
            acc_r += xr * c - xi * s
            acc_i += xr * s + xi * c
        if APPLY_DIV:
            dv = (k * 0 + div).to(tl.float64)
            acc_r = acc_r / dv
            acc_i = acc_i / dv
        dst = base + (out0 + u * Ns) * INNER
        tl.store(dst_ptr + 2 * dst, acc_r.to(dst_ptr.dtype.element_ty), mask=m)
        tl.store(dst_ptr + 2 * dst + 1, acc_i.to(dst_ptr.dtype.element_ty), mask=m)


@triton.jit
def _scale_copy_kernel(src_ptr, dst_ptr, n2, div, APPLY_DIV: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n2
    v = tl.load(src_ptr + offs, mask=m, other=0.0).to(tl.float64)
    if APPLY_DIV:
        v = v / (offs * 0 + div).to(tl.float64)
    tl.store(dst_ptr + offs, v.to(dst_ptr.dtype.element_ty), mask=m)


def _as_complex(x):
    if x.is_complex():
        return x if x.is_contiguous() else x.contiguous()
    ctype = torch.complex128 if x.dtype == torch.float64 else torch.complex64
    return x.to(ctype).contiguous()


def fft(x, axis=-1, inverse=False, div=1, out=None):
    """1-D DFT of complex ``x`` along ``axis`` (sign -1 forward, +1 inverse), divided by ``div``.

    Returns a new tensor unless ``out`` (same shape/dtype, contiguous, not ``x``) is given.
    """
    x = _as_complex(x)
    shape = x.shape
    axis = axis % x.dim()
    N = int(shape[axis])
    INNER = int(math.prod(shape[axis + 1:]))
    batch = x.numel() // max(N, 1)
    if out is None:
        out = torch.empty_like(x)
    xr = torch.view_as_real(x)
    radices = _factor(N) if N > 1 else []
    if not radices:
        n2 = 2 * x.numel()
        _scale_copy_kernel[(triton.cdiv(n2, _BLOCK),)](
            xr, torch.view_as_real(out), n2, div, APPLY_DIV=div != 1, BLOCK=_BLOCK
        )
        return out
    # Ping-pong: the last stage writes ``out``; the stage before it a scratch buffer.
    scratch = torch.empty_like(x) if len(radices) > 1 else None
    targets = []
    for s in range(len(radices)):
        remaining = len(radices) - 1 - s
        targets.append(out if remaining % 2 == 0 else scratch)
    src = xr
    Ns = 1
    for s, P in enumerate(radices):
        dst = torch.view_as_real(targets[s])
        NP = N // P
        total = batch * NP
        last = s == len(radices) - 1
        _stage_kernel[(triton.cdiv(total, _BLOCK),)](
            src, dst, total, N, INNER, Ns, NP, div,
            P=P, INV=inverse, APPLY_DIV=last and div != 1, BLOCK=_BLOCK,
        )
        src = dst
        Ns *= P
    return out


def ifft(x, axis=-1, div=None, out=None):
    """NumPy-style inverse (divides by N unless ``div`` is given)."""
    x = _as_complex(x)
    n = int(x.shape[axis % x.dim()])
    return fft(x, axis=axis, inverse=True, div=n if div is None else div, out=out)


def fftn(x, axes=None, inverse=False, div=1, out=None):
    """n-D DFT over ``axes`` (default: all), divided by ``div`` once at the end."""
    x = _as_complex(x)
    axes = list(range(x.dim())) if axes is None else [a % x.dim() for a in axes]
    cur = x
    for i, a in enumerate(axes):
        last = i == len(axes) - 1
        cur = fft(cur, axis=a, inverse=inverse, div=div if last else 1, out=out if last else None)
    return cur


def ifftn(x, axes=None, div=None, out=None):
    """NumPy-style n-D inverse (divides by the product of the transformed lengths)."""
    x = _as_complex(x)
    axes_l = list(range(x.dim())) if axes is None else [a % x.dim() for a in axes]
    n = int(math.prod(int(x.shape[a]) for a in axes_l))
    return fftn(x, axes=axes_l, inverse=True, div=n if div is None else div, out=out)
