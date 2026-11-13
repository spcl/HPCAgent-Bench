# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Triton vexx_k (QE exact exchange, exx_bp::vexx_bp_k): the per-grid-point work in Triton, FFTs and the small
projector contractions in torch.

A plain (unoptimized) port with the structure and loop granularity of ``vexx_k_numpy.py``: for each band ii, one
partner band jbnd at a time (one FFT pair per partner, as the reference), each stage below its own kernel or call.
Batching the partners, and fusing the per-element stages, are left for the optimization stage. The kernels and the
torch helpers take a leading partner axis of length J; here it is always 1:
  pair density   rhoc[j] = sum_ip conj(phi[j, ip]) * psi_ii[ip] / omega       (Triton)
  augmentation   addusxx_r / addusxx_g on rhoc                                  (torch, batched over j)
  FFT            rhocg = fftn(rhoc)                                             (torch.fft = cuFFT)
  Coulomb        vc[j] = facb * rhocg[j] * occ[j] / nqs                         (Triton)
  newdxx_g / _r / paw_newdxx into deexx[ii]                                     (torch, batched over j)
  FFT            vcr = ifftn(vc)
  accumulate     result[ii, ip] += vcr * phi[jbnd, ip]                           (Triton)
The setup scatter (psi to the FFT grid) and the final gather back to the G-sphere are Triton kernels too. Band
bookkeeping (ibands, egrp_pairs, ranges) is small integer data read once on the host.

The ``coulomb_fac_q`` / ``qgm_q`` / ``sf_q`` keyword inputs are supported; the Coulomb-truncation and
gamma-extrapolation keywords (none of which the manifest passes) raise.
"""

import numpy as np
import torch
import triton
import triton.language as tl

_E2 = 2.0
_FPI = 4.0 * np.pi
_PI = np.pi
BLOCK = 256


@triton.jit
def scatter_kernel(psi, grid, nlg, valid, n, ld, off, nrxxs, BLOCK: tl.constexpr):
    """grid[b, nlg[i]] = psi[b, off + i] for i < n, for the bands b with valid[b] (the rest stay zero)."""
    band = tl.program_id(0).to(tl.int64)
    i = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = (i < n) & (tl.load(valid + band) != 0)
    cell = tl.load(nlg + i, mask=mask, other=0)
    src = psi + 2 * (band * ld + off + i)
    dst = grid + 2 * (band * nrxxs + cell)
    tl.store(dst, tl.load(src, mask=mask), mask=mask)
    tl.store(dst + 1, tl.load(src + 1, mask=mask), mask=mask)


@triton.jit
def pair_density_kernel(phi, psic, rhoc, scal, nrxxs, NPOL: tl.constexpr, BLOCK: tl.constexpr):
    """rhoc[j, r] = (sum_ip conj(phi[j, ip, r]) * psic[ip, r]) * scal[0] (scal = 1 / omega)."""
    j = tl.program_id(0).to(tl.int64)
    r = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = r < nrxxs
    acc_re = tl.zeros((BLOCK,), dtype=phi.dtype.element_ty)
    acc_im = tl.zeros((BLOCK,), dtype=phi.dtype.element_ty)
    for ip in tl.static_range(NPOL):
        pp = phi + 2 * ((j * NPOL + ip) * nrxxs + r)
        qp = psic + 2 * (ip * nrxxs + r)
        a_re = tl.load(pp, mask=mask)
        a_im = tl.load(pp + 1, mask=mask)
        b_re = tl.load(qp, mask=mask)
        b_im = tl.load(qp + 1, mask=mask)
        # conj(a) * b
        acc_re += a_re * b_re + a_im * b_im
        acc_im += a_re * b_im - a_im * b_re
    s = tl.load(scal)
    dst = rhoc + 2 * (j * nrxxs + r)
    tl.store(dst, acc_re * s, mask=mask)
    tl.store(dst + 1, acc_im * s, mask=mask)


@triton.jit
def coulomb_kernel(rhocg, facb, occ, nrxxs, BLOCK: tl.constexpr):
    """rhocg[j, r] = facb[r] * rhocg[j, r] * occ[j], in place (occ = x_occupation / nqs)."""
    j = tl.program_id(0).to(tl.int64)
    r = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = r < nrxxs
    f = tl.load(facb + r, mask=mask)
    o = tl.load(occ + j)
    p = rhocg + 2 * (j * nrxxs + r)
    tl.store(p, f * tl.load(p, mask=mask) * o, mask=mask)
    tl.store(p + 1, f * tl.load(p + 1, mask=mask) * o, mask=mask)


@triton.jit
def accumulate_kernel(vcr, phi, result, J, nrxxs, NPOL: tl.constexpr, BLOCK: tl.constexpr):
    """result[ip, r] += vcr[j, r] * phi[j, ip, r], summed over j = 0..J-1 in order."""
    ip = tl.program_id(0).to(tl.int64)
    r = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = r < nrxxs
    dst = result + 2 * (ip * nrxxs + r)
    acc_re = tl.load(dst, mask=mask)
    acc_im = tl.load(dst + 1, mask=mask)
    for j in range(J):
        vp = vcr + 2 * (j * nrxxs + r)
        pp = phi + 2 * ((j * NPOL + ip) * nrxxs + r)
        v_re = tl.load(vp, mask=mask)
        v_im = tl.load(vp + 1, mask=mask)
        p_re = tl.load(pp, mask=mask)
        p_im = tl.load(pp + 1, mask=mask)
        acc_re += v_re * p_re - v_im * p_im
        acc_im += v_re * p_im + v_im * p_re
    tl.store(dst, acc_re, mask=mask)
    tl.store(dst + 1, acc_im, mask=mask)


@triton.jit
def gather_kernel(rg, big, nlg, alpha, row, n, nrxxs, ld, BLOCK: tl.constexpr):
    """big[row, ip * n + i] -= alpha[0] * rg[ip, nlg[i]] for i < n (one program row per polarization ip)."""
    ip = tl.program_id(0).to(tl.int64)
    i = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = i < n
    cell = tl.load(nlg + i, mask=mask, other=0)
    a = tl.load(alpha)
    src = rg + 2 * (ip * nrxxs + cell)
    dst = big + 2 * (row * ld + ip * n + i)
    tl.store(dst, tl.load(dst, mask=mask) - a * tl.load(src, mask=mask), mask=mask)
    tl.store(dst + 1, tl.load(dst + 1, mask=mask) - a * tl.load(src + 1, mask=mask), mask=mask)


def _g2_convolution(g, xk, xkq, ngm, tpiba2, exxdiv, eps_qdiv, gau_scrlen, erf_scrlen, erfc_scrlen, yukawa):
    """exx_base::g2_convolution, the branches the manifest reaches (no Coulomb truncation, no gamma extrapolation)."""
    q = xk[None, :] - xkq[None, :] + g[:ngm, :]
    qq = torch.sum(q * q, dim=1) * tpiba2
    gf = torch.ones(ngm, dtype=g.dtype, device=g.device)
    nonsing = qq > eps_qdiv
    qqn = torch.where(nonsing, qq, torch.ones_like(qq))
    if gau_scrlen > 0.0:
        return _E2 * (_PI / gau_scrlen) ** 1.5 * torch.exp(-qq / 4.0 / gau_scrlen) * gf
    if erfc_scrlen > 0.0:
        fac = _E2 * _FPI / qqn * (1.0 - torch.exp(-qqn / 4.0 / (erfc_scrlen * erfc_scrlen))) * gf
    elif erf_scrlen > 0.0:
        fac = _E2 * _FPI / qqn * torch.exp(-qqn / 4.0 / (erf_scrlen * erf_scrlen)) * gf
    else:
        fac = _E2 * _FPI / (qqn + yukawa) * gf
    fac = torch.where(nonsing, fac, torch.full_like(fac, -exxdiv))
    if yukawa > 0.0:
        fac = torch.where(nonsing, fac, fac + _E2 * _FPI / (qq + yukawa))
    if erfc_scrlen > 0.0:
        fac = torch.where(nonsing, fac, fac + _E2 * _PI / (erfc_scrlen * erfc_scrlen))
    return fac


class _Augmentation:
    """The ultrasoft / PAW helpers of the reference, batched over the J partner bands (becphi is (J, nkb))."""

    def __init__(self, ijtoh, ofsbeta, nh, tabxx_box, tabxx_qr, ke, cdt):
        dev = ofsbeta.device
        self.ikb = ofsbeta.to(torch.int64)[:, None] + torch.arange(nh, device=dev)[None, :]  # (nat, nh)
        self.ikb_flat = self.ikb.reshape(-1)
        self.ijtoh = ijtoh.to(torch.int64)
        self.box = tabxx_box.to(torch.int64)
        self.qr = tabxx_qr[:, self.ijtoh, :].to(cdt)  # (nat, nh, nh, maxbox)
        self.ke = ke.to(cdt)

    def addusxx_g(self, rhocg, nl0, qgm, becphi, becpsi, eigqts, sfac):
        q_pairs = qgm[self.ijtoh, :]  # (nh, nh, ngm)
        inner1 = torch.einsum("ijg,aj->aig", q_pairs, becpsi[self.ikb])
        inner2 = torch.einsum("aig,Jai->Jag", inner1, becphi[:, self.ikb].conj())
        sf = sfac * eigqts[:, None]
        rhocg[:, nl0] += torch.sum(sf[None] * inner2, dim=1)

    def newdxx_g(self, vc, nl0, qgm, becphi, deexx_row, eigqts, sfac, omega):
        sf = sfac * eigqts[:, None]
        aux2 = vc[:, nl0].conj()[:, None, :] * sf[None]  # (J, nat, ngm)
        q_pairs = qgm[self.ijtoh, :].conj()
        t = torch.einsum("Jag,ijg->Jaij", aux2.conj(), q_pairs)
        delta = omega * torch.einsum("Jaj,Jaij->ai", becphi[:, self.ikb], t)
        deexx_row.index_add_(0, self.ikb_flat, delta.reshape(-1))

    def addusxx_r(self, rhoc, becphi, becpsi):
        weight = becphi[:, self.ikb].conj()[:, :, :, None] * becpsi[self.ikb][None, :, None, :]  # (J, nat, nh, nh)
        contrib = torch.einsum("aijb,Jaij->Jab", self.qr, weight)
        rhoc.index_add_(1, self.box.reshape(-1), contrib.reshape(contrib.shape[0], -1))

    def newdxx_r(self, vcr, becphi, deexx_row, omega, nnr):
        vc_box = vcr[:, self.box]  # (J, nat, maxbox)
        aux = torch.einsum("aijb,Jab->Jaij", self.qr, vc_box)
        delta = (omega / nnr) * torch.einsum("Jaj,Jaij->ai", becphi[:, self.ikb], aux)
        deexx_row.index_add_(0, self.ikb_flat, delta.reshape(-1))

    def paw_newdxx(self, weight, becphi, becpsi, deexx_row):
        b = becphi[:, self.ikb]
        t = torch.einsum("uoji,Jaj,Jau,ao->Jai", self.ke, b, b.conj(), becpsi[self.ikb])
        delta = torch.sum(weight[:, None, None] * 0.5 * t, dim=0)
        deexx_row.index_add_(0, self.ikb_flat, delta.reshape(-1))


def vexx_k(
    psi,
    hpsi,
    exxbuff,
    x_occupation,
    g,
    nl,
    nlm,
    igk_exx,
    index_xk,
    index_xkq,
    xk,
    xkq_collect,
    ibands,
    nibands,
    egrp_pairs,
    all_start,
    all_end,
    iexx_istart,
    iexx_iend,
    becpsi,
    becxx,
    qgm,
    ijtoh,
    ofsbeta,
    eigqts,
    sfac,
    vkb,
    tabxx_box,
    tabxx_qr,
    ke,
    exxalfa,
    omega,
    tpiba2,
    exxdiv,
    eps_qdiv,
    gau_scrlen,
    erf_scrlen,
    erfc_scrlen,
    yukawa,
    eps_occ,
    nqs,
    n,
    m,
    npwx,
    npol,
    nrxxs,
    ngm,
    n1,
    n2,
    n3,
    nbnd,
    nat,
    nh,
    nkb,
    max_pairs,
    jblock,
    negrp,
    iexx_start,
    my_egrp_id,
    current_k,
    current_ik,
    okvan,
    okpaw,
    noncolin,
    tqr,
    gamma_only,
    coulomb_fac_q=None,
    qgm_q=None,
    sf_q=None,
    x_gamma_extrapolation=False,
    grid_factor=1.0,
    at=None,
    nq1=1,
    nq2=1,
    nq3=1,
    eps_gcv=1e-6,
    use_coulomb_vcut_ws=False,
    use_coulomb_vcut_spheric=False,
    vcut_a=None,
    vcut_cutoff=0.0,
    vcut_corrected=None,
):
    if x_gamma_extrapolation or use_coulomb_vcut_ws or use_coulomb_vcut_spheric:
        raise NotImplementedError("vexx_k_triton: gamma extrapolation / Coulomb truncation are not ported")
    nqs, n, m, npwx, npol = int(nqs), int(n), int(m), int(npwx), int(npol)
    nrxxs, ngm, n1, n2, n3 = int(nrxxs), int(ngm), int(n1), int(n2), int(n3)
    nat, nh, nkb, max_pairs = int(nat), int(nh), int(nkb), int(max_pairs)
    negrp, iexx_start, eg = int(negrp), int(iexx_start), int(my_egrp_id)
    current_k, current_ik = int(current_k), int(current_ik)
    okvan, okpaw, tqr, gamma_only = bool(okvan), bool(okpaw), bool(tqr), bool(gamma_only)
    omega, exxalfa, eps_occ = float(omega), float(exxalfa), float(eps_occ)
    dev = psi.device
    cdt = psi.dtype
    rdt = g.dtype
    grid = (n3, n2, n1)  # QE's column-major (n1, n2, n3) grid, as C-order axes
    nqs_inv = 1.0 / nqs

    # Band bookkeeping: small integer tables, read once on the host.
    ibands_h = ibands.cpu().numpy().astype(np.int64)
    nibands_h = nibands.cpu().numpy()
    pairs_h = egrp_pairs.cpu().numpy()
    all_start_h, all_end_h = all_start.cpu().numpy(), all_end.cpu().numpy()
    index_xk_h, index_xkq_h = index_xk.cpu().numpy(), index_xkq.cpu().numpy()

    nl64 = nl.to(torch.int64)
    nl0 = nl64[:ngm]
    gki = igk_exx[current_k - 1, :n].to(torch.int64)
    nlg = nl64[gki].contiguous()
    aug = _Augmentation(ijtoh, ofsbeta, nh, tabxx_box, tabxx_qr, ke, cdt) if okvan else None
    omega_inv = torch.tensor([1.0 / omega], dtype=rdt, device=dev)
    alpha_t = torch.tensor([exxalfa], dtype=rdt, device=dev)

    my_n = int(nibands_h[eg])
    exxbuff_w = exxbuff.clone() if negrp > 1 else exxbuff  # only the band-group shift mutates it

    # setup: my bands scattered to the grid (invalid ones left zero), to real space in one batched IFFT per polarization
    ibnd_all = ibands_h[eg, :my_n]
    valid = torch.from_numpy(((ibnd_all != 0) & (ibnd_all <= m)).astype(np.int32)).to(dev)
    temppsic = torch.zeros((my_n, npol, nrxxs), dtype=cdt, device=dev)
    for ip in range(npol):
        tg = torch.zeros((my_n, nrxxs), dtype=cdt, device=dev)
        if my_n > 0 and n > 0:
            scatter_kernel[(my_n, triton.cdiv(n, BLOCK))](
                torch.view_as_real(psi),
                torch.view_as_real(tg),
                nlg,
                valid,
                n,
                psi.shape[1],
                ip * npwx,
                nrxxs,
                BLOCK=BLOCK,
            )
        temppsic[:, ip, :] = torch.fft.ifftn(tg.view((my_n,) + grid), dim=(-3, -2, -1)).reshape(my_n, nrxxs)

    deexx = torch.zeros((my_n, nkb), dtype=cdt, device=dev)
    result = torch.zeros((my_n, npol, nrxxs), dtype=cdt, device=dev)
    big_result = torch.zeros((m, n * npol), dtype=cdt, device=dev)
    g_r = triton.cdiv(nrxxs, BLOCK)

    def _pair(jbnd, ii, ikq, ik, all_start_tmp, becpsi_i, qgm_use, sfac_use, eigqts_use, facb):
        """The reference's per-partner body: band ii against partner band jbnd (J = 1 below)."""
        J = 1
        buf0 = jbnd - all_start_tmp + iexx_start - 1
        phi = exxbuff_w[ikq, buf0 : buf0 + 1, :].contiguous()  # (1, npol * nrxxs)
        phi_ri = torch.view_as_real(phi)
        becphi = becxx[ikq, jbnd - 1 : jbnd, :]  # (1, nkb)
        occ = x_occupation[ik, jbnd - 1 : jbnd] * nqs_inv  # (1,)
        rhoc = torch.empty((J, nrxxs), dtype=cdt, device=dev)
        pair_density_kernel[(J, g_r)](
            phi_ri,
            torch.view_as_real(temppsic[ii]),
            torch.view_as_real(rhoc),
            omega_inv,
            nrxxs,
            NPOL=npol,
            BLOCK=BLOCK,
        )
        if okvan and tqr:
            aug.addusxx_r(rhoc, becphi, becpsi_i)
        rhocg = torch.fft.fftn(rhoc.view((J,) + grid), dim=(-3, -2, -1)).reshape(J, nrxxs)
        if okvan and not tqr:
            aug.addusxx_g(rhocg, nl0, qgm_use, becphi, becpsi_i, eigqts_use, sfac_use)
        coulomb_kernel[(J, g_r)](torch.view_as_real(rhocg), facb, occ.contiguous(), nrxxs, BLOCK=BLOCK)
        vc = rhocg
        if okvan and not tqr:
            aug.newdxx_g(vc, nl0, qgm_use, becphi, deexx[ii], eigqts_use, sfac_use, omega)
        vcr = torch.fft.ifftn(vc.view((J,) + grid), dim=(-3, -2, -1)).reshape(J, nrxxs)
        if okvan and tqr:
            aug.newdxx_r(vcr, becphi, deexx[ii], omega, nrxxs)
        if okpaw:
            aug.paw_newdxx(occ.to(cdt), becphi, becpsi_i, deexx[ii])
        accumulate_kernel[(npol, g_r)](
            torch.view_as_real(vcr), phi_ri, torch.view_as_real(result[ii]), J, nrxxs, NPOL=npol, BLOCK=BLOCK
        )

    coulomb_cache = {}
    for iq in range(1, nqs + 1):
        ikq = int(index_xkq_h[iq - 1, current_ik - 1])
        ik = int(index_xk_h[ikq])
        if coulomb_fac_q is not None:
            fac = coulomb_fac_q[iq - 1, :ngm].to(rdt)
        else:
            if iq not in coulomb_cache:
                coulomb_cache[iq] = _g2_convolution(
                    g,
                    xk[current_k - 1, :],
                    xkq_collect[ikq, :],
                    ngm,
                    float(tpiba2),
                    float(exxdiv),
                    float(eps_qdiv),
                    float(gau_scrlen),
                    float(erf_scrlen),
                    float(erfc_scrlen),
                    float(yukawa),
                )
            fac = coulomb_cache[iq]
        facb = torch.zeros(nrxxs, dtype=fac.dtype, device=dev)
        facb[nl0] = fac

        if qgm_q is not None:
            qgm_use, sfac_use = qgm_q[iq - 1], sf_q[iq - 1]
            eigqts_use = torch.ones(nat, dtype=cdt, device=dev)
        else:
            qgm_use, sfac_use, eigqts_use = qgm, sfac, eigqts

        for iegrp in range(1, negrp + 1):
            wegrp = (iegrp + eg - 1) % negrp + 1
            all_start_tmp = int(all_start_h[wegrp - 1])
            all_end_tmp = int(all_end_h[wegrp - 1])
            for ii in range(my_n):
                ibnd = int(ibands_h[eg, ii])
                if ibnd == 0 or ibnd > m:
                    continue
                match = pairs_h[eg, :max_pairs, 0] == ibnd
                if not np.any(match):
                    continue
                jv = pairs_h[eg, :max_pairs, 1]
                jmin = int(np.min(np.where(match, jv, np.max(jv))))
                jmax = int(np.max(np.where(match, jv, np.min(jv))))
                jstart = max(jmin, all_start_tmp)
                jend = min(jmax, all_end_tmp)
                if jend < jstart:
                    continue
                becpsi_i = becpsi[ibnd - 1, :]
                for jbnd in range(jstart, jend + 1):
                    _pair(jbnd, ii, ikq, ik, all_start_tmp, becpsi_i, qgm_use, sfac_use, eigqts_use, facb)

            if negrp > 1:
                exxbuff_w[ikq] = torch.roll(exxbuff_w[ikq], -1, dims=0)

    # finalize: result(r) -> G-sphere, onto big_result; one FFT per band (over its polarizations), as the reference
    if okvan:
        d_all = deexx[:, aug.ikb_flat]
        keep = d_all.abs() >= eps_occ
        dv_all = torch.where(keep, d_all.real.to(cdt) if gamma_only else d_all, torch.zeros_like(d_all))
        vkb_sel = vkb[aug.ikb_flat, :n]
    for ii in range(my_n):
        ibnd = int(ibands_h[eg, ii])
        if ibnd == 0 or ibnd > m:
            continue
        rg = torch.fft.fftn(result[ii].view((npol,) + grid), dim=(-3, -2, -1)).reshape(npol, nrxxs)
        if n > 0:
            gather_kernel[(npol, triton.cdiv(n, BLOCK))](
                torch.view_as_real(rg),
                torch.view_as_real(big_result),
                nlg,
                alpha_t,
                ibnd - 1,
                n,
                nrxxs,
                n * npol,
                BLOCK=BLOCK,
            )
        if okvan:
            big_result[ibnd - 1, :n] -= exxalfa * (dv_all[ii] @ vkb_sel)

    istart = int(iexx_istart[eg])
    if istart > 0:
        ending = m if negrp == 1 else (int(iexx_iend[eg]) - istart + 1)
        for ip in range(npol):
            hpsi[:ending, ip * npwx : ip * npwx + n] += big_result[
                istart - 1 : istart - 1 + ending, ip * n : ip * n + n
            ]
