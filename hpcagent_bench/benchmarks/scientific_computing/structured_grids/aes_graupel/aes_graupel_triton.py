# Hand-written Triton baseline for ICON AES graupel (port of aes_graupel_numba / aes_graupel_numpy).
import triton
import triton.language as tl
from triton.language.extra import libdevice

LQR = tl.constexpr(0)
LQI = tl.constexpr(1)
LQS = tl.constexpr(2)
LQG = tl.constexpr(3)
LQC = tl.constexpr(4)
LQV = tl.constexpr(5)
NX = tl.constexpr(6)

_TM = tl.constexpr(273.15)
_QMIN = tl.constexpr(1.0e-15)
_M0 = tl.constexpr(1.0e-12)
_RD = tl.constexpr(287.04)
_RV = tl.constexpr(461.51)
_CPD = tl.constexpr(1004.64)
_CVD = tl.constexpr(_CPD.value - _RD.value)
_CPV = tl.constexpr(1869.46)
_CVV = tl.constexpr(_CPV.value - _RV.value)
_CLW = tl.constexpr((3.1733 + 1.0) * _CPD.value)
_CI = tl.constexpr(2108.0)
_LVC = tl.constexpr(2.5008e6 - (_CPV.value - _CLW.value) * _TM.value)
_LSC = tl.constexpr(2.8345e6 - (_CPV.value - _CI.value) * _TM.value)
_TX = tl.constexpr(3339.5)
_R00 = tl.constexpr(1.225)
_V0S = tl.constexpr(25.0)
_V1S = tl.constexpr(0.5)
_AMS = tl.constexpr(0.069)
_BMS = tl.constexpr(2.0)
_TH1 = tl.constexpr(_TM.value - 6.0)
_TH2 = tl.constexpr(_TM.value - 25.0)
_THOM = tl.constexpr(_TM.value - 37.0)


def _configs():
    return [triton.Config({}, num_warps=n) for n in (1, 2, 4)]


@triton.jit
def _qsat_rho(t, rho):
    return 610.78 * libdevice.exp(17.269 * (t - _TM) / (t - 35.86)) / (rho * _RV * t)


@triton.jit
def _qsat_ice(t, rho):
    return 610.78 * libdevice.exp(21.875 * (t - _TM) / (t - 7.66)) / (rho * _RV * t)


@triton.jit
def _ie(t, qv, qliq, qice, rho, dz):
    qtot = qliq + qice + qv
    cv = _CVD * (1.0 - qtot) + _CVV * qv + _CLW * qliq + _CI * qice
    return rho * dz * (cv * t - qliq * _LVC - qice * _LSC)


@triton.jit
def _t_ie(en, qv, qliq, qice, rho, dz):
    qtot = qliq + qice + qv
    cv = (_CVD * (1.0 - qtot) + _CVV * qv + _CLW * qliq + _CI * qice) * rho * dz
    return (en + rho * dz * (qliq * _LVC + qice * _LSC)) / cv


@triton.jit
def _snow_n(t, rho, qs):
    n0 = tl.full((), 8.0e5, t.dtype)
    has = qs > _QMIN
    tc = tl.maximum(tl.minimum(t, _TM), _TM - 40.0) - _TM
    alf = libdevice.pow(10.0, -1.65 + tc * (5.45e-2 + tc * 3.27e-4))
    bet = 1.42 + tc * (1.19e-2 + tc * 9.6e-5)
    qu = tl.where(has, qs, 1.0)
    n0s = 13.5 * libdevice.pow((qu + 2.0e-6) * rho / _AMS, 4.0 - 3.0 * bet) / (alf * alf * alf)
    y = libdevice.exp(-0.107 * tc)
    n0s1 = 13.5 * 5.65e5
    lo = tl.maximum(0.5 * n0s1 * y, 1.0e6)
    hi = tl.minimum(1.0e2 * n0s1 * y, 1.0e9)
    return tl.where(has, tl.minimum(hi, tl.maximum(lo, n0s)), n0)


@triton.jit
def _snow_l(rho, qs, ns):
    has = qs > _QMIN
    qu = tl.where(has, qs, 1.0)
    lam = libdevice.pow(_AMS * 2.0 * ns / (qu * rho), 1.0 / (_BMS + 1.0))
    return tl.where(has, lam, tl.full((), 1.0e10, rho.dtype))


@triton.jit
def _fall(density, factor, exponent, offset):
    dt = density.dtype
    base = density + tl.full((), offset, dt)
    return tl.full((), factor, dt) * libdevice.pow(base, tl.full((), exponent, dt))


@triton.jit
def _process_rates(t, p, rho, qv, qc, qi, qr, qs, qg, nc, dt, sig, sx):
    z = tl.full((), 0.0, t.dtype)
    for i in tl.static_range(NX):
        for j in tl.static_range(NX):
            tl.store(sx + i * NX + j, z)
    dvsw = qv - _qsat_rho(t, rho)
    qvsi = _qsat_ice(t, rho)
    dvsi = qv - qvsi
    ns = _snow_n(t, rho, qs)
    ls = _snow_l(rho, qs, ns)
    acts = (qc > 1.0e-6) & (t > _THOM)
    tot = tl.where(acts, qc + qr, 1.0)
    tau = tl.maximum(1.0e-30, tl.minimum(1.0 - qc / tot, 0.90))
    phi = libdevice.pow(tau, 0.68)
    phi = 6.0e2 * phi * libdevice.pow(1.0 - phi, 3.0)
    auk = 9.44e09 / (20.0 * 2.60e-10) * 4.0 * 6.0 / 9.0
    xau = tl.where(acts, auk * libdevice.pow(qc * qc / nc, 2.0) * (1.0 + phi / libdevice.pow(1.0 - tau, 2.0)), 0.0)
    xac = tl.where(acts, 5.25 * qc * qr * libdevice.pow(tau / (tau + 5.0e-05), 4.0), 0.0)
    tl.store(sx + LQC * NX + LQR, xau + xac)
    acts_rv = (qr > _QMIN) & (dvsw + qc <= 0.0)
    tc = t - _TM
    evm = (0.61 + tc * (-0.0163 + 1.111e-4 * tc)) * (-dvsw) / dt
    rv = tl.minimum(
        1.536e-3 * (1.0 + 19.0621 * libdevice.pow(qr * rho, 0.16667)) * (-dvsw) * libdevice.pow(qr * rho, 0.55555),
        evm,
    )
    tl.store(sx + LQR * NX + LQV, tl.where(acts_rv, rv, 0.0))
    cxi = tl.where((qc > _QMIN) & (t < _THOM), qc / dt, 0.0)
    cxi = tl.where((qi > _QMIN) & (t > _TM), -qi / dt, cxi)
    tl.store(sx + LQC * NX + LQI, cxi)
    v = tl.load(sx + LQC * NX + LQI)
    tl.store(sx + LQI * NX + LQC, -tl.minimum(v, 0.0))
    tl.store(sx + LQC * NX + LQI, tl.maximum(v, 0.0))
    acts = (tl.minimum(qc, qs) > _QMIN) & (t > _THOM)
    tl.store(sx + LQC * NX + LQS, tl.where(acts, (2.61 * 0.9 * _V0S * ns) * qc * libdevice.pow(ls, -(_V1S + 3.0)), 0.0))
    acts = (tl.minimum(qc, qg) > _QMIN) & (t > _THOM)
    tl.store(sx + LQC * NX + LQG, tl.where(acts, 4.43 * qc * libdevice.pow(qg * rho, 0.94878), 0.0))
    cold = t < _TM
    ni = tl.minimum(250.0e3, 5.0 * libdevice.exp(0.304 * (_TM - t))) / rho
    mi = tl.maximum(_M0, tl.minimum(qi / ni, 1.0e-9))
    xs = tl.maximum(
        tl.maximum(tl.minimum(libdevice.exp(0.09 * (t - _TM)), 1.0), 0.075),
        3.5e-3 * (t - (_TM - 85.0)),
    )
    eta = z
    idep = z
    b = 1.94
    a = 2.8345e6 * 2.8345e6 / (2.40e-2 * _RV)
    cx = 2.22e-5 * libdevice.pow(_TM, -b) * 101325.0
    xdep = cx / _RD * libdevice.pow(t, b - 1.0)
    eta_c = xdep / (1.0 + a * xdep * qvsi / (t * t))
    a0 = 4.0 * libdevice.pow(130.0, -1.0 / 3.0)
    vi = (a0 * eta_c) * rho * qi * libdevice.pow(mi, -0.67) * dvsi
    vi = tl.where(vi > 0.0, tl.minimum(vi, dvsi / dt), tl.maximum(tl.maximum(vi, dvsi / dt), -qi / dt))
    vi = tl.where(qi > _QMIN, vi, 0.0)
    eta = tl.where(cold & sig, eta_c, 0.0)
    vi_pos = tl.where(cold & sig, tl.maximum(vi, 0.0), 0.0)
    tl.store(sx + LQV * NX + LQI, vi_pos)
    tl.store(sx + LQI * NX + LQV, tl.where(cold & sig, -tl.minimum(vi, 0.0), z))
    idep = tl.where(cold & sig, tl.minimum(vi_pos, dvsi / dt), 0.0)
    b2 = 2.0 / 3.0
    dep = tl.where((cold & sig) & (qi > _QMIN), tl.maximum(0.0, idep) * (b2 / (libdevice.pow(3.0e-9 / mi, b2) - 1.0)), 0.0)
    tl.store(sx + LQI * NX + LQS, dep)
    its = tl.where(
        (cold & sig) & (qi > _QMIN),
        xs * (1.0e-3 * tl.maximum(0.0, qi) + qi * (2.61 * _V0S * ns) * libdevice.pow(ls, -(_V1S + 3.0))),
        0.0,
    )
    tl.store(sx + LQI * NX + LQS, tl.load(sx + LQI * NX + LQS) + its)
    ig = tl.where((cold & sig) & (qi > _QMIN) & (qg > _QMIN), xs * qi * 2.46 * libdevice.pow(rho * qg, 0.94878), 0.0)
    ig = tl.where(
        (cold & sig) & (qi > _QMIN) & (qr > _QMIN),
        ig + 1.72 * qi * libdevice.pow(rho * qr, 7.0 / 8.0),
        ig,
    )
    tl.store(sx + LQI * NX + LQG, ig)
    tl.store(
        sx + LQS * NX + LQG,
        tl.where((cold & sig) & (tl.minimum(qc, qs) > _QMIN) & (t > _THOM), 0.5 * qc * libdevice.pow(qs * rho, 0.75), 0.0),
    )
    tfr = _TM - 2.0
    rg = tl.where(
        (cold & sig) & (qr > _QMIN) & (t < tfr) & (t > _THOM) & ((dvsw + qc <= 0.0) | (qr > 0.1 * qc)),
        (libdevice.exp(0.66 * (tfr - t)) - 1.0) * (9.95e-5 * libdevice.pow(qr * rho, 1.75)),
        0.0,
    )
    rg = tl.where((cold & sig) & (qr > _QMIN) & (t < tfr) & (t <= _THOM), qr / dt, rg)
    rg = tl.where(
        (cold & sig) & (tl.minimum(qi, qr) > _QMIN) & (qs > 1.0e-7),
        rg + 1.24e-3 * (qi / mi) * libdevice.pow(rho * qr, 13.0 / 8.0),
        rg,
    )
    tl.store(sx + LQR * NX + LQG, rg)
    nuc = tl.where(
        cold
        & (qi <= _QMIN)
        & (((t < _TH2) & (dvsi > 0.0)) | ((t <= _TH1) & (qc > _QMIN))),
        tl.minimum(_M0 * ni, tl.maximum(0.0, dvsi)) / dt,
        0.0,
    )
    tl.store(sx + LQV * NX + LQI, tl.load(sx + LQV * NX + LQI) + nuc)
    warm = ~cold
    tr = tl.load(sx + LQC * NX + LQR) + tl.load(sx + LQC * NX + LQS) + tl.load(sx + LQC * NX + LQG)
    tl.store(sx + LQC * NX + LQR, tl.where(warm, tr, tl.load(sx + LQC * NX + LQR)))
    tl.store(sx + LQC * NX + LQS, tl.where(warm, 0.0, tl.load(sx + LQC * NX + LQS)))
    tl.store(sx + LQC * NX + LQG, tl.where(warm, 0.0, tl.load(sx + LQC * NX + LQG)))
    dvsw0 = qv - _qsat_rho(_TM, rho)
    vs = z
    a1 = 0.4182 * libdevice.sqrt(_V0S / 1.75e-5)
    vs_c = (4.0 * ns * eta / rho) * (1.0 + a1 * libdevice.pow(ls, -(_V1S + 1.0) / 2.0)) * dvsi / (ls * ls + 1.0e-15)
    vs_c = tl.where(vs_c > 0.0, tl.minimum(vs_c, dvsi / dt - idep), vs_c)
    vs_c = tl.where(qs <= 1.0e-7, tl.minimum(vs_c, 0.0), vs_c)
    vs_w1 = (31282.3 / p + 0.241897) * tl.minimum(0.0, dvsw0) * libdevice.pow(qs * rho, 0.8)
    vs_w2 = (0.28003 - 0.146293e-6 * p) * dvsw * libdevice.pow(qs * rho, 0.8)
    vs = tl.where(sig & (qs > _QMIN) & (t < _TM), vs_c, vs)
    vs = tl.where(sig & (qs > _QMIN) & (t >= _TM) & (t > (_TM - _TX * dvsw0)), vs_w1, vs)
    vs = tl.where(sig & (qs > _QMIN) & (t >= _TM) & (t <= (_TM - _TX * dvsw0)), vs_w2, vs)
    vs = tl.where(sig & (qs > _QMIN), tl.maximum(vs, -qs / dt), 0.0)
    tl.store(sx + LQV * NX + LQS, tl.maximum(vs, 0.0))
    tl.store(sx + LQS * NX + LQV, -tl.minimum(vs, 0.0))
    vg = tl.where(
        sig & (qg > _QMIN) & (t < _TM),
        (0.398561 - 0.00152398 * t + 2554.99 / p + 2.6531e-7 * p) * dvsi * libdevice.pow(qg * rho, 0.6),
        0.0,
    )
    vg = tl.where(
        sig & (qg > _QMIN) & (t >= _TM) & (t > (_TM - _TX * dvsw0)),
        (0.153907 - 7.86703e-07 * p) * tl.minimum(0.0, dvsw0) * libdevice.pow(qg * rho, 0.6),
        vg,
    )
    vg = tl.where(
        sig & (qg > _QMIN) & (t >= _TM) & (t <= (_TM - _TX * dvsw0)),
        (0.0418521 - 4.7524e-8 * p) * dvsw * libdevice.pow(qg * rho, 0.6),
        vg,
    )
    vg = tl.where(sig & (qg > _QMIN), tl.maximum(vg, -qg / dt), 0.0)
    tl.store(sx + LQV * NX + LQG, tl.maximum(vg, 0.0))
    tl.store(sx + LQG * NX + LQV, -tl.minimum(vg, 0.0))
    sr = tl.where(
        sig & (t > tl.maximum(_TM, _TM - _TX * dvsw0)) & (qs > _QMIN),
        (79.6863 / p + 0.612654e-3) * (t - _TM + (_TX - 389.5) * dvsw0) * libdevice.pow(qs * rho, 0.8),
        0.0,
    )
    tl.store(sx + LQS * NX + LQR, sr)
    gr = tl.where(
        sig & (t > tl.maximum(_TM, _TM - _TX * dvsw0)) & (qg > _QMIN),
        (12.31698 / p + 7.39441e-05) * (t - _TM + (_TX - 389.5) * dvsw0) * libdevice.pow(qg * rho, 0.6),
        0.0,
    )
    tl.store(sx + LQG * NX + LQR, gr)


# Outputs are updated in place; autotune trials must each start from the same fields.
@triton.autotune(
    configs=_configs(),
    key=["ke", "nvec"],
    cache_results=True,
    restore_value=["t", "qv", "qc", "qi", "qr", "qs", "qg", "prr", "pri", "prs", "prg", "pflx", "pre"],
)
@triton.jit(do_not_specialize=["ke", "nvec", "ivstart", "kstart"])
def _graupel_column(
    dz,
    p,
    rho,
    t,
    qv,
    qc,
    qi,
    qr,
    qs,
    qg,
    qnc,
    prr,
    pri,
    prs,
    prg,
    pflx,
    pre,
    dt,
    ivstart,
    kstart,
    nvec,
    ke,
    stride,
    sx_ptr,
    sk_ptr,
    dq_ptr,
):
    iv = tl.program_id(0) + ivstart
    if iv >= nvec:
        return
    fd = dz.dtype.element_ty
    zero = tl.full((), 0.0, fd)
    dt_f = tl.full((), dt, fd)
    sx = sx_ptr + iv * NX * NX
    sk = sk_ptr + iv * NX
    dq = dq_ptr + iv * NX
    nc = tl.load(qnc + ivstart)
    km_r = ke
    km_i = ke
    km_s = ke
    km_g = ke
    fl_r = zero
    fl_i = zero
    fl_s = zero
    fl_g = zero
    vt_r = zero
    vt_i = zero
    vt_s = zero
    vt_g = zero
    eflx = zero
    for k in range(kstart, ke):
        off = k * stride + iv
        qrk = tl.load(qr + off)
        qik = tl.load(qi + off)
        qsk = tl.load(qs + off)
        qgk = tl.load(qg + off)
        km_r = tl.where(qrk > _QMIN, tl.minimum(km_r, k), km_r)
        km_i = tl.where(qik > _QMIN, tl.minimum(km_i, k), km_i)
        km_s = tl.where(qsk > _QMIN, tl.minimum(km_s, k), km_s)
        km_g = tl.where(qgk > _QMIN, tl.minimum(km_g, k), km_g)
        tk = tl.load(t + off)
        pk = tl.load(p + off)
        rh = tl.load(rho + off)
        qvk = tl.load(qv + off)
        qck = tl.load(qc + off)
        cold_v = (tk < _TH2) & (qvk > _qsat_ice(tk, rh))
        peak = tl.maximum(tl.maximum(tl.maximum(qck, qrk), tl.maximum(qsk, qik)), qgk)
        active = (peak > _QMIN) | cold_v
        qx0 = qrk
        qx1 = qik
        qx2 = qsk
        qx3 = qgk
        qx4 = qck
        qx5 = qvk
        sig = tl.maximum(tl.maximum(qx2, qx1), qx3) > _QMIN
        _process_rates(tk, pk, rh, qx5, qx4, qx1, qx0, qx2, qx3, nc, dt_f, sig, sx)
        for iqx in tl.static_range(NX):
            ssum = zero
            lim = sig | (iqx == LQC) | (iqx == LQV) | (iqx == LQR)
            for j in tl.static_range(NX):
                ssum = ssum + tl.load(sx + iqx * NX + j)
            tl.store(sk + iqx, tl.where(lim, ssum, 0.0))
            qxi = tl.where(iqx == LQR, qx0, tl.where(iqx == LQI, qx1, tl.where(iqx == LQS, qx2, tl.where(iqx == LQG, qx3, tl.where(iqx == LQC, qx4, qx5)))))
            stot = qxi / dt_f
            skv = tl.load(sk + iqx)
            over = lim & (skv > stot) & (qxi > _QMIN)
            for j in tl.static_range(NX):
                v = tl.load(sx + iqx * NX + j)
                tl.store(sx + iqx * NX + j, tl.where(over, v * stot / skv, v))
            ssum2 = zero
            for j in tl.static_range(NX):
                ssum2 = ssum2 + tl.load(sx + iqx * NX + j)
            tl.store(sk + iqx, tl.where(lim, tl.where(over, ssum2, skv), 0.0))
        for iqx in tl.static_range(NX):
            acc = zero
            for j in tl.static_range(NX):
                acc = acc + tl.load(sx + j * NX + iqx)
            tl.store(dq + iqx, acc - tl.load(sk + iqx))
            qxi = tl.where(iqx == LQR, qx0, tl.where(iqx == LQI, qx1, tl.where(iqx == LQS, qx2, tl.where(iqx == LQG, qx3, tl.where(iqx == LQC, qx4, qx5)))))
            qnew = tl.maximum(zero, qxi + tl.load(dq + iqx) * dt_f)
            qx0 = tl.where(iqx == LQR, qnew, qx0)
            qx1 = tl.where(iqx == LQI, qnew, qx1)
            qx2 = tl.where(iqx == LQS, qnew, qx2)
            qx3 = tl.where(iqx == LQG, qnew, qx3)
            qx4 = tl.where(iqx == LQC, qnew, qx4)
            qx5 = tl.where(iqx == LQV, qnew, qx5)
        qice = qx2 + qx1 + qx3
        qliq = qx4 + qx0
        qtot = qx5 + qice + qliq
        cv = _CVD + (_CVV - _CVD) * qtot + (_CLW - _CVV) * qliq + (_CI - _CVV) * qice
        heat = (tl.load(dq + LQC) + tl.load(dq + LQR)) * (_LVC - (_CLW - _CVV) * tk)
        heat = heat + (tl.load(dq + LQI) + tl.load(dq + LQS) + tl.load(dq + LQG)) * (_LSC - (_CI - _CVV) * tk)
        tk = tl.where(active, tk + dt_f * heat / cv, tk)
        tl.store(t + off, tk)
        tl.store(qr + off, tl.where(active, qx0, qrk))
        tl.store(qi + off, tl.where(active, qx1, qik))
        tl.store(qs + off, tl.where(active, qx2, qsk))
        tl.store(qg + off, tl.where(active, qx3, qgk))
        tl.store(qc + off, tl.where(active, qx4, qck))
        tl.store(qv + off, tl.where(active, qx5, qvk))
    for k in range(kstart, ke):
        off = k * stride + iv
        kp1 = tl.minimum(ke - 1, k + 1)
        kfirst = tl.minimum(tl.minimum(km_r, km_i), tl.minimum(km_s, km_g))
        reached = k >= kfirst
        qck = tl.load(qc + off)
        qrk = tl.load(qr + off)
        qik = tl.load(qi + off)
        qsk = tl.load(qs + off)
        qgk = tl.load(qg + off)
        qvk = tl.load(qv + off)
        tk = tl.load(t + off)
        rh = tl.load(rho + off)
        dzv = tl.load(dz + off)
        qliq = qck + qrk
        qice = qsk + qik + qgk
        eint = _ie(tk, qvk, qliq, qice, rh, dzv) + eflx
        zeta = dt_f / (tl.full((), 2.0, fd) * dzv)
        xrho = libdevice.sqrt(_R00 / rh)
        # rain
        do_r = reached & (k >= km_r)
        rho_x = qrk * rh
        fin = rho_x / zeta + 2.0 * fl_r
        fp = tl.minimum(rho_x * xrho * _fall(rho_x, 14.58, 0.111, 1.0e-12), fin)
        qnew = zeta * (fin - fp) / ((1.0 + zeta * vt_r) * rh)
        fl_r = tl.where(do_r, (qnew * rh * vt_r + fp) * 0.5, fl_r)
        vt_r = tl.where(do_r, xrho * _fall((qnew + tl.load(qr + kp1 * stride + iv)) * 0.5 * rh, 14.58, 0.111, 1.0e-12), vt_r)
        qrk = tl.where(do_r, qnew, qrk)
        # ice
        do_i = reached & (k >= km_i)
        rho_x = qik * rh
        vc = libdevice.pow(xrho, 2.0 / 3.0)
        fin = rho_x / zeta + 2.0 * fl_i
        fp = tl.minimum(rho_x * vc * _fall(rho_x, 1.25, 0.160, 1.0e-12), fin)
        qnew = zeta * (fin - fp) / ((1.0 + zeta * vt_i) * rh)
        fl_i = tl.where(do_i, (qnew * rh * vt_i + fp) * 0.5, fl_i)
        vt_i = tl.where(do_i, vc * _fall((qnew + tl.load(qi + kp1 * stride + iv)) * 0.5 * rh, 1.25, 0.160, 1.0e-12), vt_i)
        qik = tl.where(do_i, qnew, qik)
        # snow
        do_s = reached & (k >= km_s)
        ns = _snow_n(tk, rh, qsk)
        vc = xrho * libdevice.pow(ns, -1.0 / 6.0)
        rho_x = qsk * rh
        fin = rho_x / zeta + 2.0 * fl_s
        fp = tl.minimum(rho_x * vc * _fall(rho_x, 57.80, 0.5 / 3.0, 1.0e-12), fin)
        qnew = zeta * (fin - fp) / ((1.0 + zeta * vt_s) * rh)
        fl_s = tl.where(do_s, (qnew * rh * vt_s + fp) * 0.5, fl_s)
        vt_s = tl.where(do_s, vc * _fall((qnew + tl.load(qs + kp1 * stride + iv)) * 0.5 * rh, 57.80, 0.5 / 3.0, 1.0e-12), vt_s)
        qsk = tl.where(do_s, qnew, qsk)
        # graupel
        do_g = reached & (k >= km_g)
        rho_x = qgk * rh
        fin = rho_x / zeta + 2.0 * fl_g
        fp = tl.minimum(rho_x * xrho * _fall(rho_x, 12.24, 0.217, 1.0e-08), fin)
        qnew = zeta * (fin - fp) / ((1.0 + zeta * vt_g) * rh)
        fl_g = tl.where(do_g, (qnew * rh * vt_g + fp) * 0.5, fl_g)
        vt_g = tl.where(do_g, xrho * _fall((qnew + tl.load(qg + kp1 * stride + iv)) * 0.5 * rh, 12.24, 0.217, 1.0e-08), vt_g)
        qgk = tl.where(do_g, qnew, qgk)
        ice_fl = fl_s + fl_i + fl_g
        tk1 = tl.load(t + kp1 * stride + iv)
        energy_flux = dt_f * (fl_r * (_CLW * tk - _CVD * tk1 - _LVC) + ice_fl * (_CI * tk - _CVD * tk1 - _LSC))
        eflx = tl.where(reached, energy_flux, eflx)
        qliq = qck + qrk
        qice = qsk + qik + qgk
        tnew = _t_ie(eint - eflx, qvk, qliq, qice, rh, dzv)
        tl.store(pflx + off, tl.where(reached, ice_fl + fl_r, tl.load(pflx + off)))
        tl.store(t + off, tl.where(reached, tnew, tk))
        tl.store(qr + off, qrk)
        tl.store(qi + off, qik)
        tl.store(qs + off, qsk)
        tl.store(qg + off, qgk)
    if kstart < ke:
        tl.store(prr + iv, fl_r)
        tl.store(pri + iv, fl_i)
        tl.store(prs + iv, fl_s)
        tl.store(prg + iv, fl_g)
        tl.store(pre + iv, eflx / dt_f)


def _graupel_step(dz, p, rho, t, qv, qc, qi, qr, qs, qg, qnc, prr_gsp, pri_gsp, prs_gsp, prg_gsp, pflx, pre_gsp, dt, ivstart, kstart, nvec, ke):
    ivstart = int(ivstart)
    kstart = int(kstart)
    nvec = int(nvec)
    ke = int(ke)
    dt = float(dt)
    import torch

    stride = int(nvec)
    scratch_sx = torch.empty((nvec, 36), dtype=t.dtype, device=t.device)
    scratch_sk = torch.empty((nvec, 6), dtype=t.dtype, device=t.device)
    scratch_dq = torch.empty((nvec, 6), dtype=t.dtype, device=t.device)
    grid = (max(0, nvec - ivstart),)
    _graupel_column[grid](
        dz,
        p,
        rho,
        t,
        qv,
        qc,
        qi,
        qr,
        qs,
        qg,
        qnc,
        prr_gsp,
        pri_gsp,
        prs_gsp,
        prg_gsp,
        pflx,
        pre_gsp,
        dt,
        ivstart,
        kstart,
        nvec,
        ke,
        stride,
        scratch_sx,
        scratch_sk,
        scratch_dq,
    )


def aes_graupel(
    dz,
    p,
    pflx,
    pre_gsp,
    prg_gsp,
    pri_gsp,
    prr_gsp,
    prs_gsp,
    qc,
    qg,
    qi,
    qnc,
    qr,
    qs,
    qv,
    rho,
    t,
    dt,
    ivstart,
    kstart,
    nvec,
    ke,
    nsteps,
):
    import torch

    nsteps = int(nsteps)
    t0 = t.clone()
    qv0 = qv.clone()
    for step in range(nsteps):
        if step > 0:
            t.copy_((t0 + t) * 0.5)
            qv.copy_((qv0 + qv) * 0.5)
        pflx.zero_()
        _graupel_step(
            dz,
            p,
            rho,
            t,
            qv,
            qc,
            qi,
            qr,
            qs,
            qg,
            qnc,
            prr_gsp,
            pri_gsp,
            prs_gsp,
            prg_gsp,
            pflx,
            pre_gsp,
            dt,
            ivstart,
            kstart,
            nvec,
            ke,
        )
