import triton
import triton.language as tl

_NCLV = tl.constexpr(5)
_QTMST = tl.constexpr(1.0 / 3600.0)


@triton.autotune(
    configs=[triton.Config({}, num_warps=nw) for nw in [1, 2, 4]],
    key=["nlev", "klon"],
    cache_results=True,
    restore_value=["flux_ptr"],
)
@triton.jit(do_not_specialize=["nlev", "klon", "ncldtop", "nclv"])
def _solver_column_kernel(
    q_ptr,
    zdtgdp_ptr,
    zrdtgdp_ptr,
    rho_ptr,
    vfall_ptr,
    solqa_ptr,
    solqb_ptr,
    latent_ptr,
    flux_ptr,
    tnd_cld_ptr,
    tnd_t_ptr,
    tnd_q_ptr,
    nlev,
    klon,
    ncldtop,
    nclv,
):
    jl = tl.program_id(0)
    if jl >= klon:
        return

    for s in tl.static_range(_NCLV):
        tl.store(flux_ptr + (s * (nlev + 1) + ncldtop) * klon + jl, 0.0)

    for jk in range(ncldtop, nlev):
        zdt = tl.load(zdtgdp_ptr + jk * klon + jl)
        zrd = tl.load(zrdtgdp_ptr + jk * klon + jl)
        rh = tl.load(rho_ptr + jk * klon + jl)
        for s in tl.static_range(_NCLV):
            vf = tl.load(vfall_ptr + s)
            sink = zdt * vf * rh
            src = tl.load(flux_ptr + (s * (nlev + 1) + jk) * klon + jl) * zdt
            qv = tl.load(q_ptr + (s * nlev + jk) * klon + jl)
            guess = tl.maximum((qv + src) / (1.0 + sink), 0.0)
            tl.store(flux_ptr + (s * (nlev + 1) + jk + 1) * klon + jl, sink * guess * zrd)

    for jk in range(ncldtop, nlev):

        zdt = tl.load(zdtgdp_ptr + jk * klon + jl)
        rh = tl.load(rho_ptr + jk * klon + jl)
        fs0 = tl.load(flux_ptr + (0 * (nlev + 1) + jk) * klon + jl) * zdt
        fs1 = tl.load(flux_ptr + (1 * (nlev + 1) + jk) * klon + jl) * zdt
        fs2 = tl.load(flux_ptr + (2 * (nlev + 1) + jk) * klon + jl) * zdt
        fs3 = tl.load(flux_ptr + (3 * (nlev + 1) + jk) * klon + jl) * zdt
        fs4 = tl.load(flux_ptr + (4 * (nlev + 1) + jk) * klon + jl) * zdt
        fk0 = zdt * tl.load(vfall_ptr + 0) * rh
        fk1 = zdt * tl.load(vfall_ptr + 1) * rh
        fk2 = zdt * tl.load(vfall_ptr + 2) * rh
        fk3 = zdt * tl.load(vfall_ptr + 3) * rh
        fk4 = zdt * tl.load(vfall_ptr + 4) * rh

        l00 = 1.0 + fk0 + tl.load(solqb_ptr + ((0 * _NCLV + 0) * nlev + jk) * klon + jl) + tl.load(solqb_ptr + ((0 * _NCLV + 1) * nlev + jk) * klon + jl) + tl.load(solqb_ptr + ((0 * _NCLV + 2) * nlev + jk) * klon + jl) + tl.load(solqb_ptr + ((0 * _NCLV + 3) * nlev + jk) * klon + jl) + tl.load(solqb_ptr + ((0 * _NCLV + 4) * nlev + jk) * klon + jl)
        l01 = -tl.load(solqb_ptr + ((0 * _NCLV + 1) * nlev + jk) * klon + jl)
        l02 = -tl.load(solqb_ptr + ((0 * _NCLV + 2) * nlev + jk) * klon + jl)
        l03 = -tl.load(solqb_ptr + ((0 * _NCLV + 3) * nlev + jk) * klon + jl)
        l04 = -tl.load(solqb_ptr + ((0 * _NCLV + 4) * nlev + jk) * klon + jl)
        l11 = 1.0 + fk1 + tl.load(solqb_ptr + ((1 * _NCLV + 0) * nlev + jk) * klon + jl) + tl.load(solqb_ptr + ((1 * _NCLV + 1) * nlev + jk) * klon + jl) + tl.load(solqb_ptr + ((1 * _NCLV + 2) * nlev + jk) * klon + jl) + tl.load(solqb_ptr + ((1 * _NCLV + 3) * nlev + jk) * klon + jl) + tl.load(solqb_ptr + ((1 * _NCLV + 4) * nlev + jk) * klon + jl)
        l10 = -tl.load(solqb_ptr + ((1 * _NCLV + 0) * nlev + jk) * klon + jl)
        l12 = -tl.load(solqb_ptr + ((1 * _NCLV + 2) * nlev + jk) * klon + jl)
        l13 = -tl.load(solqb_ptr + ((1 * _NCLV + 3) * nlev + jk) * klon + jl)
        l14 = -tl.load(solqb_ptr + ((1 * _NCLV + 4) * nlev + jk) * klon + jl)
        l22 = 1.0 + fk2 + tl.load(solqb_ptr + ((2 * _NCLV + 0) * nlev + jk) * klon + jl) + tl.load(solqb_ptr + ((2 * _NCLV + 1) * nlev + jk) * klon + jl) + tl.load(solqb_ptr + ((2 * _NCLV + 2) * nlev + jk) * klon + jl) + tl.load(solqb_ptr + ((2 * _NCLV + 3) * nlev + jk) * klon + jl) + tl.load(solqb_ptr + ((2 * _NCLV + 4) * nlev + jk) * klon + jl)
        l20 = -tl.load(solqb_ptr + ((2 * _NCLV + 0) * nlev + jk) * klon + jl)
        l21 = -tl.load(solqb_ptr + ((2 * _NCLV + 1) * nlev + jk) * klon + jl)
        l23 = -tl.load(solqb_ptr + ((2 * _NCLV + 3) * nlev + jk) * klon + jl)
        l24 = -tl.load(solqb_ptr + ((2 * _NCLV + 4) * nlev + jk) * klon + jl)
        l33 = 1.0 + fk3 + tl.load(solqb_ptr + ((3 * _NCLV + 0) * nlev + jk) * klon + jl) + tl.load(solqb_ptr + ((3 * _NCLV + 1) * nlev + jk) * klon + jl) + tl.load(solqb_ptr + ((3 * _NCLV + 2) * nlev + jk) * klon + jl) + tl.load(solqb_ptr + ((3 * _NCLV + 3) * nlev + jk) * klon + jl) + tl.load(solqb_ptr + ((3 * _NCLV + 4) * nlev + jk) * klon + jl)
        l30 = -tl.load(solqb_ptr + ((3 * _NCLV + 0) * nlev + jk) * klon + jl)
        l31 = -tl.load(solqb_ptr + ((3 * _NCLV + 1) * nlev + jk) * klon + jl)
        l32 = -tl.load(solqb_ptr + ((3 * _NCLV + 2) * nlev + jk) * klon + jl)
        l34 = -tl.load(solqb_ptr + ((3 * _NCLV + 4) * nlev + jk) * klon + jl)
        l44 = 1.0 + fk4 + tl.load(solqb_ptr + ((4 * _NCLV + 0) * nlev + jk) * klon + jl) + tl.load(solqb_ptr + ((4 * _NCLV + 1) * nlev + jk) * klon + jl) + tl.load(solqb_ptr + ((4 * _NCLV + 2) * nlev + jk) * klon + jl) + tl.load(solqb_ptr + ((4 * _NCLV + 3) * nlev + jk) * klon + jl) + tl.load(solqb_ptr + ((4 * _NCLV + 4) * nlev + jk) * klon + jl)
        l40 = -tl.load(solqb_ptr + ((4 * _NCLV + 0) * nlev + jk) * klon + jl)
        l41 = -tl.load(solqb_ptr + ((4 * _NCLV + 1) * nlev + jk) * klon + jl)
        l42 = -tl.load(solqb_ptr + ((4 * _NCLV + 2) * nlev + jk) * klon + jl)
        l43 = -tl.load(solqb_ptr + ((4 * _NCLV + 3) * nlev + jk) * klon + jl)
        q0 = tl.load(q_ptr + (0 * nlev + jk) * klon + jl) + tl.load(solqa_ptr + ((0 * _NCLV + 0) * nlev + jk) * klon + jl) + tl.load(solqa_ptr + ((1 * _NCLV + 0) * nlev + jk) * klon + jl) + tl.load(solqa_ptr + ((2 * _NCLV + 0) * nlev + jk) * klon + jl) + tl.load(solqa_ptr + ((3 * _NCLV + 0) * nlev + jk) * klon + jl) + tl.load(solqa_ptr + ((4 * _NCLV + 0) * nlev + jk) * klon + jl) + fs0
        q1 = tl.load(q_ptr + (1 * nlev + jk) * klon + jl) + tl.load(solqa_ptr + ((0 * _NCLV + 1) * nlev + jk) * klon + jl) + tl.load(solqa_ptr + ((1 * _NCLV + 1) * nlev + jk) * klon + jl) + tl.load(solqa_ptr + ((2 * _NCLV + 1) * nlev + jk) * klon + jl) + tl.load(solqa_ptr + ((3 * _NCLV + 1) * nlev + jk) * klon + jl) + tl.load(solqa_ptr + ((4 * _NCLV + 1) * nlev + jk) * klon + jl) + fs1
        q2 = tl.load(q_ptr + (2 * nlev + jk) * klon + jl) + tl.load(solqa_ptr + ((0 * _NCLV + 2) * nlev + jk) * klon + jl) + tl.load(solqa_ptr + ((1 * _NCLV + 2) * nlev + jk) * klon + jl) + tl.load(solqa_ptr + ((2 * _NCLV + 2) * nlev + jk) * klon + jl) + tl.load(solqa_ptr + ((3 * _NCLV + 2) * nlev + jk) * klon + jl) + tl.load(solqa_ptr + ((4 * _NCLV + 2) * nlev + jk) * klon + jl) + fs2
        q3 = tl.load(q_ptr + (3 * nlev + jk) * klon + jl) + tl.load(solqa_ptr + ((0 * _NCLV + 3) * nlev + jk) * klon + jl) + tl.load(solqa_ptr + ((1 * _NCLV + 3) * nlev + jk) * klon + jl) + tl.load(solqa_ptr + ((2 * _NCLV + 3) * nlev + jk) * klon + jl) + tl.load(solqa_ptr + ((3 * _NCLV + 3) * nlev + jk) * klon + jl) + tl.load(solqa_ptr + ((4 * _NCLV + 3) * nlev + jk) * klon + jl) + fs3
        q4 = tl.load(q_ptr + (4 * nlev + jk) * klon + jl) + tl.load(solqa_ptr + ((0 * _NCLV + 4) * nlev + jk) * klon + jl) + tl.load(solqa_ptr + ((1 * _NCLV + 4) * nlev + jk) * klon + jl) + tl.load(solqa_ptr + ((2 * _NCLV + 4) * nlev + jk) * klon + jl) + tl.load(solqa_ptr + ((3 * _NCLV + 4) * nlev + jk) * klon + jl) + tl.load(solqa_ptr + ((4 * _NCLV + 4) * nlev + jk) * klon + jl) + fs4
        l01 = l01 / l00
        l11 = l11 - l01 * l10
        l21 = l21 - l01 * l20
        l31 = l31 - l01 * l30
        l41 = l41 - l01 * l40
        l02 = l02 / l00
        l12 = l12 - l02 * l10
        l22 = l22 - l02 * l20
        l32 = l32 - l02 * l30
        l42 = l42 - l02 * l40
        l03 = l03 / l00
        l13 = l13 - l03 * l10
        l23 = l23 - l03 * l20
        l33 = l33 - l03 * l30
        l43 = l43 - l03 * l40
        l04 = l04 / l00
        l14 = l14 - l04 * l10
        l24 = l24 - l04 * l20
        l34 = l34 - l04 * l30
        l44 = l44 - l04 * l40
        l12 = l12 / l11
        l22 = l22 - l12 * l21
        l32 = l32 - l12 * l31
        l42 = l42 - l12 * l41
        l13 = l13 / l11
        l23 = l23 - l13 * l21
        l33 = l33 - l13 * l31
        l43 = l43 - l13 * l41
        l14 = l14 / l11
        l24 = l24 - l14 * l21
        l34 = l34 - l14 * l31
        l44 = l44 - l14 * l41
        l23 = l23 / l22
        l33 = l33 - l23 * l32
        l43 = l43 - l23 * l42
        l24 = l24 / l22
        l34 = l34 - l24 * l32
        l44 = l44 - l24 * l42
        l34 = l34 / l33
        l44 = l44 - l34 * l43
        q1 = q1 - l01 * q0
        q2 = q2 - l02 * q0
        q2 = q2 - l12 * q1
        q3 = q3 - l03 * q0
        q3 = q3 - l13 * q1
        q3 = q3 - l23 * q2
        q4 = q4 - l04 * q0
        q4 = q4 - l14 * q1
        q4 = q4 - l24 * q2
        q4 = q4 - l34 * q3
        q4 = q4 / l44
        q3 = q3 - l43 * q4
        q3 = q3 / l33
        q2 = q2 - l32 * q3
        q2 = q2 - l42 * q4
        q2 = q2 / l22
        q1 = q1 - l21 * q2
        q1 = q1 - l31 * q3
        q1 = q1 - l41 * q4
        q1 = q1 / l11
        q0 = q0 - l10 * q1
        q0 = q0 - l20 * q2
        q0 = q0 - l30 * q3
        q0 = q0 - l40 * q4
        q0 = q0 / l00
        d0 = tl.minimum(q0, 0.0)
        q0 = tl.maximum(q0, 0.0)
        q4 = q4 + d0
        d1 = tl.minimum(q1, 0.0)
        q1 = tl.maximum(q1, 0.0)
        q4 = q4 + d1
        d2 = tl.minimum(q2, 0.0)
        q2 = tl.maximum(q2, 0.0)
        q4 = q4 + d2
        d3 = tl.minimum(q3, 0.0)
        q3 = tl.maximum(q3, 0.0)
        q4 = q4 + d3
        tacc = 0.0
        fluxq0 = fs0 - fk0 * q0
        lat0 = tl.load(latent_ptr + 0)
        q_in0 = tl.load(q_ptr + (0 * nlev + jk) * klon + jl)
        tacc = tacc + lat0 * (q0 - q_in0 - fluxq0) * _QTMST
        tl.store(tnd_cld_ptr + (0 * nlev + jk) * klon + jl, (q0 - q_in0) * _QTMST)
        fluxq1 = fs1 - fk1 * q1
        lat1 = tl.load(latent_ptr + 1)
        q_in1 = tl.load(q_ptr + (1 * nlev + jk) * klon + jl)
        tacc = tacc + lat1 * (q1 - q_in1 - fluxq1) * _QTMST
        tl.store(tnd_cld_ptr + (1 * nlev + jk) * klon + jl, (q1 - q_in1) * _QTMST)
        fluxq2 = fs2 - fk2 * q2
        lat2 = tl.load(latent_ptr + 2)
        q_in2 = tl.load(q_ptr + (2 * nlev + jk) * klon + jl)
        tacc = tacc + lat2 * (q2 - q_in2 - fluxq2) * _QTMST
        tl.store(tnd_cld_ptr + (2 * nlev + jk) * klon + jl, (q2 - q_in2) * _QTMST)
        fluxq3 = fs3 - fk3 * q3
        lat3 = tl.load(latent_ptr + 3)
        q_in3 = tl.load(q_ptr + (3 * nlev + jk) * klon + jl)
        tacc = tacc + lat3 * (q3 - q_in3 - fluxq3) * _QTMST
        tl.store(tnd_cld_ptr + (3 * nlev + jk) * klon + jl, (q3 - q_in3) * _QTMST)
        tl.store(tnd_t_ptr + jk * klon + jl, tacc)
        q_in4 = tl.load(q_ptr + (4 * nlev + jk) * klon + jl)
        tl.store(tnd_q_ptr + jk * klon + jl, (q4 - q_in4) * _QTMST)
        tl.store(tnd_cld_ptr + (4 * nlev + jk) * klon + jl, 0.0)

def cloudsc_solver_carry(
    q,
    zdtgdp,
    zrdtgdp,
    rho,
    vfall,
    solqa,
    solqb,
    latent,
    flux,
    tnd_cld,
    tnd_t,
    tnd_q,
    nlev,
    klon,
    ncldtop,
    nclv,
):
    nv = int(nlev)
    kl = int(klon)
    top = int(ncldtop)
    nc = int(nclv)
    _solver_column_kernel[(kl,)](
        q,
        zdtgdp,
        zrdtgdp,
        rho,
        vfall,
        solqa,
        solqb,
        latent,
        flux,
        tnd_cld,
        tnd_t,
        tnd_q,
        nv,
        kl,
        top,
        nc,
    )

