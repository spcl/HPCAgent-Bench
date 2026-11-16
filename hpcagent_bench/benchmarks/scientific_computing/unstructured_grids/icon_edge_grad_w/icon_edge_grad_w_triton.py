import torch
import triton
import triton.language as tl


def get_configs():
    return [
            triton.Config({"BLOCK_K": 16,  "BLOCK_N": 64}, num_warps=4),
            triton.Config({"BLOCK_K": 32,  "BLOCK_N": 32}, num_warps=4),
            triton.Config({"BLOCK_K": 32,  "BLOCK_N": 64}, num_warps=4),
            triton.Config({"BLOCK_K": 64,  "BLOCK_N": 32}, num_warps=4),
            triton.Config({"BLOCK_K": 64,  "BLOCK_N": 64}, num_warps=8),
            triton.Config({"BLOCK_K": 128, "BLOCK_N": 32}, num_warps=8),
            ]

@triton.autotune(configs=get_configs(), key=["nlev", "nproma", "nblks_e"])
@triton.jit
def _icon_edge_grad_w_kernel(w_ptr,
                             z_w_v_ptr,
                             cell_idx_ptr,
                             cell_blk_ptr,
                             vert_idx_ptr,
                             vert_blk_ptr,
                             vn_ie_ptr,
                             z_vt_ie_ptr,
                             inv_dual_ptr,
                             inv_primal_ptr,
                             tangent_ptr,
                             z_v_grad_w_ptr,
                             nproma,
                             nlev,
                             s_w_b, s_w_k, s_w_c,
                             s_zw_b, s_zw_k, s_zw_c,
                             s_vn_b, s_vn_k, s_vn_c,
                             s_zt_b, s_zt_k, s_zt_c,
                             s_zvgw_b, s_zvgw_k, s_zvgw_c,
                             s_id_b, s_id_c,
                             s_ip_b, s_ip_c,
                             s_tn_b, s_tn_c,
                             s_ci_n, s_ci_b, s_ci_c,
                             s_cb_n, s_cb_b, s_cb_c,
                             s_vi_n, s_vi_b, s_vi_c,
                             s_vb_n, s_vb_b, s_vb_c,
                             BLOCK_K: tl.constexpr,
                             BLOCK_N: tl.constexpr
                             ):
    pid_k = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_n = tl.program_id(2)
    offsets_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_k = offsets_k < nlev
    mask_n = offsets_n < nproma
    mask = mask_k[:, None] & mask_n[None, :]

    offsets_ci = pid_b * s_ci_b + offsets_n * s_ci_c
    c0i = tl.load(cell_idx_ptr + offsets_ci, mask=mask_n, other=0)
    c1i = tl.load(cell_idx_ptr + offsets_ci + s_ci_n, mask=mask_n, other=0)
    offsets_cb = pid_b * s_cb_b + offsets_n * s_cb_c
    c0b = tl.load(cell_blk_ptr + offsets_cb, mask=mask_n, other=0)
    c1b = tl.load(cell_blk_ptr + offsets_cb + s_cb_n, mask=mask_n, other=0)

    offsets_vi = pid_b * s_vi_b + offsets_n * s_vi_c
    v0i = tl.load(vert_idx_ptr + offsets_vi, mask=mask_n, other=0)
    v1i = tl.load(vert_idx_ptr + offsets_vi + s_vi_n, mask=mask_n, other=0)
    offsets_vb = pid_b * s_vb_b + offsets_n * s_vb_c
    v0b = tl.load(vert_blk_ptr + offsets_vb, mask=mask_n, other=0)
    v1b = tl.load(vert_blk_ptr + offsets_vb + s_vb_n, mask=mask_n, other=0)

    #cell values
    w0 = tl.load(w_ptr + c0b[None, :]*s_w_b + offsets_k[:, None]*s_w_k + c0i[None, :]*s_w_c, mask=mask, other=0.0)
    w1 = tl.load(w_ptr + c1b[None, :]*s_w_b + offsets_k[:, None]*s_w_k + c1i[None, :]*s_w_c, mask=mask, other=0.0)
    #vertex values
    zvw0 = tl.load(z_w_v_ptr + v0b[None, :]*s_zw_b + offsets_k[:, None]*s_zw_k + v0i[None, :]*s_zw_c, mask=mask, other=0.0)
    zvw1 = tl.load(z_w_v_ptr + v1b[None, :]*s_zw_b + offsets_k[:, None]*s_zw_k + v1i[None, :]*s_zw_c, mask=mask, other=0.0)

    vn = tl.load(vn_ie_ptr + pid_b * s_vn_b + offsets_k[:, None] * s_vn_k + offsets_n[None, :] * s_vn_c, mask=mask, other=0.0)
    idl = tl.load(inv_dual_ptr + pid_b * s_id_b + offsets_n * s_id_c, mask=mask_n, other=0.0)
    zvt = tl.load(z_vt_ie_ptr + pid_b * s_zt_b + offsets_k[:, None] * s_zt_k + offsets_n[None, :] * s_zt_c, mask=mask, other=0.0)
    tng = tl.load(tangent_ptr + pid_b * s_tn_b + offsets_n * s_tn_c, mask=mask_n, other=0.0)
    ipr = tl.load(inv_primal_ptr + pid_b * s_ip_b + offsets_n * s_ip_c, mask=mask_n, other=0.0)

    z_v_grad_w = vn * idl[None, :] * (w0 -w1) + zvt * ipr[None, :] * tng[None, :] * (zvw0 - zvw1)

    tl.store(z_v_grad_w_ptr + pid_b * s_zvgw_b + offsets_k[:, None] * s_zvgw_k + offsets_n[None, :] * s_zvgw_c, z_v_grad_w.to(z_v_grad_w_ptr.dtype.element_ty), mask=mask)



def icon_edge_grad_w(w: torch.Tensor,
                     z_w_v: torch.Tensor,
                     cell_idx: torch.Tensor,
                     cell_blk: torch.Tensor,
                     vert_idx: torch.Tensor,
                     vert_blk: torch.Tensor,
                     vn_ie: torch.Tensor,
                     z_vt_ie: torch.Tensor,
                     inv_dual: torch.Tensor,
                     inv_primal: torch.Tensor,
                     tangent: torch.Tensor,
                     z_v_grad_w: torch.Tensor,
                     nproma: int,
                     nlev: int,
                     nblks_c: int,
                     nblks_e: int,
                     nblks_v: int,
                     ):
    grid = lambda meta: (triton.cdiv(nlev, meta['BLOCK_K']), nblks_e, triton.cdiv(nproma, meta['BLOCK_N']))
    _icon_edge_grad_w_kernel[grid](w, z_w_v, cell_idx, cell_blk,
                                   vert_idx, vert_blk, vn_ie, z_vt_ie,
                                   inv_dual, inv_primal, tangent, z_v_grad_w,
                                   nproma, nlev,
                                   *w.stride(),
                                   *z_w_v.stride(),
                                   *vn_ie.stride(),
                                   *z_vt_ie.stride(),
                                   *z_v_grad_w.stride(),
                                   *inv_dual.stride(),
                                   *inv_primal.stride(),
                                   *tangent.stride(),
                                   *cell_idx.stride(),
                                   *cell_blk.stride(),
                                   *vert_idx.stride(),
                                   *vert_blk.stride(),)