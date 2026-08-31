/* DaCe AUTO-GENERATED FILE. DO NOT MODIFY */
#include <dace/dace.h>


struct velocity_tendencies_state_t {

};

void __program_velocity_tendencies_internal(velocity_tendencies_state_t*__state, bool * __restrict__ i_am_accel_node, bool * __restrict__ lextra_diffu, bool * __restrict__ lvert_nest, int * __restrict__ nflatlev, int * __restrict__ nrdmax, bool * __restrict__ p_diag_ddt_vn_adv_is_associated, double * __restrict__ p_diag_ddt_vn_apc_pc, bool * __restrict__ p_diag_ddt_vn_cor_is_associated, double * __restrict__ p_diag_ddt_vn_cor_pc, double * __restrict__ p_diag_ddt_w_adv_pc, double * __restrict__ p_diag_max_vcfl_dyn, double * __restrict__ p_diag_vn_ie, double * __restrict__ p_diag_vn_ie_ubc, double * __restrict__ p_diag_vt, double * __restrict__ p_diag_w_concorr_c, double * __restrict__ p_int_c_lin_e, double * __restrict__ p_int_cells_aw_verts, double * __restrict__ p_int_e_bln_c_s, double * __restrict__ p_int_geofac_grdiv, double * __restrict__ p_int_geofac_n2s, double * __restrict__ p_int_geofac_rot, double * __restrict__ p_int_rbf_vec_coeff_e, double * __restrict__ p_metrics_coeff1_dwdz, double * __restrict__ p_metrics_coeff2_dwdz, double * __restrict__ p_metrics_coeff_gradekin, double * __restrict__ p_metrics_ddqz_z_full_e, double * __restrict__ p_metrics_ddqz_z_half, double * __restrict__ p_metrics_ddxn_z_full, double * __restrict__ p_metrics_ddxt_z_full, double * __restrict__ p_metrics_deepatmo_gradh_ifc, double * __restrict__ p_metrics_deepatmo_gradh_mc, double * __restrict__ p_metrics_deepatmo_invr_ifc, double * __restrict__ p_metrics_deepatmo_invr_mc, double * __restrict__ p_metrics_wgtfac_c, double * __restrict__ p_metrics_wgtfac_e, double * __restrict__ p_metrics_wgtfacq_e, double * __restrict__ p_patch_cells_area, bool * __restrict__ p_patch_cells_decomp_info_owner_mask, int * __restrict__ p_patch_cells_edge_blk, int * __restrict__ p_patch_cells_edge_idx, int * __restrict__ p_patch_cells_end_block, int * __restrict__ p_patch_cells_end_index, int * __restrict__ p_patch_cells_neighbor_blk, int * __restrict__ p_patch_cells_neighbor_idx, int * __restrict__ p_patch_cells_start_block, int * __restrict__ p_patch_cells_start_index, double * __restrict__ p_patch_edges_area_edge, int * __restrict__ p_patch_edges_cell_blk, int * __restrict__ p_patch_edges_cell_idx, int * __restrict__ p_patch_edges_end_block, int * __restrict__ p_patch_edges_end_index, double * __restrict__ p_patch_edges_f_e, double * __restrict__ p_patch_edges_fn_e, double * __restrict__ p_patch_edges_ft_e, double * __restrict__ p_patch_edges_inv_dual_edge_length, double * __restrict__ p_patch_edges_inv_primal_edge_length, int * __restrict__ p_patch_edges_quad_blk, int * __restrict__ p_patch_edges_quad_idx, int * __restrict__ p_patch_edges_start_block, int * __restrict__ p_patch_edges_start_index, double * __restrict__ p_patch_edges_tangent_orientation, int * __restrict__ p_patch_edges_vertex_blk, int * __restrict__ p_patch_edges_vertex_idx, int * __restrict__ p_patch_id, int * __restrict__ p_patch_verts_cell_blk, int * __restrict__ p_patch_verts_cell_idx, int * __restrict__ p_patch_verts_edge_blk, int * __restrict__ p_patch_verts_edge_idx, int * __restrict__ p_patch_verts_end_block, int * __restrict__ p_patch_verts_end_index, int * __restrict__ p_patch_verts_start_block, int * __restrict__ p_patch_verts_start_index, double * __restrict__ p_prog_vn, double * __restrict__ p_prog_w, int * __restrict__ timer_intp, int * __restrict__ timer_solve_nh_veltend, double * __restrict__ z_kin_hor_e, double * __restrict__ z_vt_ie, double * __restrict__ z_w_concorr_me, double dt_linintp_ubc, double dtime, int istep, bool ldeepatmo, bool lvn_only, int nproma, int ntnd, int64_t offset_p_diag_ddt_vn_apc_pc_d0, int64_t offset_p_diag_ddt_vn_apc_pc_d1, int64_t offset_p_diag_ddt_vn_apc_pc_d2, int64_t offset_p_diag_ddt_vn_apc_pc_d3, int64_t offset_p_diag_ddt_vn_cor_pc_d0, int64_t offset_p_diag_ddt_vn_cor_pc_d1, int64_t offset_p_diag_ddt_vn_cor_pc_d2, int64_t offset_p_diag_ddt_vn_cor_pc_d3, int64_t offset_p_diag_ddt_w_adv_pc_d0, int64_t offset_p_diag_ddt_w_adv_pc_d1, int64_t offset_p_diag_ddt_w_adv_pc_d2, int64_t offset_p_diag_ddt_w_adv_pc_d3, int64_t offset_p_diag_vn_ie_d0, int64_t offset_p_diag_vn_ie_d1, int64_t offset_p_diag_vn_ie_d2, int64_t offset_p_diag_vn_ie_ubc_d0, int64_t offset_p_diag_vn_ie_ubc_d1, int64_t offset_p_diag_vn_ie_ubc_d2, int64_t offset_p_diag_vt_d0, int64_t offset_p_diag_vt_d1, int64_t offset_p_diag_vt_d2, int64_t offset_p_diag_w_concorr_c_d0, int64_t offset_p_diag_w_concorr_c_d1, int64_t offset_p_diag_w_concorr_c_d2, int64_t offset_p_int_c_lin_e_d0, int64_t offset_p_int_c_lin_e_d1, int64_t offset_p_int_c_lin_e_d2, int64_t offset_p_int_cells_aw_verts_d0, int64_t offset_p_int_cells_aw_verts_d1, int64_t offset_p_int_cells_aw_verts_d2, int64_t offset_p_int_e_bln_c_s_d0, int64_t offset_p_int_e_bln_c_s_d1, int64_t offset_p_int_e_bln_c_s_d2, int64_t offset_p_int_geofac_grdiv_d0, int64_t offset_p_int_geofac_grdiv_d1, int64_t offset_p_int_geofac_grdiv_d2, int64_t offset_p_int_geofac_n2s_d0, int64_t offset_p_int_geofac_n2s_d1, int64_t offset_p_int_geofac_n2s_d2, int64_t offset_p_int_geofac_rot_d0, int64_t offset_p_int_geofac_rot_d1, int64_t offset_p_int_geofac_rot_d2, int64_t offset_p_int_rbf_vec_coeff_e_d0, int64_t offset_p_int_rbf_vec_coeff_e_d1, int64_t offset_p_int_rbf_vec_coeff_e_d2, int64_t offset_p_metrics_coeff1_dwdz_d0, int64_t offset_p_metrics_coeff1_dwdz_d1, int64_t offset_p_metrics_coeff1_dwdz_d2, int64_t offset_p_metrics_coeff2_dwdz_d0, int64_t offset_p_metrics_coeff2_dwdz_d1, int64_t offset_p_metrics_coeff2_dwdz_d2, int64_t offset_p_metrics_coeff_gradekin_d0, int64_t offset_p_metrics_coeff_gradekin_d1, int64_t offset_p_metrics_coeff_gradekin_d2, int64_t offset_p_metrics_ddqz_z_full_e_d0, int64_t offset_p_metrics_ddqz_z_full_e_d1, int64_t offset_p_metrics_ddqz_z_full_e_d2, int64_t offset_p_metrics_ddqz_z_half_d0, int64_t offset_p_metrics_ddqz_z_half_d1, int64_t offset_p_metrics_ddqz_z_half_d2, int64_t offset_p_metrics_ddxn_z_full_d0, int64_t offset_p_metrics_ddxn_z_full_d1, int64_t offset_p_metrics_ddxn_z_full_d2, int64_t offset_p_metrics_ddxt_z_full_d0, int64_t offset_p_metrics_ddxt_z_full_d1, int64_t offset_p_metrics_ddxt_z_full_d2, int64_t offset_p_metrics_deepatmo_gradh_ifc_d0, int64_t offset_p_metrics_deepatmo_gradh_mc_d0, int64_t offset_p_metrics_deepatmo_invr_ifc_d0, int64_t offset_p_metrics_deepatmo_invr_mc_d0, int64_t offset_p_metrics_wgtfac_c_d0, int64_t offset_p_metrics_wgtfac_c_d1, int64_t offset_p_metrics_wgtfac_c_d2, int64_t offset_p_metrics_wgtfac_e_d0, int64_t offset_p_metrics_wgtfac_e_d1, int64_t offset_p_metrics_wgtfac_e_d2, int64_t offset_p_metrics_wgtfacq_e_d0, int64_t offset_p_metrics_wgtfacq_e_d1, int64_t offset_p_metrics_wgtfacq_e_d2, int64_t offset_p_patch_cells_area_d0, int64_t offset_p_patch_cells_area_d1, int64_t offset_p_patch_cells_decomp_info_owner_mask_d0, int64_t offset_p_patch_cells_decomp_info_owner_mask_d1, int64_t offset_p_patch_cells_edge_blk_d0, int64_t offset_p_patch_cells_edge_blk_d1, int64_t offset_p_patch_cells_edge_blk_d2, int64_t offset_p_patch_cells_edge_idx_d0, int64_t offset_p_patch_cells_edge_idx_d1, int64_t offset_p_patch_cells_edge_idx_d2, int64_t offset_p_patch_cells_neighbor_blk_d0, int64_t offset_p_patch_cells_neighbor_blk_d1, int64_t offset_p_patch_cells_neighbor_blk_d2, int64_t offset_p_patch_cells_neighbor_idx_d0, int64_t offset_p_patch_cells_neighbor_idx_d1, int64_t offset_p_patch_cells_neighbor_idx_d2, int64_t offset_p_patch_cells_start_block_d0, int64_t offset_p_patch_cells_start_index_d0, int64_t offset_p_patch_edges_area_edge_d0, int64_t offset_p_patch_edges_area_edge_d1, int64_t offset_p_patch_edges_cell_blk_d0, int64_t offset_p_patch_edges_cell_blk_d1, int64_t offset_p_patch_edges_cell_blk_d2, int64_t offset_p_patch_edges_cell_idx_d0, int64_t offset_p_patch_edges_cell_idx_d1, int64_t offset_p_patch_edges_cell_idx_d2, int64_t offset_p_patch_edges_f_e_d0, int64_t offset_p_patch_edges_f_e_d1, int64_t offset_p_patch_edges_fn_e_d0, int64_t offset_p_patch_edges_fn_e_d1, int64_t offset_p_patch_edges_ft_e_d0, int64_t offset_p_patch_edges_ft_e_d1, int64_t offset_p_patch_edges_inv_dual_edge_length_d0, int64_t offset_p_patch_edges_inv_dual_edge_length_d1, int64_t offset_p_patch_edges_inv_primal_edge_length_d0, int64_t offset_p_patch_edges_inv_primal_edge_length_d1, int64_t offset_p_patch_edges_quad_blk_d0, int64_t offset_p_patch_edges_quad_blk_d1, int64_t offset_p_patch_edges_quad_blk_d2, int64_t offset_p_patch_edges_quad_idx_d0, int64_t offset_p_patch_edges_quad_idx_d1, int64_t offset_p_patch_edges_quad_idx_d2, int64_t offset_p_patch_edges_start_block_d0, int64_t offset_p_patch_edges_start_index_d0, int64_t offset_p_patch_edges_tangent_orientation_d0, int64_t offset_p_patch_edges_tangent_orientation_d1, int64_t offset_p_patch_edges_vertex_blk_d0, int64_t offset_p_patch_edges_vertex_blk_d1, int64_t offset_p_patch_edges_vertex_blk_d2, int64_t offset_p_patch_edges_vertex_idx_d0, int64_t offset_p_patch_edges_vertex_idx_d1, int64_t offset_p_patch_edges_vertex_idx_d2, int64_t offset_p_patch_verts_cell_blk_d0, int64_t offset_p_patch_verts_cell_blk_d1, int64_t offset_p_patch_verts_cell_blk_d2, int64_t offset_p_patch_verts_cell_idx_d0, int64_t offset_p_patch_verts_cell_idx_d1, int64_t offset_p_patch_verts_cell_idx_d2, int64_t offset_p_patch_verts_edge_blk_d0, int64_t offset_p_patch_verts_edge_blk_d1, int64_t offset_p_patch_verts_edge_blk_d2, int64_t offset_p_patch_verts_edge_idx_d0, int64_t offset_p_patch_verts_edge_idx_d1, int64_t offset_p_patch_verts_edge_idx_d2, int64_t offset_p_patch_verts_start_block_d0, int64_t offset_p_patch_verts_start_index_d0, int64_t offset_p_prog_vn_d0, int64_t offset_p_prog_vn_d1, int64_t offset_p_prog_vn_d2, int64_t offset_p_prog_w_d0, int64_t offset_p_prog_w_d1, int64_t offset_p_prog_w_d2, int64_t p_diag_ddt_vn_apc_pc_d0, int64_t p_diag_ddt_vn_apc_pc_d1, int64_t p_diag_ddt_vn_apc_pc_d2, int64_t p_diag_ddt_vn_cor_pc_d0, int64_t p_diag_ddt_vn_cor_pc_d1, int64_t p_diag_ddt_vn_cor_pc_d2, int64_t p_diag_ddt_w_adv_pc_d0, int64_t p_diag_ddt_w_adv_pc_d1, int64_t p_diag_ddt_w_adv_pc_d2, int64_t p_diag_vn_ie_d0, int64_t p_diag_vn_ie_d1, int64_t p_diag_vn_ie_ubc_d0, int64_t p_diag_vn_ie_ubc_d1, int64_t p_diag_vt_d0, int64_t p_diag_vt_d1, int64_t p_diag_w_concorr_c_d0, int64_t p_diag_w_concorr_c_d1, int64_t p_int_c_lin_e_d0, int64_t p_int_c_lin_e_d1, int64_t p_int_cells_aw_verts_d0, int64_t p_int_cells_aw_verts_d1, int64_t p_int_e_bln_c_s_d0, int64_t p_int_e_bln_c_s_d1, int64_t p_int_geofac_grdiv_d0, int64_t p_int_geofac_grdiv_d1, int64_t p_int_geofac_n2s_d0, int64_t p_int_geofac_n2s_d1, int64_t p_int_geofac_rot_d0, int64_t p_int_geofac_rot_d1, int64_t p_int_rbf_vec_coeff_e_d0, int64_t p_int_rbf_vec_coeff_e_d1, int64_t p_metrics_coeff1_dwdz_d0, int64_t p_metrics_coeff1_dwdz_d1, int64_t p_metrics_coeff2_dwdz_d0, int64_t p_metrics_coeff2_dwdz_d1, int64_t p_metrics_coeff_gradekin_d0, int64_t p_metrics_coeff_gradekin_d1, int64_t p_metrics_ddqz_z_full_e_d0, int64_t p_metrics_ddqz_z_full_e_d1, int64_t p_metrics_ddqz_z_half_d0, int64_t p_metrics_ddqz_z_half_d1, int64_t p_metrics_ddxn_z_full_d0, int64_t p_metrics_ddxn_z_full_d1, int64_t p_metrics_ddxt_z_full_d0, int64_t p_metrics_ddxt_z_full_d1, int64_t p_metrics_wgtfac_c_d0, int64_t p_metrics_wgtfac_c_d1, int64_t p_metrics_wgtfac_e_d0, int64_t p_metrics_wgtfac_e_d1, int64_t p_metrics_wgtfacq_e_d0, int64_t p_metrics_wgtfacq_e_d1, int64_t p_patch_cells_area_d0, int64_t p_patch_cells_decomp_info_owner_mask_d0, int64_t p_patch_cells_edge_blk_d0, int64_t p_patch_cells_edge_blk_d1, int64_t p_patch_cells_edge_idx_d0, int64_t p_patch_cells_edge_idx_d1, int64_t p_patch_cells_neighbor_blk_d0, int64_t p_patch_cells_neighbor_blk_d1, int64_t p_patch_cells_neighbor_idx_d0, int64_t p_patch_cells_neighbor_idx_d1, int64_t p_patch_edges_area_edge_d0, int64_t p_patch_edges_cell_blk_d0, int64_t p_patch_edges_cell_blk_d1, int64_t p_patch_edges_cell_idx_d0, int64_t p_patch_edges_cell_idx_d1, int64_t p_patch_edges_f_e_d0, int64_t p_patch_edges_fn_e_d0, int64_t p_patch_edges_ft_e_d0, int64_t p_patch_edges_inv_dual_edge_length_d0, int64_t p_patch_edges_inv_primal_edge_length_d0, int64_t p_patch_edges_quad_blk_d0, int64_t p_patch_edges_quad_blk_d1, int64_t p_patch_edges_quad_idx_d0, int64_t p_patch_edges_quad_idx_d1, int64_t p_patch_edges_tangent_orientation_d0, int64_t p_patch_edges_vertex_blk_d0, int64_t p_patch_edges_vertex_blk_d1, int64_t p_patch_edges_vertex_idx_d0, int64_t p_patch_edges_vertex_idx_d1, int p_patch_nblks_c, int p_patch_nblks_e, int p_patch_nblks_v, int p_patch_nlev, int p_patch_nlevp1, int p_patch_nshift, int64_t p_patch_verts_cell_blk_d0, int64_t p_patch_verts_cell_blk_d1, int64_t p_patch_verts_cell_idx_d0, int64_t p_patch_verts_cell_idx_d1, int64_t p_patch_verts_edge_blk_d0, int64_t p_patch_verts_edge_blk_d1, int64_t p_patch_verts_edge_idx_d0, int64_t p_patch_verts_edge_idx_d1, int64_t p_prog_vn_d0, int64_t p_prog_vn_d1, int64_t p_prog_w_d0, int64_t p_prog_w_d1, int timers_level, int64_t z_kin_hor_e_d0, int64_t z_kin_hor_e_d1, int64_t z_vt_ie_d0, int64_t z_vt_ie_d1, int64_t z_w_concorr_me_d0, int64_t z_w_concorr_me_d1)
{
    bool *cfl_clipping;
    cfl_clipping = new (std::align_val_t(64)) bool [((nproma * (p_patch_nlevp1 - 1)) + nproma)];
    bool *levelmask;
    levelmask = new (std::align_val_t(64)) bool [p_patch_nlev];
    bool *levmask;
    levmask = new (std::align_val_t(64)) bool [((p_patch_nblks_c * (p_patch_nlev - 1)) + p_patch_nblks_c)];
    double *vcflmax;
    vcflmax = new (std::align_val_t(64)) double [p_patch_nblks_c];
    double *z_ekinh;
    z_ekinh = new (std::align_val_t(64)) double [((((nproma * p_patch_nlev) * (p_patch_nblks_c - 1)) + (nproma * (p_patch_nlev - 1))) + nproma)];
    double *z_v_grad_w;
    z_v_grad_w = new (std::align_val_t(64)) double [((((nproma * p_patch_nlev) * (p_patch_nblks_e - 1)) + (nproma * (p_patch_nlev - 1))) + nproma)];
    double *z_w_con_c;
    z_w_con_c = new (std::align_val_t(64)) double [((nproma * (p_patch_nlevp1 - 1)) + nproma)];
    double *z_w_con_c_full;
    z_w_con_c_full = new (std::align_val_t(64)) double [((((nproma * p_patch_nlev) * (p_patch_nblks_c - 1)) + (nproma * (p_patch_nlev - 1))) + nproma)];
    double *z_w_concorr_mc;
    z_w_concorr_mc = new (std::align_val_t(64)) double [((nproma * (p_patch_nlev - 1)) + nproma)];
    double *z_w_v;
    z_w_v = new (std::align_val_t(64)) double [((((nproma * p_patch_nlevp1) * (p_patch_nblks_v - 1)) + (nproma * (p_patch_nlevp1 - 1))) + nproma)];
    double *zeta;
    zeta = new (std::align_val_t(64)) double [((((nproma * p_patch_nlev) * (p_patch_nblks_v - 1)) + (nproma * (p_patch_nlev - 1))) + nproma)];
    double _QQred_lift_0;
    double cfl_w_limit;
    double difcoef;
    int ie;
    bool l_vert_nested;
    double maxvcfl;
    int nrdmax_jg;
    int rl_end;
    int rl_start;
    double scalfac_exdiff;
    double vcfl;
    double w_con_e;
    int cells2verts_scalar_ri_i_endidx_in;
    int cells2verts_scalar_ri_i_startidx_in;
    int f__assoc_scalar_6;
    int f__assoc_scalar_7;
    int get_indices_v_i_endidx_in;
    int get_indices_v_i_startidx_in;
    int get_indices_v_irl_end;
    int f__assoc_scalar_9;
    int get_indices_e_i_endidx_in;
    int get_indices_e_i_startidx_in;
    int f__assoc_scalar_11;
    int get_indices_e_i_endidx_in_inl1;
    int get_indices_e_i_startidx_in_inl1;
    int f__assoc_scalar_13;
    int i_endidx_in_inl2;
    int i_startidx_in_inl2;
    int f__assoc_scalar_15;
    int get_indices_c_i_endidx_in;
    int get_indices_c_i_startidx_in;
    int f__assoc_scalar_17;
    int get_indices_c_i_endidx_in_inl1;
    int get_indices_c_i_startidx_in_inl1;
    int f__assoc_scalar_19;
    int i_endidx_in_inl3;
    int i_startidx_in_inl3;
    int64_t if_cond_252;
    int64_t if_cond_262;
    int64_t if_cond_324;
    int64_t if_cond_329;
    int64_t if_cond_406;
    int64_t if_cond_416;
    int scal_p_patch_id;
    bool scal_lextra_diffu;
    bool scal_lvert_nest;
    bool scal_p_diag_ddt_vn_adv_is_associated;
    bool scal_p_diag_ddt_vn_cor_is_associated;
    double scal_p_diag_max_vcfl_dyn;
    int64_t if_cond_1;
    int64_t if_cond_4;
    int jg;
    int nflatlev_jg;
    int nlev;
    int nlevp1;
    int64_t if_cond_14;
    int64_t if_cond_19;
    int rot_vertex_ri_elev;
    int rot_vertex_ri_i_startblk;
    int rot_vertex_ri_i_endblk;
    int64_t if_cond_106;
    int i_startblk;
    int i_endblk;
    int64_t if_cond_168;
    int i_startblk_2;
    int i_endblk_2;
    int64_t loopend_341;
    int64_t loopbegin_343;
    int64_t if_cond_439;
    int cells2verts_scalar_ri_elev;
    int cells2verts_scalar_ri_i_startblk;
    int cells2verts_scalar_ri_i_endblk;
    int64_t if_cond_27;
    int64_t if_cond_64;
    int64_t _loop_it_0;
    int64_t if_cond_32;
    int64_t if_cond_33;
    int cells2verts_scalar_ri_lib_i_startidx;
    int cells2verts_scalar_ri_lib_i_endidx;
    int64_t if_cond_38;
    int64_t _loop_it_1;
    int64_t _loop_it_2;
    int64_t p_patch_verts_cell_idx_at0;
    int64_t p_patch_verts_cell_blk_at1;
    int64_t p_patch_verts_cell_idx_at2;
    int64_t p_patch_verts_cell_blk_at3;
    int64_t p_patch_verts_cell_idx_at4;
    int64_t p_patch_verts_cell_blk_at5;
    int64_t p_patch_verts_cell_idx_at6;
    int64_t p_patch_verts_cell_blk_at7;
    int64_t p_patch_verts_cell_idx_at8;
    int64_t p_patch_verts_cell_blk_at9;
    int64_t p_patch_verts_cell_idx_at10;
    int64_t p_patch_verts_cell_blk_at11;
    int64_t _loop_it_3;
    int64_t if_cond_74;
    int64_t if_cond_75;
    int rot_vertex_ri_i_startidx;
    int rot_vertex_ri_i_endidx;
    int64_t if_cond_80;
    int64_t _loop_it_4;
    int64_t _loop_it_5;
    int64_t p_patch_verts_edge_idx_at12;
    int64_t p_patch_verts_edge_blk_at13;
    int64_t p_patch_verts_edge_idx_at14;
    int64_t p_patch_verts_edge_blk_at15;
    int64_t p_patch_verts_edge_idx_at16;
    int64_t p_patch_verts_edge_blk_at17;
    int64_t p_patch_verts_edge_idx_at18;
    int64_t p_patch_verts_edge_blk_at19;
    int64_t p_patch_verts_edge_idx_at20;
    int64_t p_patch_verts_edge_blk_at21;
    int64_t p_patch_verts_edge_idx_at22;
    int64_t p_patch_verts_edge_blk_at23;
    int64_t _loop_it_6;
    int f__assoc_scalar_8;
    int get_indices_e_irl_end;
    int i_startidx;
    int i_endidx;
    int64_t if_cond_133;
    int64_t if_cond_140;
    int64_t _loop_it_7;
    int64_t _loop_it_8;
    int64_t p_patch_edges_quad_idx_at24;
    int64_t p_patch_edges_quad_blk_at25;
    int64_t p_patch_edges_quad_idx_at26;
    int64_t p_patch_edges_quad_blk_at27;
    int64_t p_patch_edges_quad_idx_at28;
    int64_t p_patch_edges_quad_blk_at29;
    int64_t p_patch_edges_quad_idx_at30;
    int64_t p_patch_edges_quad_blk_at31;
    int64_t _loop_it_9;
    int64_t _loop_it_10;
    int64_t _loop_it_11;
    int64_t _loop_it_12;
    int64_t _loop_it_13;
    int64_t _loop_it_14;
    int64_t _loop_it_15;
    int64_t _loop_it_16;
    int64_t _loop_it_17;
    int f__assoc_scalar_10;
    int get_indices_e_irl_end_inl1;
    int64_t _loop_it_18;
    int64_t _loop_it_19;
    int64_t p_patch_edges_cell_idx_at32;
    int64_t p_patch_edges_cell_blk_at33;
    int64_t p_patch_edges_cell_idx_at34;
    int64_t p_patch_edges_cell_blk_at35;
    int64_t p_patch_edges_vertex_idx_at36;
    int64_t p_patch_edges_vertex_blk_at37;
    int64_t p_patch_edges_vertex_idx_at38;
    int64_t p_patch_edges_vertex_blk_at39;
    int64_t _loop_it_20;
    int f__assoc_scalar_12;
    int irl_end_inl2;
    int64_t _loop_it_21;
    int64_t _loop_it_22;
    int64_t _loop_it_23;
    int f__assoc_scalar_14;
    int get_indices_c_irl_end;
    int64_t if_cond_191;
    int64_t if_cond_192;
    int64_t if_cond_217;
    int64_t loopend_235;
    int64_t loopend_239;
    int64_t loopbegin_241;
    int64_t loopend_244;
    int64_t loopbegin_245;
    int64_t if_cond_279;
    int64_t if_cond_197;
    int64_t _loop_it_24;
    int64_t _loop_it_25;
    int64_t p_patch_cells_edge_idx_at40;
    int64_t p_patch_cells_edge_blk_at41;
    int64_t p_patch_cells_edge_idx_at42;
    int64_t p_patch_cells_edge_blk_at43;
    int64_t p_patch_cells_edge_idx_at44;
    int64_t p_patch_cells_edge_blk_at45;
    int64_t loopbegin_228;
    int64_t _loop_it_26;
    int64_t _loop_it_27;
    int64_t p_patch_cells_edge_idx_at46;
    int64_t p_patch_cells_edge_blk_at47;
    int64_t p_patch_cells_edge_idx_at48;
    int64_t p_patch_cells_edge_blk_at49;
    int64_t p_patch_cells_edge_idx_at50;
    int64_t p_patch_cells_edge_blk_at51;
    int64_t _loop_it_28;
    int64_t _loop_it_29;
    int64_t _loop_it_30;
    int64_t _loop_it_31;
    int64_t _loop_it_32;
    int64_t _loop_it_33;
    int64_t _loop_it_34;
    int64_t _loop_it_35;
    int64_t _loop_it_36;
    int clip_count;
    int64_t if_cond_258;
    int64_t _loop_it_37;
    int64_t _loop_it_38;
    int64_t if_cond_268;
    int64_t if_cond_269;
    int64_t _loop_it_39;
    int64_t _loop_it_40;
    int64_t if_cond_282;
    int f__assoc_scalar_16;
    int get_indices_c_irl_end_inl1;
    int64_t if_cond_289;
    int64_t if_cond_290;
    int64_t if_cond_317;
    int i_startidx_2;
    int i_endidx_2;
    int64_t if_cond_295;
    int64_t _loop_it_41;
    int64_t _loop_it_42;
    int64_t _loop_it_43;
    int64_t _loop_it_44;
    int64_t p_patch_cells_edge_idx_at52;
    int64_t p_patch_cells_edge_blk_at53;
    int64_t p_patch_cells_edge_idx_at54;
    int64_t p_patch_cells_edge_blk_at55;
    int64_t p_patch_cells_edge_idx_at56;
    int64_t p_patch_cells_edge_blk_at57;
    int64_t loopend_319;
    int64_t loopbegin_321;
    int64_t _loop_it_45;
    int64_t _loop_it_46;
    int64_t p_patch_cells_neighbor_idx_at58;
    int64_t p_patch_cells_neighbor_blk_at59;
    int64_t p_patch_cells_neighbor_idx_at60;
    int64_t p_patch_cells_neighbor_blk_at61;
    int64_t p_patch_cells_neighbor_idx_at62;
    int64_t p_patch_cells_neighbor_blk_at63;
    int64_t _loop_it_47;
    int64_t _loop_it_48;
    int64_t _loop_it_49;
    int f__assoc_scalar_18;
    int irl_end_inl3;
    int64_t if_cond_359;
    int64_t if_cond_399;
    int64_t if_cond_373;
    int64_t _loop_it_50;
    int64_t _loop_it_51;
    int64_t p_patch_edges_cell_idx_at64;
    int64_t p_patch_edges_cell_blk_at65;
    int64_t p_patch_edges_cell_idx_at66;
    int64_t p_patch_edges_cell_blk_at67;
    int64_t p_patch_edges_vertex_idx_at68;
    int64_t p_patch_edges_vertex_blk_at69;
    int64_t p_patch_edges_vertex_idx_at70;
    int64_t p_patch_edges_vertex_blk_at71;
    int64_t _loop_it_52;
    int64_t _loop_it_53;
    int64_t if_cond_389;
    int64_t _loop_it_54;
    int64_t _loop_it_55;
    int64_t p_patch_edges_cell_idx_at72;
    int64_t p_patch_edges_cell_blk_at73;
    int64_t p_patch_edges_cell_idx_at74;
    int64_t p_patch_edges_cell_blk_at75;
    int64_t p_patch_edges_vertex_idx_at76;
    int64_t p_patch_edges_vertex_blk_at77;
    int64_t p_patch_edges_vertex_idx_at78;
    int64_t p_patch_edges_vertex_blk_at79;
    int64_t _loop_it_56;
    int64_t _loop_it_57;
    int64_t p_patch_edges_cell_idx_at80;
    int64_t p_patch_edges_cell_blk_at81;
    int64_t p_patch_edges_cell_idx_at82;
    int64_t p_patch_edges_cell_blk_at83;
    int64_t loopend_402;
    int64_t loopbegin_403;
    int64_t _loop_it_58;
    int64_t _loop_it_59;
    int64_t p_patch_edges_cell_idx_at84;
    int64_t p_patch_edges_cell_blk_at85;
    int64_t p_patch_edges_cell_idx_at86;
    int64_t p_patch_edges_cell_blk_at87;
    int64_t p_patch_edges_quad_idx_at88;
    int64_t p_patch_edges_quad_blk_at89;
    int64_t p_patch_edges_quad_idx_at90;
    int64_t p_patch_edges_quad_blk_at91;
    int64_t p_patch_edges_quad_idx_at92;
    int64_t p_patch_edges_quad_blk_at93;
    int64_t p_patch_edges_quad_idx_at94;
    int64_t p_patch_edges_quad_blk_at95;
    int64_t p_patch_edges_vertex_idx_at96;
    int64_t p_patch_edges_vertex_blk_at97;
    int64_t p_patch_edges_vertex_idx_at98;
    int64_t p_patch_edges_vertex_blk_at99;
    int64_t _loop_it_60;

    {

        {
            int _cpy_in = p_patch_id[0];
            int _cpy_out;

            ///////////////////
            // Tasklet code (copy_p_patch_id_to_scal_p_patch_id)
            _cpy_out = _cpy_in;
            ///////////////////

            scal_p_patch_id = _cpy_out;
        }
        {
            bool _cpy_in = lextra_diffu[0];
            bool _cpy_out;

            ///////////////////
            // Tasklet code (copy_lextra_diffu_to_scal_lextra_diffu)
            _cpy_out = _cpy_in;
            ///////////////////

            scal_lextra_diffu = _cpy_out;
        }
        {
            bool _cpy_in = lvert_nest[0];
            bool _cpy_out;

            ///////////////////
            // Tasklet code (copy_lvert_nest_to_scal_lvert_nest)
            _cpy_out = _cpy_in;
            ///////////////////

            scal_lvert_nest = _cpy_out;
        }
        {
            bool _cpy_in = p_diag_ddt_vn_adv_is_associated[0];
            bool _cpy_out;

            ///////////////////
            // Tasklet code (copy_p_diag_ddt_vn_adv_is_associated_to_scal_p_diag_ddt_vn_adv_is_associated)
            _cpy_out = _cpy_in;
            ///////////////////

            scal_p_diag_ddt_vn_adv_is_associated = _cpy_out;
        }
        {
            bool _cpy_in = p_diag_ddt_vn_cor_is_associated[0];
            bool _cpy_out;

            ///////////////////
            // Tasklet code (copy_p_diag_ddt_vn_cor_is_associated_to_scal_p_diag_ddt_vn_cor_is_associated)
            _cpy_out = _cpy_in;
            ///////////////////

            scal_p_diag_ddt_vn_cor_is_associated = _cpy_out;
        }
        {
            double _cpy_in = p_diag_max_vcfl_dyn[0];
            double _cpy_out;

            ///////////////////
            // Tasklet code (copy_p_diag_max_vcfl_dyn_to_scal_p_diag_max_vcfl_dyn)
            _cpy_out = _cpy_in;
            ///////////////////

            scal_p_diag_max_vcfl_dyn = _cpy_out;
        }

    }

    if_cond_1 = (timers_level > 5);


    if (if_cond_1) {

    }


    if_cond_4 = (scal_lvert_nest && (p_patch_nshift > 0));


    if (if_cond_4) {
        {

            {
                bool _out;

                ///////////////////
                // Tasklet code (set_l_vert_nested)
                _out = -1;
                ///////////////////

                l_vert_nested = _out;
            }

        }
    } else {
        {

            {
                bool _out;

                ///////////////////
                // Tasklet code (set_l_vert_nested)
                _out = 0;
                ///////////////////

                l_vert_nested = _out;
            }

        }
    }


    jg = scal_p_patch_id;
    {

        {
            int _in_nrdmax_0 = nrdmax[(jg - 1)];
            int _out_nrdmax_jg;

            ///////////////////
            // Tasklet code (t_10)
            _out_nrdmax_jg = _in_nrdmax_0;
            ///////////////////

            nrdmax_jg = _out_nrdmax_jg;
        }

    }
    nflatlev_jg = nflatlev[(jg - 1)];

    nlev = p_patch_nlev;

    nlevp1 = p_patch_nlevp1;

    if_cond_14 = scal_lextra_diffu;


    if (if_cond_14) {
        {

            {
                double _in_dtime = dtime;
                double _out;

                ///////////////////
                // Tasklet code (set_cfl_w_limit)
                _out = (0.65 / _in_dtime);
                ///////////////////

                cfl_w_limit = _out;
            }
            {
                double _in_cfl_w_limit = cfl_w_limit;
                double _in_dtime = dtime;
                double _out;

                ///////////////////
                // Tasklet code (set_scalfac_exdiff)
                _out = (0.05 / (_in_dtime * (0.85 - (_in_cfl_w_limit * _in_dtime))));
                ///////////////////

                scalfac_exdiff = _out;
            }

        }
    } else {
        {

            {
                double _in_dtime = dtime;
                double _out;

                ///////////////////
                // Tasklet code (set_cfl_w_limit)
                _out = (0.85 / _in_dtime);
                ///////////////////

                cfl_w_limit = _out;
            }
            {
                double _out;

                ///////////////////
                // Tasklet code (set_scalfac_exdiff)
                _out = 0.0;
                ///////////////////

                scalfac_exdiff = _out;
            }

        }
    }


    if_cond_19 = (lvn_only != true);


    if (if_cond_19) {
        {
            int f__assoc_scalar_0;
            bool f__assoc_scalar_1;
            int slev;

            {
                int _out;

                ///////////////////
                // Tasklet code (set_f__assoc_scalar_0)
                _out = -5;
                ///////////////////

                f__assoc_scalar_0 = _out;
            }
            {
                bool _out;

                ///////////////////
                // Tasklet code (set_f__assoc_scalar_1)
                _out = -1;
                ///////////////////

                f__assoc_scalar_1 = _out;
            }
            {
                int _out;

                ///////////////////
                // Tasklet code (set_slev)
                _out = 1;
                ///////////////////

                slev = _out;
            }

        }
        cells2verts_scalar_ri_elev = ((p_prog_w_d1 + 1) - 1);
        {
            int cells2verts_scalar_ri_rl_end;
            int cells2verts_scalar_ri_rl_start;

            {
                int _out;

                ///////////////////
                // Tasklet code (set_cells2verts_scalar_ri_rl_start)
                _out = 2;
                ///////////////////

                cells2verts_scalar_ri_rl_start = _out;
            }
            {
                int _out;

                ///////////////////
                // Tasklet code (set_cells2verts_scalar_ri_rl_end)
                _out = -5;
                ///////////////////

                cells2verts_scalar_ri_rl_end = _out;
            }

        }
        cells2verts_scalar_ri_i_startblk = p_patch_verts_start_block[(2 - offset_p_patch_verts_start_block_d0)];

        cells2verts_scalar_ri_i_endblk = p_patch_verts_end_block[0];
        {

            {
                int _in_p_patch_verts_start_index_0 = p_patch_verts_start_index[(2 - offset_p_patch_verts_start_index_d0)];
                int _out_cells2verts_scalar_ri_i_startidx_in;

                ///////////////////
                // Tasklet code (t_25)
                _out_cells2verts_scalar_ri_i_startidx_in = _in_p_patch_verts_start_index_0;
                ///////////////////

                cells2verts_scalar_ri_i_startidx_in = _out_cells2verts_scalar_ri_i_startidx_in;
            }
            {
                int _in_p_patch_verts_end_index_0 = p_patch_verts_end_index[0];
                int _out_cells2verts_scalar_ri_i_endidx_in;

                ///////////////////
                // Tasklet code (t_26)
                _out_cells2verts_scalar_ri_i_endidx_in = _in_p_patch_verts_end_index_0;
                ///////////////////

                cells2verts_scalar_ri_i_endidx_in = _out_cells2verts_scalar_ri_i_endidx_in;
            }

        }
        if_cond_27 = (timers_level > 10);


        if (if_cond_27) {

        }

        {
            int f__assoc_scalar_2;
            bool f__assoc_scalar_3;
            bool lzacc;

            {
                int _out;

                ///////////////////
                // Tasklet code (set_f__assoc_scalar_2)
                _out = 1;
                ///////////////////

                f__assoc_scalar_2 = _out;
            }
            {
                bool _out;

                ///////////////////
                // Tasklet code (set_f__assoc_scalar_3)
                _out = -1;
                ///////////////////

                f__assoc_scalar_3 = _out;
            }
            {
                bool _out;

                ///////////////////
                // Tasklet code (set_lzacc)
                _out = 0;
                ///////////////////

                lzacc = _out;
            }

        }

        for (_loop_it_0 = cells2verts_scalar_ri_i_startblk; (_loop_it_0 < (cells2verts_scalar_ri_i_endblk + 1)); _loop_it_0 = (_loop_it_0 + 1)) {

            if_cond_32 = (_loop_it_0 == cells2verts_scalar_ri_i_startblk);

            if_cond_33 = (_loop_it_0 == cells2verts_scalar_ri_i_endblk);


            if (if_cond_32) {

                cells2verts_scalar_ri_lib_i_startidx = cells2verts_scalar_ri_i_startidx_in;

                cells2verts_scalar_ri_lib_i_endidx = nproma;

                if_cond_38 = (_loop_it_0 == cells2verts_scalar_ri_i_endblk);


                if (if_cond_38) {

                    cells2verts_scalar_ri_lib_i_endidx = cells2verts_scalar_ri_i_endidx_in;

                }

            } else if (if_cond_33) {

                cells2verts_scalar_ri_lib_i_startidx = 1;

                cells2verts_scalar_ri_lib_i_endidx = cells2verts_scalar_ri_i_endidx_in;

            } else {

                cells2verts_scalar_ri_lib_i_startidx = 1;

                cells2verts_scalar_ri_lib_i_endidx = nproma;

            }


            for (_loop_it_1 = 1; (_loop_it_1 < (cells2verts_scalar_ri_elev + 1)); _loop_it_1 = (_loop_it_1 + 1)) {

                for (_loop_it_2 = cells2verts_scalar_ri_lib_i_startidx; (_loop_it_2 < (cells2verts_scalar_ri_lib_i_endidx + 1)); _loop_it_2 = (_loop_it_2 + 1)) {

                    p_patch_verts_cell_idx_at0 = p_patch_verts_cell_idx[(((_loop_it_2 - offset_p_patch_verts_cell_idx_d0) + ((p_patch_verts_cell_idx_d0 * p_patch_verts_cell_idx_d1) * (1 - offset_p_patch_verts_cell_idx_d2))) + (p_patch_verts_cell_idx_d0 * (_loop_it_0 - offset_p_patch_verts_cell_idx_d1)))];

                    p_patch_verts_cell_blk_at1 = p_patch_verts_cell_blk[(((_loop_it_2 - offset_p_patch_verts_cell_blk_d0) + ((p_patch_verts_cell_blk_d0 * p_patch_verts_cell_blk_d1) * (1 - offset_p_patch_verts_cell_blk_d2))) + (p_patch_verts_cell_blk_d0 * (_loop_it_0 - offset_p_patch_verts_cell_blk_d1)))];

                    p_patch_verts_cell_idx_at2 = p_patch_verts_cell_idx[(((_loop_it_2 - offset_p_patch_verts_cell_idx_d0) + ((p_patch_verts_cell_idx_d0 * p_patch_verts_cell_idx_d1) * (2 - offset_p_patch_verts_cell_idx_d2))) + (p_patch_verts_cell_idx_d0 * (_loop_it_0 - offset_p_patch_verts_cell_idx_d1)))];

                    p_patch_verts_cell_blk_at3 = p_patch_verts_cell_blk[(((_loop_it_2 - offset_p_patch_verts_cell_blk_d0) + ((p_patch_verts_cell_blk_d0 * p_patch_verts_cell_blk_d1) * (2 - offset_p_patch_verts_cell_blk_d2))) + (p_patch_verts_cell_blk_d0 * (_loop_it_0 - offset_p_patch_verts_cell_blk_d1)))];

                    p_patch_verts_cell_idx_at4 = p_patch_verts_cell_idx[(((_loop_it_2 - offset_p_patch_verts_cell_idx_d0) + ((p_patch_verts_cell_idx_d0 * p_patch_verts_cell_idx_d1) * (3 - offset_p_patch_verts_cell_idx_d2))) + (p_patch_verts_cell_idx_d0 * (_loop_it_0 - offset_p_patch_verts_cell_idx_d1)))];

                    p_patch_verts_cell_blk_at5 = p_patch_verts_cell_blk[(((_loop_it_2 - offset_p_patch_verts_cell_blk_d0) + ((p_patch_verts_cell_blk_d0 * p_patch_verts_cell_blk_d1) * (3 - offset_p_patch_verts_cell_blk_d2))) + (p_patch_verts_cell_blk_d0 * (_loop_it_0 - offset_p_patch_verts_cell_blk_d1)))];

                    p_patch_verts_cell_idx_at6 = p_patch_verts_cell_idx[(((_loop_it_2 - offset_p_patch_verts_cell_idx_d0) + ((p_patch_verts_cell_idx_d0 * p_patch_verts_cell_idx_d1) * (4 - offset_p_patch_verts_cell_idx_d2))) + (p_patch_verts_cell_idx_d0 * (_loop_it_0 - offset_p_patch_verts_cell_idx_d1)))];

                    p_patch_verts_cell_blk_at7 = p_patch_verts_cell_blk[(((_loop_it_2 - offset_p_patch_verts_cell_blk_d0) + ((p_patch_verts_cell_blk_d0 * p_patch_verts_cell_blk_d1) * (4 - offset_p_patch_verts_cell_blk_d2))) + (p_patch_verts_cell_blk_d0 * (_loop_it_0 - offset_p_patch_verts_cell_blk_d1)))];

                    p_patch_verts_cell_idx_at8 = p_patch_verts_cell_idx[(((_loop_it_2 - offset_p_patch_verts_cell_idx_d0) + ((p_patch_verts_cell_idx_d0 * p_patch_verts_cell_idx_d1) * (5 - offset_p_patch_verts_cell_idx_d2))) + (p_patch_verts_cell_idx_d0 * (_loop_it_0 - offset_p_patch_verts_cell_idx_d1)))];

                    p_patch_verts_cell_blk_at9 = p_patch_verts_cell_blk[(((_loop_it_2 - offset_p_patch_verts_cell_blk_d0) + ((p_patch_verts_cell_blk_d0 * p_patch_verts_cell_blk_d1) * (5 - offset_p_patch_verts_cell_blk_d2))) + (p_patch_verts_cell_blk_d0 * (_loop_it_0 - offset_p_patch_verts_cell_blk_d1)))];

                    p_patch_verts_cell_idx_at10 = p_patch_verts_cell_idx[(((_loop_it_2 - offset_p_patch_verts_cell_idx_d0) + ((p_patch_verts_cell_idx_d0 * p_patch_verts_cell_idx_d1) * (6 - offset_p_patch_verts_cell_idx_d2))) + (p_patch_verts_cell_idx_d0 * (_loop_it_0 - offset_p_patch_verts_cell_idx_d1)))];

                    p_patch_verts_cell_blk_at11 = p_patch_verts_cell_blk[(((_loop_it_2 - offset_p_patch_verts_cell_blk_d0) + ((p_patch_verts_cell_blk_d0 * p_patch_verts_cell_blk_d1) * (6 - offset_p_patch_verts_cell_blk_d2))) + (p_patch_verts_cell_blk_d0 * (_loop_it_0 - offset_p_patch_verts_cell_blk_d1)))];

                    {

                        {
                            double _in_p_int_cells_aw_verts_0 = p_int_cells_aw_verts[(((_loop_it_2 - offset_p_int_cells_aw_verts_d0) + ((p_int_cells_aw_verts_d0 * p_int_cells_aw_verts_d1) * (_loop_it_0 - offset_p_int_cells_aw_verts_d2))) + (p_int_cells_aw_verts_d0 * (1 - offset_p_int_cells_aw_verts_d1)))];
                            double _in_p_int_cells_aw_verts_1 = p_int_cells_aw_verts[(((_loop_it_2 - offset_p_int_cells_aw_verts_d0) + ((p_int_cells_aw_verts_d0 * p_int_cells_aw_verts_d1) * (_loop_it_0 - offset_p_int_cells_aw_verts_d2))) + (p_int_cells_aw_verts_d0 * (2 - offset_p_int_cells_aw_verts_d1)))];
                            double _in_p_int_cells_aw_verts_2 = p_int_cells_aw_verts[(((_loop_it_2 - offset_p_int_cells_aw_verts_d0) + ((p_int_cells_aw_verts_d0 * p_int_cells_aw_verts_d1) * (_loop_it_0 - offset_p_int_cells_aw_verts_d2))) + (p_int_cells_aw_verts_d0 * (3 - offset_p_int_cells_aw_verts_d1)))];
                            double _in_p_int_cells_aw_verts_3 = p_int_cells_aw_verts[(((_loop_it_2 - offset_p_int_cells_aw_verts_d0) + ((p_int_cells_aw_verts_d0 * p_int_cells_aw_verts_d1) * (_loop_it_0 - offset_p_int_cells_aw_verts_d2))) + (p_int_cells_aw_verts_d0 * (4 - offset_p_int_cells_aw_verts_d1)))];
                            double _in_p_int_cells_aw_verts_4 = p_int_cells_aw_verts[(((_loop_it_2 - offset_p_int_cells_aw_verts_d0) + ((p_int_cells_aw_verts_d0 * p_int_cells_aw_verts_d1) * (_loop_it_0 - offset_p_int_cells_aw_verts_d2))) + (p_int_cells_aw_verts_d0 * (5 - offset_p_int_cells_aw_verts_d1)))];
                            double _in_p_int_cells_aw_verts_5 = p_int_cells_aw_verts[(((_loop_it_2 - offset_p_int_cells_aw_verts_d0) + ((p_int_cells_aw_verts_d0 * p_int_cells_aw_verts_d1) * (_loop_it_0 - offset_p_int_cells_aw_verts_d2))) + (p_int_cells_aw_verts_d0 * (6 - offset_p_int_cells_aw_verts_d1)))];
                            double _in_p_prog_w_0 = p_prog_w[((((- offset_p_prog_w_d0) + p_patch_verts_cell_idx_at0) + ((p_prog_w_d0 * p_prog_w_d1) * ((- offset_p_prog_w_d2) + p_patch_verts_cell_blk_at1))) + (p_prog_w_d0 * (_loop_it_1 - offset_p_prog_w_d1)))];
                            double _in_p_prog_w_1 = p_prog_w[((((- offset_p_prog_w_d0) + p_patch_verts_cell_idx_at2) + ((p_prog_w_d0 * p_prog_w_d1) * ((- offset_p_prog_w_d2) + p_patch_verts_cell_blk_at3))) + (p_prog_w_d0 * (_loop_it_1 - offset_p_prog_w_d1)))];
                            double _in_p_prog_w_2 = p_prog_w[((((- offset_p_prog_w_d0) + p_patch_verts_cell_idx_at4) + ((p_prog_w_d0 * p_prog_w_d1) * ((- offset_p_prog_w_d2) + p_patch_verts_cell_blk_at5))) + (p_prog_w_d0 * (_loop_it_1 - offset_p_prog_w_d1)))];
                            double _in_p_prog_w_3 = p_prog_w[((((- offset_p_prog_w_d0) + p_patch_verts_cell_idx_at6) + ((p_prog_w_d0 * p_prog_w_d1) * ((- offset_p_prog_w_d2) + p_patch_verts_cell_blk_at7))) + (p_prog_w_d0 * (_loop_it_1 - offset_p_prog_w_d1)))];
                            double _in_p_prog_w_4 = p_prog_w[((((- offset_p_prog_w_d0) + p_patch_verts_cell_idx_at8) + ((p_prog_w_d0 * p_prog_w_d1) * ((- offset_p_prog_w_d2) + p_patch_verts_cell_blk_at9))) + (p_prog_w_d0 * (_loop_it_1 - offset_p_prog_w_d1)))];
                            double _in_p_prog_w_5 = p_prog_w[((((- offset_p_prog_w_d0) + p_patch_verts_cell_idx_at10) + ((p_prog_w_d0 * p_prog_w_d1) * ((- offset_p_prog_w_d2) + p_patch_verts_cell_blk_at11))) + (p_prog_w_d0 * (_loop_it_1 - offset_p_prog_w_d1)))];
                            double _out_z_w_v;

                            ///////////////////
                            // Tasklet code (t_0)
                            _out_z_w_v = ((((((_in_p_int_cells_aw_verts_0 * _in_p_prog_w_0) + (_in_p_int_cells_aw_verts_1 * _in_p_prog_w_1)) + (_in_p_int_cells_aw_verts_2 * _in_p_prog_w_2)) + (_in_p_int_cells_aw_verts_3 * _in_p_prog_w_3)) + (_in_p_int_cells_aw_verts_4 * _in_p_prog_w_4)) + (_in_p_int_cells_aw_verts_5 * _in_p_prog_w_5));
                            ///////////////////

                            z_w_v[(((_loop_it_2 + ((nproma * p_patch_nlevp1) * (_loop_it_0 - 1))) + (nproma * (_loop_it_1 - 1))) - 1)] = _out_z_w_v;
                        }

                    }

                }



            }



        }



        if_cond_64 = (timers_level > 10);


        if (if_cond_64) {

        }

    }

    {
        int f__assoc_scalar_4;
        bool f__assoc_scalar_5;

        {
            int _out;

            ///////////////////
            // Tasklet code (set_f__assoc_scalar_4)
            _out = -5;
            ///////////////////

            f__assoc_scalar_4 = _out;
        }
        {
            bool _out;

            ///////////////////
            // Tasklet code (set_f__assoc_scalar_5)
            _out = -1;
            ///////////////////

            f__assoc_scalar_5 = _out;
        }

    }
    rot_vertex_ri_elev = ((p_prog_vn_d1 + 1) - 1);
    {
        int rot_vertex_ri_rl_end;
        int rot_vertex_ri_rl_start;

        {
            int _out;

            ///////////////////
            // Tasklet code (set_rot_vertex_ri_rl_start)
            _out = 2;
            ///////////////////

            rot_vertex_ri_rl_start = _out;
        }
        {
            int _out;

            ///////////////////
            // Tasklet code (set_rot_vertex_ri_rl_end)
            _out = -5;
            ///////////////////

            rot_vertex_ri_rl_end = _out;
        }

    }
    rot_vertex_ri_i_startblk = p_patch_verts_start_block[(2 - offset_p_patch_verts_start_block_d0)];

    rot_vertex_ri_i_endblk = p_patch_verts_end_block[0];


    for (_loop_it_3 = rot_vertex_ri_i_startblk; (_loop_it_3 < (rot_vertex_ri_i_endblk + 1)); _loop_it_3 = (_loop_it_3 + 1)) {
        {

            {
                int _out;

                ///////////////////
                // Tasklet code (set_f__assoc_scalar_6)
                _out = 2;
                ///////////////////

                f__assoc_scalar_6 = _out;
            }
            {
                int _out;

                ///////////////////
                // Tasklet code (set_f__assoc_scalar_7)
                _out = -5;
                ///////////////////

                f__assoc_scalar_7 = _out;
            }
            {
                int _in_p_patch_verts_start_index_0 = p_patch_verts_start_index[(2 - offset_p_patch_verts_start_index_d0)];
                int _out_get_indices_v_i_startidx_in;

                ///////////////////
                // Tasklet code (t_72)
                _out_get_indices_v_i_startidx_in = _in_p_patch_verts_start_index_0;
                ///////////////////

                get_indices_v_i_startidx_in = _out_get_indices_v_i_startidx_in;
            }
            {
                int _out;

                ///////////////////
                // Tasklet code (set_get_indices_v_irl_end)
                _out = -5;
                ///////////////////

                get_indices_v_irl_end = _out;
            }
            {
                int _in_p_patch_verts_end_index_0 = p_patch_verts_end_index[0];
                int _out_get_indices_v_i_endidx_in;

                ///////////////////
                // Tasklet code (t_73)
                _out_get_indices_v_i_endidx_in = _in_p_patch_verts_end_index_0;
                ///////////////////

                get_indices_v_i_endidx_in = _out_get_indices_v_i_endidx_in;
            }

        }
        if_cond_74 = (_loop_it_3 == rot_vertex_ri_i_startblk);

        if_cond_75 = (_loop_it_3 == rot_vertex_ri_i_endblk);


        if (if_cond_74) {

            rot_vertex_ri_i_startidx = get_indices_v_i_startidx_in;

            rot_vertex_ri_i_endidx = nproma;

            if_cond_80 = (_loop_it_3 == rot_vertex_ri_i_endblk);


            if (if_cond_80) {

                rot_vertex_ri_i_endidx = get_indices_v_i_endidx_in;

            }

        } else if (if_cond_75) {

            rot_vertex_ri_i_startidx = 1;

            rot_vertex_ri_i_endidx = get_indices_v_i_endidx_in;

        } else {

            rot_vertex_ri_i_startidx = 1;

            rot_vertex_ri_i_endidx = nproma;

        }


        for (_loop_it_4 = 1; (_loop_it_4 < (rot_vertex_ri_elev + 1)); _loop_it_4 = (_loop_it_4 + 1)) {

            for (_loop_it_5 = rot_vertex_ri_i_startidx; (_loop_it_5 < (rot_vertex_ri_i_endidx + 1)); _loop_it_5 = (_loop_it_5 + 1)) {

                p_patch_verts_edge_idx_at12 = p_patch_verts_edge_idx[(((_loop_it_5 - offset_p_patch_verts_edge_idx_d0) + ((p_patch_verts_edge_idx_d0 * p_patch_verts_edge_idx_d1) * (1 - offset_p_patch_verts_edge_idx_d2))) + (p_patch_verts_edge_idx_d0 * (_loop_it_3 - offset_p_patch_verts_edge_idx_d1)))];

                p_patch_verts_edge_blk_at13 = p_patch_verts_edge_blk[(((_loop_it_5 - offset_p_patch_verts_edge_blk_d0) + ((p_patch_verts_edge_blk_d0 * p_patch_verts_edge_blk_d1) * (1 - offset_p_patch_verts_edge_blk_d2))) + (p_patch_verts_edge_blk_d0 * (_loop_it_3 - offset_p_patch_verts_edge_blk_d1)))];

                p_patch_verts_edge_idx_at14 = p_patch_verts_edge_idx[(((_loop_it_5 - offset_p_patch_verts_edge_idx_d0) + ((p_patch_verts_edge_idx_d0 * p_patch_verts_edge_idx_d1) * (2 - offset_p_patch_verts_edge_idx_d2))) + (p_patch_verts_edge_idx_d0 * (_loop_it_3 - offset_p_patch_verts_edge_idx_d1)))];

                p_patch_verts_edge_blk_at15 = p_patch_verts_edge_blk[(((_loop_it_5 - offset_p_patch_verts_edge_blk_d0) + ((p_patch_verts_edge_blk_d0 * p_patch_verts_edge_blk_d1) * (2 - offset_p_patch_verts_edge_blk_d2))) + (p_patch_verts_edge_blk_d0 * (_loop_it_3 - offset_p_patch_verts_edge_blk_d1)))];

                p_patch_verts_edge_idx_at16 = p_patch_verts_edge_idx[(((_loop_it_5 - offset_p_patch_verts_edge_idx_d0) + ((p_patch_verts_edge_idx_d0 * p_patch_verts_edge_idx_d1) * (3 - offset_p_patch_verts_edge_idx_d2))) + (p_patch_verts_edge_idx_d0 * (_loop_it_3 - offset_p_patch_verts_edge_idx_d1)))];

                p_patch_verts_edge_blk_at17 = p_patch_verts_edge_blk[(((_loop_it_5 - offset_p_patch_verts_edge_blk_d0) + ((p_patch_verts_edge_blk_d0 * p_patch_verts_edge_blk_d1) * (3 - offset_p_patch_verts_edge_blk_d2))) + (p_patch_verts_edge_blk_d0 * (_loop_it_3 - offset_p_patch_verts_edge_blk_d1)))];

                p_patch_verts_edge_idx_at18 = p_patch_verts_edge_idx[(((_loop_it_5 - offset_p_patch_verts_edge_idx_d0) + ((p_patch_verts_edge_idx_d0 * p_patch_verts_edge_idx_d1) * (4 - offset_p_patch_verts_edge_idx_d2))) + (p_patch_verts_edge_idx_d0 * (_loop_it_3 - offset_p_patch_verts_edge_idx_d1)))];

                p_patch_verts_edge_blk_at19 = p_patch_verts_edge_blk[(((_loop_it_5 - offset_p_patch_verts_edge_blk_d0) + ((p_patch_verts_edge_blk_d0 * p_patch_verts_edge_blk_d1) * (4 - offset_p_patch_verts_edge_blk_d2))) + (p_patch_verts_edge_blk_d0 * (_loop_it_3 - offset_p_patch_verts_edge_blk_d1)))];

                p_patch_verts_edge_idx_at20 = p_patch_verts_edge_idx[(((_loop_it_5 - offset_p_patch_verts_edge_idx_d0) + ((p_patch_verts_edge_idx_d0 * p_patch_verts_edge_idx_d1) * (5 - offset_p_patch_verts_edge_idx_d2))) + (p_patch_verts_edge_idx_d0 * (_loop_it_3 - offset_p_patch_verts_edge_idx_d1)))];

                p_patch_verts_edge_blk_at21 = p_patch_verts_edge_blk[(((_loop_it_5 - offset_p_patch_verts_edge_blk_d0) + ((p_patch_verts_edge_blk_d0 * p_patch_verts_edge_blk_d1) * (5 - offset_p_patch_verts_edge_blk_d2))) + (p_patch_verts_edge_blk_d0 * (_loop_it_3 - offset_p_patch_verts_edge_blk_d1)))];

                p_patch_verts_edge_idx_at22 = p_patch_verts_edge_idx[(((_loop_it_5 - offset_p_patch_verts_edge_idx_d0) + ((p_patch_verts_edge_idx_d0 * p_patch_verts_edge_idx_d1) * (6 - offset_p_patch_verts_edge_idx_d2))) + (p_patch_verts_edge_idx_d0 * (_loop_it_3 - offset_p_patch_verts_edge_idx_d1)))];

                p_patch_verts_edge_blk_at23 = p_patch_verts_edge_blk[(((_loop_it_5 - offset_p_patch_verts_edge_blk_d0) + ((p_patch_verts_edge_blk_d0 * p_patch_verts_edge_blk_d1) * (6 - offset_p_patch_verts_edge_blk_d2))) + (p_patch_verts_edge_blk_d0 * (_loop_it_3 - offset_p_patch_verts_edge_blk_d1)))];

                {

                    {
                        double _in_p_int_geofac_rot_0 = p_int_geofac_rot[(((_loop_it_5 - offset_p_int_geofac_rot_d0) + ((p_int_geofac_rot_d0 * p_int_geofac_rot_d1) * (_loop_it_3 - offset_p_int_geofac_rot_d2))) + (p_int_geofac_rot_d0 * (1 - offset_p_int_geofac_rot_d1)))];
                        double _in_p_int_geofac_rot_1 = p_int_geofac_rot[(((_loop_it_5 - offset_p_int_geofac_rot_d0) + ((p_int_geofac_rot_d0 * p_int_geofac_rot_d1) * (_loop_it_3 - offset_p_int_geofac_rot_d2))) + (p_int_geofac_rot_d0 * (2 - offset_p_int_geofac_rot_d1)))];
                        double _in_p_int_geofac_rot_2 = p_int_geofac_rot[(((_loop_it_5 - offset_p_int_geofac_rot_d0) + ((p_int_geofac_rot_d0 * p_int_geofac_rot_d1) * (_loop_it_3 - offset_p_int_geofac_rot_d2))) + (p_int_geofac_rot_d0 * (3 - offset_p_int_geofac_rot_d1)))];
                        double _in_p_int_geofac_rot_3 = p_int_geofac_rot[(((_loop_it_5 - offset_p_int_geofac_rot_d0) + ((p_int_geofac_rot_d0 * p_int_geofac_rot_d1) * (_loop_it_3 - offset_p_int_geofac_rot_d2))) + (p_int_geofac_rot_d0 * (4 - offset_p_int_geofac_rot_d1)))];
                        double _in_p_int_geofac_rot_4 = p_int_geofac_rot[(((_loop_it_5 - offset_p_int_geofac_rot_d0) + ((p_int_geofac_rot_d0 * p_int_geofac_rot_d1) * (_loop_it_3 - offset_p_int_geofac_rot_d2))) + (p_int_geofac_rot_d0 * (5 - offset_p_int_geofac_rot_d1)))];
                        double _in_p_int_geofac_rot_5 = p_int_geofac_rot[(((_loop_it_5 - offset_p_int_geofac_rot_d0) + ((p_int_geofac_rot_d0 * p_int_geofac_rot_d1) * (_loop_it_3 - offset_p_int_geofac_rot_d2))) + (p_int_geofac_rot_d0 * (6 - offset_p_int_geofac_rot_d1)))];
                        double _in_p_prog_vn_0 = p_prog_vn[((((- offset_p_prog_vn_d0) + p_patch_verts_edge_idx_at12) + ((p_prog_vn_d0 * p_prog_vn_d1) * ((- offset_p_prog_vn_d2) + p_patch_verts_edge_blk_at13))) + (p_prog_vn_d0 * (_loop_it_4 - offset_p_prog_vn_d1)))];
                        double _in_p_prog_vn_1 = p_prog_vn[((((- offset_p_prog_vn_d0) + p_patch_verts_edge_idx_at14) + ((p_prog_vn_d0 * p_prog_vn_d1) * ((- offset_p_prog_vn_d2) + p_patch_verts_edge_blk_at15))) + (p_prog_vn_d0 * (_loop_it_4 - offset_p_prog_vn_d1)))];
                        double _in_p_prog_vn_2 = p_prog_vn[((((- offset_p_prog_vn_d0) + p_patch_verts_edge_idx_at16) + ((p_prog_vn_d0 * p_prog_vn_d1) * ((- offset_p_prog_vn_d2) + p_patch_verts_edge_blk_at17))) + (p_prog_vn_d0 * (_loop_it_4 - offset_p_prog_vn_d1)))];
                        double _in_p_prog_vn_3 = p_prog_vn[((((- offset_p_prog_vn_d0) + p_patch_verts_edge_idx_at18) + ((p_prog_vn_d0 * p_prog_vn_d1) * ((- offset_p_prog_vn_d2) + p_patch_verts_edge_blk_at19))) + (p_prog_vn_d0 * (_loop_it_4 - offset_p_prog_vn_d1)))];
                        double _in_p_prog_vn_4 = p_prog_vn[((((- offset_p_prog_vn_d0) + p_patch_verts_edge_idx_at20) + ((p_prog_vn_d0 * p_prog_vn_d1) * ((- offset_p_prog_vn_d2) + p_patch_verts_edge_blk_at21))) + (p_prog_vn_d0 * (_loop_it_4 - offset_p_prog_vn_d1)))];
                        double _in_p_prog_vn_5 = p_prog_vn[((((- offset_p_prog_vn_d0) + p_patch_verts_edge_idx_at22) + ((p_prog_vn_d0 * p_prog_vn_d1) * ((- offset_p_prog_vn_d2) + p_patch_verts_edge_blk_at23))) + (p_prog_vn_d0 * (_loop_it_4 - offset_p_prog_vn_d1)))];
                        double _out_zeta;

                        ///////////////////
                        // Tasklet code (t_0)
                        _out_zeta = ((((((_in_p_prog_vn_0 * _in_p_int_geofac_rot_0) + (_in_p_prog_vn_1 * _in_p_int_geofac_rot_1)) + (_in_p_prog_vn_2 * _in_p_int_geofac_rot_2)) + (_in_p_prog_vn_3 * _in_p_int_geofac_rot_3)) + (_in_p_prog_vn_4 * _in_p_int_geofac_rot_4)) + (_in_p_prog_vn_5 * _in_p_int_geofac_rot_5));
                        ///////////////////

                        zeta[(((_loop_it_5 + ((nproma * p_patch_nlev) * (_loop_it_3 - 1))) + (nproma * (_loop_it_4 - 1))) - 1)] = _out_zeta;
                    }

                }

            }



        }



    }



    if_cond_106 = (istep == 1);


    if (if_cond_106) {
        {

            {
                int _out;

                ///////////////////
                // Tasklet code (set_rl_start)
                _out = 5;
                ///////////////////

                rl_start = _out;
            }
            {
                int _out;

                ///////////////////
                // Tasklet code (set_rl_end)
                _out = -10;
                ///////////////////

                rl_end = _out;
            }

        }
        i_startblk = p_patch_edges_start_block[(5 - offset_p_patch_edges_start_block_d0)];

        i_endblk = p_patch_edges_end_block[0];


        for (_loop_it_6 = i_startblk; (_loop_it_6 < (i_endblk + 1)); _loop_it_6 = (_loop_it_6 + 1)) {

            f__assoc_scalar_8 = 5;
            {

                {
                    int _out;

                    ///////////////////
                    // Tasklet code (set_f__assoc_scalar_9)
                    _out = -10;
                    ///////////////////

                    f__assoc_scalar_9 = _out;
                }
                {
                    int _in_p_patch_edges_start_index_0 = p_patch_edges_start_index[(f__assoc_scalar_8 - offset_p_patch_edges_start_index_d0)];
                    int _out_get_indices_e_i_startidx_in;

                    ///////////////////
                    // Tasklet code (t_114)
                    _out_get_indices_e_i_startidx_in = _in_p_patch_edges_start_index_0;
                    ///////////////////

                    get_indices_e_i_startidx_in = _out_get_indices_e_i_startidx_in;
                }

            }
            get_indices_e_irl_end = f__assoc_scalar_9;
            {

                {
                    int _in_p_patch_edges_end_index_0 = p_patch_edges_end_index[(get_indices_e_irl_end + 10)];
                    int _out_get_indices_e_i_endidx_in;

                    ///////////////////
                    // Tasklet code (t_116)
                    _out_get_indices_e_i_endidx_in = _in_p_patch_edges_end_index_0;
                    ///////////////////

                    get_indices_e_i_endidx_in = _out_get_indices_e_i_endidx_in;
                }

            }
            i_startidx = ((_loop_it_6 != i_startblk) ? 1 : ((get_indices_e_i_startidx_in < 1) ? 1 : get_indices_e_i_startidx_in));

            i_endidx = ((_loop_it_6 != i_endblk) ? nproma : get_indices_e_i_endidx_in);


            for (_loop_it_7 = 1; (_loop_it_7 < (nlev + 1)); _loop_it_7 = (_loop_it_7 + 1)) {

                for (_loop_it_8 = i_startidx; (_loop_it_8 < (i_endidx + 1)); _loop_it_8 = (_loop_it_8 + 1)) {

                    p_patch_edges_quad_idx_at24 = p_patch_edges_quad_idx[(((_loop_it_8 - offset_p_patch_edges_quad_idx_d0) + ((p_patch_edges_quad_idx_d0 * p_patch_edges_quad_idx_d1) * (1 - offset_p_patch_edges_quad_idx_d2))) + (p_patch_edges_quad_idx_d0 * (_loop_it_6 - offset_p_patch_edges_quad_idx_d1)))];

                    p_patch_edges_quad_blk_at25 = p_patch_edges_quad_blk[(((_loop_it_8 - offset_p_patch_edges_quad_blk_d0) + ((p_patch_edges_quad_blk_d0 * p_patch_edges_quad_blk_d1) * (1 - offset_p_patch_edges_quad_blk_d2))) + (p_patch_edges_quad_blk_d0 * (_loop_it_6 - offset_p_patch_edges_quad_blk_d1)))];

                    p_patch_edges_quad_idx_at26 = p_patch_edges_quad_idx[(((_loop_it_8 - offset_p_patch_edges_quad_idx_d0) + ((p_patch_edges_quad_idx_d0 * p_patch_edges_quad_idx_d1) * (2 - offset_p_patch_edges_quad_idx_d2))) + (p_patch_edges_quad_idx_d0 * (_loop_it_6 - offset_p_patch_edges_quad_idx_d1)))];

                    p_patch_edges_quad_blk_at27 = p_patch_edges_quad_blk[(((_loop_it_8 - offset_p_patch_edges_quad_blk_d0) + ((p_patch_edges_quad_blk_d0 * p_patch_edges_quad_blk_d1) * (2 - offset_p_patch_edges_quad_blk_d2))) + (p_patch_edges_quad_blk_d0 * (_loop_it_6 - offset_p_patch_edges_quad_blk_d1)))];

                    p_patch_edges_quad_idx_at28 = p_patch_edges_quad_idx[(((_loop_it_8 - offset_p_patch_edges_quad_idx_d0) + ((p_patch_edges_quad_idx_d0 * p_patch_edges_quad_idx_d1) * (3 - offset_p_patch_edges_quad_idx_d2))) + (p_patch_edges_quad_idx_d0 * (_loop_it_6 - offset_p_patch_edges_quad_idx_d1)))];

                    p_patch_edges_quad_blk_at29 = p_patch_edges_quad_blk[(((_loop_it_8 - offset_p_patch_edges_quad_blk_d0) + ((p_patch_edges_quad_blk_d0 * p_patch_edges_quad_blk_d1) * (3 - offset_p_patch_edges_quad_blk_d2))) + (p_patch_edges_quad_blk_d0 * (_loop_it_6 - offset_p_patch_edges_quad_blk_d1)))];

                    p_patch_edges_quad_idx_at30 = p_patch_edges_quad_idx[(((_loop_it_8 - offset_p_patch_edges_quad_idx_d0) + ((p_patch_edges_quad_idx_d0 * p_patch_edges_quad_idx_d1) * (4 - offset_p_patch_edges_quad_idx_d2))) + (p_patch_edges_quad_idx_d0 * (_loop_it_6 - offset_p_patch_edges_quad_idx_d1)))];

                    p_patch_edges_quad_blk_at31 = p_patch_edges_quad_blk[(((_loop_it_8 - offset_p_patch_edges_quad_blk_d0) + ((p_patch_edges_quad_blk_d0 * p_patch_edges_quad_blk_d1) * (4 - offset_p_patch_edges_quad_blk_d2))) + (p_patch_edges_quad_blk_d0 * (_loop_it_6 - offset_p_patch_edges_quad_blk_d1)))];

                    {

                        {
                            double _in_p_int_rbf_vec_coeff_e_0 = p_int_rbf_vec_coeff_e[((((- offset_p_int_rbf_vec_coeff_e_d0) + ((p_int_rbf_vec_coeff_e_d0 * p_int_rbf_vec_coeff_e_d1) * (_loop_it_6 - offset_p_int_rbf_vec_coeff_e_d2))) + (p_int_rbf_vec_coeff_e_d0 * (_loop_it_8 - offset_p_int_rbf_vec_coeff_e_d1))) + 1)];
                            double _in_p_int_rbf_vec_coeff_e_1 = p_int_rbf_vec_coeff_e[((((- offset_p_int_rbf_vec_coeff_e_d0) + ((p_int_rbf_vec_coeff_e_d0 * p_int_rbf_vec_coeff_e_d1) * (_loop_it_6 - offset_p_int_rbf_vec_coeff_e_d2))) + (p_int_rbf_vec_coeff_e_d0 * (_loop_it_8 - offset_p_int_rbf_vec_coeff_e_d1))) + 2)];
                            double _in_p_int_rbf_vec_coeff_e_2 = p_int_rbf_vec_coeff_e[((((- offset_p_int_rbf_vec_coeff_e_d0) + ((p_int_rbf_vec_coeff_e_d0 * p_int_rbf_vec_coeff_e_d1) * (_loop_it_6 - offset_p_int_rbf_vec_coeff_e_d2))) + (p_int_rbf_vec_coeff_e_d0 * (_loop_it_8 - offset_p_int_rbf_vec_coeff_e_d1))) + 3)];
                            double _in_p_int_rbf_vec_coeff_e_3 = p_int_rbf_vec_coeff_e[((((- offset_p_int_rbf_vec_coeff_e_d0) + ((p_int_rbf_vec_coeff_e_d0 * p_int_rbf_vec_coeff_e_d1) * (_loop_it_6 - offset_p_int_rbf_vec_coeff_e_d2))) + (p_int_rbf_vec_coeff_e_d0 * (_loop_it_8 - offset_p_int_rbf_vec_coeff_e_d1))) + 4)];
                            double _in_p_prog_vn_0 = p_prog_vn[((((- offset_p_prog_vn_d0) + p_patch_edges_quad_idx_at24) + ((p_prog_vn_d0 * p_prog_vn_d1) * ((- offset_p_prog_vn_d2) + p_patch_edges_quad_blk_at25))) + (p_prog_vn_d0 * (_loop_it_7 - offset_p_prog_vn_d1)))];
                            double _in_p_prog_vn_1 = p_prog_vn[((((- offset_p_prog_vn_d0) + p_patch_edges_quad_idx_at26) + ((p_prog_vn_d0 * p_prog_vn_d1) * ((- offset_p_prog_vn_d2) + p_patch_edges_quad_blk_at27))) + (p_prog_vn_d0 * (_loop_it_7 - offset_p_prog_vn_d1)))];
                            double _in_p_prog_vn_2 = p_prog_vn[((((- offset_p_prog_vn_d0) + p_patch_edges_quad_idx_at28) + ((p_prog_vn_d0 * p_prog_vn_d1) * ((- offset_p_prog_vn_d2) + p_patch_edges_quad_blk_at29))) + (p_prog_vn_d0 * (_loop_it_7 - offset_p_prog_vn_d1)))];
                            double _in_p_prog_vn_3 = p_prog_vn[((((- offset_p_prog_vn_d0) + p_patch_edges_quad_idx_at30) + ((p_prog_vn_d0 * p_prog_vn_d1) * ((- offset_p_prog_vn_d2) + p_patch_edges_quad_blk_at31))) + (p_prog_vn_d0 * (_loop_it_7 - offset_p_prog_vn_d1)))];
                            double _out_p_diag_vt;

                            ///////////////////
                            // Tasklet code (t_0)
                            _out_p_diag_vt = ((((_in_p_int_rbf_vec_coeff_e_0 * _in_p_prog_vn_0) + (_in_p_int_rbf_vec_coeff_e_1 * _in_p_prog_vn_1)) + (_in_p_int_rbf_vec_coeff_e_2 * _in_p_prog_vn_2)) + (_in_p_int_rbf_vec_coeff_e_3 * _in_p_prog_vn_3));
                            ///////////////////

                            p_diag_vt[(((_loop_it_8 - offset_p_diag_vt_d0) + ((p_diag_vt_d0 * p_diag_vt_d1) * (_loop_it_6 - offset_p_diag_vt_d2))) + (p_diag_vt_d0 * (_loop_it_7 - offset_p_diag_vt_d1)))] = _out_p_diag_vt;
                        }

                    }

                }



            }



            for (_loop_it_9 = 2; (_loop_it_9 < (nlev + 1)); _loop_it_9 = (_loop_it_9 + 1)) {

                for (_loop_it_10 = i_startidx; (_loop_it_10 < (i_endidx + 1)); _loop_it_10 = (_loop_it_10 + 1)) {
                    {

                        {
                            double _in_p_metrics_wgtfac_e_0 = p_metrics_wgtfac_e[(((_loop_it_10 - offset_p_metrics_wgtfac_e_d0) + ((p_metrics_wgtfac_e_d0 * p_metrics_wgtfac_e_d1) * (_loop_it_6 - offset_p_metrics_wgtfac_e_d2))) + (p_metrics_wgtfac_e_d0 * (_loop_it_9 - offset_p_metrics_wgtfac_e_d1)))];
                            double _in_p_metrics_wgtfac_e_1 = p_metrics_wgtfac_e[(((_loop_it_10 - offset_p_metrics_wgtfac_e_d0) + ((p_metrics_wgtfac_e_d0 * p_metrics_wgtfac_e_d1) * (_loop_it_6 - offset_p_metrics_wgtfac_e_d2))) + (p_metrics_wgtfac_e_d0 * (_loop_it_9 - offset_p_metrics_wgtfac_e_d1)))];
                            double _in_p_prog_vn_0 = p_prog_vn[(((_loop_it_10 - offset_p_prog_vn_d0) + ((p_prog_vn_d0 * p_prog_vn_d1) * (_loop_it_6 - offset_p_prog_vn_d2))) + (p_prog_vn_d0 * (_loop_it_9 - offset_p_prog_vn_d1)))];
                            double _in_p_prog_vn_1 = p_prog_vn[(((_loop_it_10 - offset_p_prog_vn_d0) + ((p_prog_vn_d0 * p_prog_vn_d1) * (_loop_it_6 - offset_p_prog_vn_d2))) + (p_prog_vn_d0 * ((_loop_it_9 - offset_p_prog_vn_d1) - 1)))];
                            double _out_p_diag_vn_ie;

                            ///////////////////
                            // Tasklet code (t_0)
                            _out_p_diag_vn_ie = ((_in_p_metrics_wgtfac_e_0 * _in_p_prog_vn_0) + ((1.0 - _in_p_metrics_wgtfac_e_1) * _in_p_prog_vn_1));
                            ///////////////////

                            p_diag_vn_ie[(((_loop_it_10 - offset_p_diag_vn_ie_d0) + ((p_diag_vn_ie_d0 * p_diag_vn_ie_d1) * (_loop_it_6 - offset_p_diag_vn_ie_d2))) + (p_diag_vn_ie_d0 * (_loop_it_9 - offset_p_diag_vn_ie_d1)))] = _out_p_diag_vn_ie;
                        }
                        {
                            double _in_p_diag_vt_0 = p_diag_vt[(((_loop_it_10 - offset_p_diag_vt_d0) + ((p_diag_vt_d0 * p_diag_vt_d1) * (_loop_it_6 - offset_p_diag_vt_d2))) + (p_diag_vt_d0 * (_loop_it_9 - offset_p_diag_vt_d1)))];
                            double _in_p_prog_vn_0 = p_prog_vn[(((_loop_it_10 - offset_p_prog_vn_d0) + ((p_prog_vn_d0 * p_prog_vn_d1) * (_loop_it_6 - offset_p_prog_vn_d2))) + (p_prog_vn_d0 * (_loop_it_9 - offset_p_prog_vn_d1)))];
                            double _out_z_kin_hor_e;

                            ///////////////////
                            // Tasklet code (t_1)
                            _out_z_kin_hor_e = (((dace::math::ipow(_in_p_prog_vn_0, 2)) + (dace::math::ipow(_in_p_diag_vt_0, 2))) * 0.5);
                            ///////////////////

                            z_kin_hor_e[(((_loop_it_10 + ((z_kin_hor_e_d0 * z_kin_hor_e_d1) * (_loop_it_6 - 1))) + (z_kin_hor_e_d0 * (_loop_it_9 - 1))) - 1)] = _out_z_kin_hor_e;
                        }

                    }

                }



            }



            if_cond_133 = (lvn_only != true);


            if (if_cond_133) {

                for (_loop_it_11 = 2; (_loop_it_11 < (nlev + 1)); _loop_it_11 = (_loop_it_11 + 1)) {

                    for (_loop_it_12 = i_startidx; (_loop_it_12 < (i_endidx + 1)); _loop_it_12 = (_loop_it_12 + 1)) {
                        {

                            {
                                double _in_p_diag_vt_0 = p_diag_vt[(((_loop_it_12 - offset_p_diag_vt_d0) + ((p_diag_vt_d0 * p_diag_vt_d1) * (_loop_it_6 - offset_p_diag_vt_d2))) + (p_diag_vt_d0 * (_loop_it_11 - offset_p_diag_vt_d1)))];
                                double _in_p_diag_vt_1 = p_diag_vt[(((_loop_it_12 - offset_p_diag_vt_d0) + ((p_diag_vt_d0 * p_diag_vt_d1) * (_loop_it_6 - offset_p_diag_vt_d2))) + (p_diag_vt_d0 * ((_loop_it_11 - offset_p_diag_vt_d1) - 1)))];
                                double _in_p_metrics_wgtfac_e_0 = p_metrics_wgtfac_e[(((_loop_it_12 - offset_p_metrics_wgtfac_e_d0) + ((p_metrics_wgtfac_e_d0 * p_metrics_wgtfac_e_d1) * (_loop_it_6 - offset_p_metrics_wgtfac_e_d2))) + (p_metrics_wgtfac_e_d0 * (_loop_it_11 - offset_p_metrics_wgtfac_e_d1)))];
                                double _in_p_metrics_wgtfac_e_1 = p_metrics_wgtfac_e[(((_loop_it_12 - offset_p_metrics_wgtfac_e_d0) + ((p_metrics_wgtfac_e_d0 * p_metrics_wgtfac_e_d1) * (_loop_it_6 - offset_p_metrics_wgtfac_e_d2))) + (p_metrics_wgtfac_e_d0 * (_loop_it_11 - offset_p_metrics_wgtfac_e_d1)))];
                                double _out_z_vt_ie;

                                ///////////////////
                                // Tasklet code (t_0)
                                _out_z_vt_ie = ((_in_p_metrics_wgtfac_e_0 * _in_p_diag_vt_0) + ((1.0 - _in_p_metrics_wgtfac_e_1) * _in_p_diag_vt_1));
                                ///////////////////

                                z_vt_ie[(((_loop_it_12 + ((z_vt_ie_d0 * z_vt_ie_d1) * (_loop_it_6 - 1))) + (z_vt_ie_d0 * (_loop_it_11 - 1))) - 1)] = _out_z_vt_ie;
                            }

                        }

                    }



                }


            }


            for (_loop_it_13 = nflatlev_jg; (_loop_it_13 < (nlev + 1)); _loop_it_13 = (_loop_it_13 + 1)) {

                for (_loop_it_14 = i_startidx; (_loop_it_14 < (i_endidx + 1)); _loop_it_14 = (_loop_it_14 + 1)) {
                    {

                        {
                            double _in_p_diag_vt_0 = p_diag_vt[(((_loop_it_14 - offset_p_diag_vt_d0) + ((p_diag_vt_d0 * p_diag_vt_d1) * (_loop_it_6 - offset_p_diag_vt_d2))) + (p_diag_vt_d0 * (_loop_it_13 - offset_p_diag_vt_d1)))];
                            double _in_p_metrics_ddxn_z_full_0 = p_metrics_ddxn_z_full[(((_loop_it_14 - offset_p_metrics_ddxn_z_full_d0) + ((p_metrics_ddxn_z_full_d0 * p_metrics_ddxn_z_full_d1) * (_loop_it_6 - offset_p_metrics_ddxn_z_full_d2))) + (p_metrics_ddxn_z_full_d0 * (_loop_it_13 - offset_p_metrics_ddxn_z_full_d1)))];
                            double _in_p_metrics_ddxt_z_full_0 = p_metrics_ddxt_z_full[(((_loop_it_14 - offset_p_metrics_ddxt_z_full_d0) + ((p_metrics_ddxt_z_full_d0 * p_metrics_ddxt_z_full_d1) * (_loop_it_6 - offset_p_metrics_ddxt_z_full_d2))) + (p_metrics_ddxt_z_full_d0 * (_loop_it_13 - offset_p_metrics_ddxt_z_full_d1)))];
                            double _in_p_prog_vn_0 = p_prog_vn[(((_loop_it_14 - offset_p_prog_vn_d0) + ((p_prog_vn_d0 * p_prog_vn_d1) * (_loop_it_6 - offset_p_prog_vn_d2))) + (p_prog_vn_d0 * (_loop_it_13 - offset_p_prog_vn_d1)))];
                            double _out_z_w_concorr_me;

                            ///////////////////
                            // Tasklet code (t_0)
                            _out_z_w_concorr_me = ((_in_p_prog_vn_0 * _in_p_metrics_ddxn_z_full_0) + (_in_p_diag_vt_0 * _in_p_metrics_ddxt_z_full_0));
                            ///////////////////

                            z_w_concorr_me[(((_loop_it_14 + ((z_w_concorr_me_d0 * z_w_concorr_me_d1) * (_loop_it_6 - 1))) + (z_w_concorr_me_d0 * (_loop_it_13 - 1))) - 1)] = _out_z_w_concorr_me;
                        }

                    }

                }



            }



            if_cond_140 = (l_vert_nested != true);


            if (if_cond_140) {

                for (_loop_it_15 = i_startidx; (_loop_it_15 < (i_endidx + 1)); _loop_it_15 = (_loop_it_15 + 1)) {
                    {

                        {
                            double _in_p_diag_vt_0 = p_diag_vt[(((_loop_it_15 - offset_p_diag_vt_d0) + ((p_diag_vt_d0 * p_diag_vt_d1) * (_loop_it_6 - offset_p_diag_vt_d2))) + (p_diag_vt_d0 * (1 - offset_p_diag_vt_d1)))];
                            double _in_p_prog_vn_0 = p_prog_vn[(((_loop_it_15 - offset_p_prog_vn_d0) + ((p_prog_vn_d0 * p_prog_vn_d1) * (_loop_it_6 - offset_p_prog_vn_d2))) + (p_prog_vn_d0 * (1 - offset_p_prog_vn_d1)))];
                            double _out_z_kin_hor_e;

                            ///////////////////
                            // Tasklet code (t_2)
                            _out_z_kin_hor_e = (((dace::math::ipow(_in_p_prog_vn_0, 2)) + (dace::math::ipow(_in_p_diag_vt_0, 2))) * 0.5);
                            ///////////////////

                            z_kin_hor_e[((_loop_it_15 + ((z_kin_hor_e_d0 * z_kin_hor_e_d1) * (_loop_it_6 - 1))) - 1)] = _out_z_kin_hor_e;
                        }
                        {
                            double _in_p_prog_vn_0 = p_prog_vn[(((_loop_it_15 - offset_p_prog_vn_d0) + ((p_prog_vn_d0 * p_prog_vn_d1) * (_loop_it_6 - offset_p_prog_vn_d2))) + (p_prog_vn_d0 * (1 - offset_p_prog_vn_d1)))];
                            double _out_p_diag_vn_ie;

                            ///////////////////
                            // Tasklet code (t_0)
                            _out_p_diag_vn_ie = _in_p_prog_vn_0;
                            ///////////////////

                            p_diag_vn_ie[(((_loop_it_15 - offset_p_diag_vn_ie_d0) + ((p_diag_vn_ie_d0 * p_diag_vn_ie_d1) * (_loop_it_6 - offset_p_diag_vn_ie_d2))) + (p_diag_vn_ie_d0 * (1 - offset_p_diag_vn_ie_d1)))] = _out_p_diag_vn_ie;
                        }
                        {
                            double _in_p_metrics_wgtfacq_e_0 = p_metrics_wgtfacq_e[(((_loop_it_15 - offset_p_metrics_wgtfacq_e_d0) + ((p_metrics_wgtfacq_e_d0 * p_metrics_wgtfacq_e_d1) * (_loop_it_6 - offset_p_metrics_wgtfacq_e_d2))) + (p_metrics_wgtfacq_e_d0 * (1 - offset_p_metrics_wgtfacq_e_d1)))];
                            double _in_p_metrics_wgtfacq_e_1 = p_metrics_wgtfacq_e[(((_loop_it_15 - offset_p_metrics_wgtfacq_e_d0) + ((p_metrics_wgtfacq_e_d0 * p_metrics_wgtfacq_e_d1) * (_loop_it_6 - offset_p_metrics_wgtfacq_e_d2))) + (p_metrics_wgtfacq_e_d0 * (2 - offset_p_metrics_wgtfacq_e_d1)))];
                            double _in_p_metrics_wgtfacq_e_2 = p_metrics_wgtfacq_e[(((_loop_it_15 - offset_p_metrics_wgtfacq_e_d0) + ((p_metrics_wgtfacq_e_d0 * p_metrics_wgtfacq_e_d1) * (_loop_it_6 - offset_p_metrics_wgtfacq_e_d2))) + (p_metrics_wgtfacq_e_d0 * (3 - offset_p_metrics_wgtfacq_e_d1)))];
                            double _in_p_prog_vn_0 = p_prog_vn[(((_loop_it_15 - offset_p_prog_vn_d0) + ((p_prog_vn_d0 * p_prog_vn_d1) * (_loop_it_6 - offset_p_prog_vn_d2))) + (p_prog_vn_d0 * (nlev - offset_p_prog_vn_d1)))];
                            double _in_p_prog_vn_1 = p_prog_vn[(((_loop_it_15 - offset_p_prog_vn_d0) + ((p_prog_vn_d0 * p_prog_vn_d1) * (_loop_it_6 - offset_p_prog_vn_d2))) + (p_prog_vn_d0 * ((nlev - offset_p_prog_vn_d1) - 1)))];
                            double _in_p_prog_vn_2 = p_prog_vn[(((_loop_it_15 - offset_p_prog_vn_d0) + ((p_prog_vn_d0 * p_prog_vn_d1) * (_loop_it_6 - offset_p_prog_vn_d2))) + (p_prog_vn_d0 * ((nlev - offset_p_prog_vn_d1) - 2)))];
                            double _out_p_diag_vn_ie;

                            ///////////////////
                            // Tasklet code (t_3)
                            _out_p_diag_vn_ie = (((_in_p_metrics_wgtfacq_e_0 * _in_p_prog_vn_0) + (_in_p_metrics_wgtfacq_e_1 * _in_p_prog_vn_1)) + (_in_p_metrics_wgtfacq_e_2 * _in_p_prog_vn_2));
                            ///////////////////

                            p_diag_vn_ie[(((_loop_it_15 - offset_p_diag_vn_ie_d0) + ((p_diag_vn_ie_d0 * p_diag_vn_ie_d1) * (_loop_it_6 - offset_p_diag_vn_ie_d2))) + (p_diag_vn_ie_d0 * (nlevp1 - offset_p_diag_vn_ie_d1)))] = _out_p_diag_vn_ie;
                        }
                        {
                            double _in_p_diag_vt_0 = p_diag_vt[(((_loop_it_15 - offset_p_diag_vt_d0) + ((p_diag_vt_d0 * p_diag_vt_d1) * (_loop_it_6 - offset_p_diag_vt_d2))) + (p_diag_vt_d0 * (1 - offset_p_diag_vt_d1)))];
                            double _out_z_vt_ie;

                            ///////////////////
                            // Tasklet code (t_1)
                            _out_z_vt_ie = _in_p_diag_vt_0;
                            ///////////////////

                            z_vt_ie[((_loop_it_15 + ((z_vt_ie_d0 * z_vt_ie_d1) * (_loop_it_6 - 1))) - 1)] = _out_z_vt_ie;
                        }

                    }

                }


            } else {

                for (_loop_it_16 = i_startidx; (_loop_it_16 < (i_endidx + 1)); _loop_it_16 = (_loop_it_16 + 1)) {
                    {

                        {
                            double _in_p_diag_vt_0 = p_diag_vt[(((_loop_it_16 - offset_p_diag_vt_d0) + ((p_diag_vt_d0 * p_diag_vt_d1) * (_loop_it_6 - offset_p_diag_vt_d2))) + (p_diag_vt_d0 * (1 - offset_p_diag_vt_d1)))];
                            double _out_z_vt_ie;

                            ///////////////////
                            // Tasklet code (t_1)
                            _out_z_vt_ie = _in_p_diag_vt_0;
                            ///////////////////

                            z_vt_ie[((_loop_it_16 + ((z_vt_ie_d0 * z_vt_ie_d1) * (_loop_it_6 - 1))) - 1)] = _out_z_vt_ie;
                        }
                        {
                            double _in_p_diag_vt_0 = p_diag_vt[(((_loop_it_16 - offset_p_diag_vt_d0) + ((p_diag_vt_d0 * p_diag_vt_d1) * (_loop_it_6 - offset_p_diag_vt_d2))) + (p_diag_vt_d0 * (1 - offset_p_diag_vt_d1)))];
                            double _in_p_prog_vn_0 = p_prog_vn[(((_loop_it_16 - offset_p_prog_vn_d0) + ((p_prog_vn_d0 * p_prog_vn_d1) * (_loop_it_6 - offset_p_prog_vn_d2))) + (p_prog_vn_d0 * (1 - offset_p_prog_vn_d1)))];
                            double _out_z_kin_hor_e;

                            ///////////////////
                            // Tasklet code (t_2)
                            _out_z_kin_hor_e = (((dace::math::ipow(_in_p_prog_vn_0, 2)) + (dace::math::ipow(_in_p_diag_vt_0, 2))) * 0.5);
                            ///////////////////

                            z_kin_hor_e[((_loop_it_16 + ((z_kin_hor_e_d0 * z_kin_hor_e_d1) * (_loop_it_6 - 1))) - 1)] = _out_z_kin_hor_e;
                        }
                        {
                            double _in_p_diag_vn_ie_ubc_0 = p_diag_vn_ie_ubc[(((_loop_it_16 - offset_p_diag_vn_ie_ubc_d0) + ((p_diag_vn_ie_ubc_d0 * p_diag_vn_ie_ubc_d1) * (_loop_it_6 - offset_p_diag_vn_ie_ubc_d2))) + (p_diag_vn_ie_ubc_d0 * (1 - offset_p_diag_vn_ie_ubc_d1)))];
                            double _in_p_diag_vn_ie_ubc_1 = p_diag_vn_ie_ubc[(((_loop_it_16 - offset_p_diag_vn_ie_ubc_d0) + ((p_diag_vn_ie_ubc_d0 * p_diag_vn_ie_ubc_d1) * (_loop_it_6 - offset_p_diag_vn_ie_ubc_d2))) + (p_diag_vn_ie_ubc_d0 * (2 - offset_p_diag_vn_ie_ubc_d1)))];
                            double _in_dt_linintp_ubc = dt_linintp_ubc;
                            double _out_p_diag_vn_ie;

                            ///////////////////
                            // Tasklet code (t_0)
                            _out_p_diag_vn_ie = (_in_p_diag_vn_ie_ubc_0 + (_in_dt_linintp_ubc * _in_p_diag_vn_ie_ubc_1));
                            ///////////////////

                            p_diag_vn_ie[(((_loop_it_16 - offset_p_diag_vn_ie_d0) + ((p_diag_vn_ie_d0 * p_diag_vn_ie_d1) * (_loop_it_6 - offset_p_diag_vn_ie_d2))) + (p_diag_vn_ie_d0 * (1 - offset_p_diag_vn_ie_d1)))] = _out_p_diag_vn_ie;
                        }
                        {
                            double _in_p_metrics_wgtfacq_e_0 = p_metrics_wgtfacq_e[(((_loop_it_16 - offset_p_metrics_wgtfacq_e_d0) + ((p_metrics_wgtfacq_e_d0 * p_metrics_wgtfacq_e_d1) * (_loop_it_6 - offset_p_metrics_wgtfacq_e_d2))) + (p_metrics_wgtfacq_e_d0 * (1 - offset_p_metrics_wgtfacq_e_d1)))];
                            double _in_p_metrics_wgtfacq_e_1 = p_metrics_wgtfacq_e[(((_loop_it_16 - offset_p_metrics_wgtfacq_e_d0) + ((p_metrics_wgtfacq_e_d0 * p_metrics_wgtfacq_e_d1) * (_loop_it_6 - offset_p_metrics_wgtfacq_e_d2))) + (p_metrics_wgtfacq_e_d0 * (2 - offset_p_metrics_wgtfacq_e_d1)))];
                            double _in_p_metrics_wgtfacq_e_2 = p_metrics_wgtfacq_e[(((_loop_it_16 - offset_p_metrics_wgtfacq_e_d0) + ((p_metrics_wgtfacq_e_d0 * p_metrics_wgtfacq_e_d1) * (_loop_it_6 - offset_p_metrics_wgtfacq_e_d2))) + (p_metrics_wgtfacq_e_d0 * (3 - offset_p_metrics_wgtfacq_e_d1)))];
                            double _in_p_prog_vn_0 = p_prog_vn[(((_loop_it_16 - offset_p_prog_vn_d0) + ((p_prog_vn_d0 * p_prog_vn_d1) * (_loop_it_6 - offset_p_prog_vn_d2))) + (p_prog_vn_d0 * (nlev - offset_p_prog_vn_d1)))];
                            double _in_p_prog_vn_1 = p_prog_vn[(((_loop_it_16 - offset_p_prog_vn_d0) + ((p_prog_vn_d0 * p_prog_vn_d1) * (_loop_it_6 - offset_p_prog_vn_d2))) + (p_prog_vn_d0 * ((nlev - offset_p_prog_vn_d1) - 1)))];
                            double _in_p_prog_vn_2 = p_prog_vn[(((_loop_it_16 - offset_p_prog_vn_d0) + ((p_prog_vn_d0 * p_prog_vn_d1) * (_loop_it_6 - offset_p_prog_vn_d2))) + (p_prog_vn_d0 * ((nlev - offset_p_prog_vn_d1) - 2)))];
                            double _out_p_diag_vn_ie;

                            ///////////////////
                            // Tasklet code (t_3)
                            _out_p_diag_vn_ie = (((_in_p_metrics_wgtfacq_e_0 * _in_p_prog_vn_0) + (_in_p_metrics_wgtfacq_e_1 * _in_p_prog_vn_1)) + (_in_p_metrics_wgtfacq_e_2 * _in_p_prog_vn_2));
                            ///////////////////

                            p_diag_vn_ie[(((_loop_it_16 - offset_p_diag_vn_ie_d0) + ((p_diag_vn_ie_d0 * p_diag_vn_ie_d1) * (_loop_it_6 - offset_p_diag_vn_ie_d2))) + (p_diag_vn_ie_d0 * (nlevp1 - offset_p_diag_vn_ie_d1)))] = _out_p_diag_vn_ie;
                        }

                    }

                }


            }


        }


    }

    {

        {
            int _out;

            ///////////////////
            // Tasklet code (set_rl_start)
            _out = 7;
            ///////////////////

            rl_start = _out;
        }
        {
            int _out;

            ///////////////////
            // Tasklet code (set_rl_end)
            _out = -9;
            ///////////////////

            rl_end = _out;
        }

    }
    i_startblk = p_patch_edges_start_block[(7 - offset_p_patch_edges_start_block_d0)];

    i_endblk = p_patch_edges_end_block[1];


    if (if_cond_19) {

        for (_loop_it_17 = i_startblk; (_loop_it_17 < (i_endblk + 1)); _loop_it_17 = (_loop_it_17 + 1)) {

            f__assoc_scalar_10 = 7;
            {

                {
                    int _out;

                    ///////////////////
                    // Tasklet code (set_f__assoc_scalar_11)
                    _out = -9;
                    ///////////////////

                    f__assoc_scalar_11 = _out;
                }
                {
                    int _in_p_patch_edges_start_index_0 = p_patch_edges_start_index[(f__assoc_scalar_10 - offset_p_patch_edges_start_index_d0)];
                    int _out_get_indices_e_i_startidx_in_inl1;

                    ///////////////////
                    // Tasklet code (t_151)
                    _out_get_indices_e_i_startidx_in_inl1 = _in_p_patch_edges_start_index_0;
                    ///////////////////

                    get_indices_e_i_startidx_in_inl1 = _out_get_indices_e_i_startidx_in_inl1;
                }

            }
            get_indices_e_irl_end_inl1 = f__assoc_scalar_11;
            {

                {
                    int _in_p_patch_edges_end_index_0 = p_patch_edges_end_index[(get_indices_e_irl_end_inl1 + 10)];
                    int _out_get_indices_e_i_endidx_in_inl1;

                    ///////////////////
                    // Tasklet code (t_153)
                    _out_get_indices_e_i_endidx_in_inl1 = _in_p_patch_edges_end_index_0;
                    ///////////////////

                    get_indices_e_i_endidx_in_inl1 = _out_get_indices_e_i_endidx_in_inl1;
                }

            }
            i_startidx = ((_loop_it_17 != i_startblk) ? 1 : ((get_indices_e_i_startidx_in_inl1 < 1) ? 1 : get_indices_e_i_startidx_in_inl1));

            i_endidx = ((_loop_it_17 != i_endblk) ? nproma : get_indices_e_i_endidx_in_inl1);


            for (_loop_it_18 = 1; (_loop_it_18 < (nlev + 1)); _loop_it_18 = (_loop_it_18 + 1)) {

                for (_loop_it_19 = i_startidx; (_loop_it_19 < (i_endidx + 1)); _loop_it_19 = (_loop_it_19 + 1)) {

                    p_patch_edges_cell_idx_at32 = p_patch_edges_cell_idx[(((_loop_it_19 - offset_p_patch_edges_cell_idx_d0) + ((p_patch_edges_cell_idx_d0 * p_patch_edges_cell_idx_d1) * (1 - offset_p_patch_edges_cell_idx_d2))) + (p_patch_edges_cell_idx_d0 * (_loop_it_17 - offset_p_patch_edges_cell_idx_d1)))];

                    p_patch_edges_cell_blk_at33 = p_patch_edges_cell_blk[(((_loop_it_19 - offset_p_patch_edges_cell_blk_d0) + ((p_patch_edges_cell_blk_d0 * p_patch_edges_cell_blk_d1) * (1 - offset_p_patch_edges_cell_blk_d2))) + (p_patch_edges_cell_blk_d0 * (_loop_it_17 - offset_p_patch_edges_cell_blk_d1)))];

                    p_patch_edges_cell_idx_at34 = p_patch_edges_cell_idx[(((_loop_it_19 - offset_p_patch_edges_cell_idx_d0) + ((p_patch_edges_cell_idx_d0 * p_patch_edges_cell_idx_d1) * (2 - offset_p_patch_edges_cell_idx_d2))) + (p_patch_edges_cell_idx_d0 * (_loop_it_17 - offset_p_patch_edges_cell_idx_d1)))];

                    p_patch_edges_cell_blk_at35 = p_patch_edges_cell_blk[(((_loop_it_19 - offset_p_patch_edges_cell_blk_d0) + ((p_patch_edges_cell_blk_d0 * p_patch_edges_cell_blk_d1) * (2 - offset_p_patch_edges_cell_blk_d2))) + (p_patch_edges_cell_blk_d0 * (_loop_it_17 - offset_p_patch_edges_cell_blk_d1)))];

                    p_patch_edges_vertex_idx_at36 = p_patch_edges_vertex_idx[(((_loop_it_19 - offset_p_patch_edges_vertex_idx_d0) + ((p_patch_edges_vertex_idx_d0 * p_patch_edges_vertex_idx_d1) * (1 - offset_p_patch_edges_vertex_idx_d2))) + (p_patch_edges_vertex_idx_d0 * (_loop_it_17 - offset_p_patch_edges_vertex_idx_d1)))];

                    p_patch_edges_vertex_blk_at37 = p_patch_edges_vertex_blk[(((_loop_it_19 - offset_p_patch_edges_vertex_blk_d0) + ((p_patch_edges_vertex_blk_d0 * p_patch_edges_vertex_blk_d1) * (1 - offset_p_patch_edges_vertex_blk_d2))) + (p_patch_edges_vertex_blk_d0 * (_loop_it_17 - offset_p_patch_edges_vertex_blk_d1)))];

                    p_patch_edges_vertex_idx_at38 = p_patch_edges_vertex_idx[(((_loop_it_19 - offset_p_patch_edges_vertex_idx_d0) + ((p_patch_edges_vertex_idx_d0 * p_patch_edges_vertex_idx_d1) * (2 - offset_p_patch_edges_vertex_idx_d2))) + (p_patch_edges_vertex_idx_d0 * (_loop_it_17 - offset_p_patch_edges_vertex_idx_d1)))];

                    p_patch_edges_vertex_blk_at39 = p_patch_edges_vertex_blk[(((_loop_it_19 - offset_p_patch_edges_vertex_blk_d0) + ((p_patch_edges_vertex_blk_d0 * p_patch_edges_vertex_blk_d1) * (2 - offset_p_patch_edges_vertex_blk_d2))) + (p_patch_edges_vertex_blk_d0 * (_loop_it_17 - offset_p_patch_edges_vertex_blk_d1)))];

                    {

                        {
                            double _in_p_diag_vn_ie_0 = p_diag_vn_ie[(((_loop_it_19 - offset_p_diag_vn_ie_d0) + ((p_diag_vn_ie_d0 * p_diag_vn_ie_d1) * (_loop_it_17 - offset_p_diag_vn_ie_d2))) + (p_diag_vn_ie_d0 * (_loop_it_18 - offset_p_diag_vn_ie_d1)))];
                            double _in_p_patch_edges_inv_dual_edge_length_0 = p_patch_edges_inv_dual_edge_length[((_loop_it_19 - offset_p_patch_edges_inv_dual_edge_length_d0) + (p_patch_edges_inv_dual_edge_length_d0 * (_loop_it_17 - offset_p_patch_edges_inv_dual_edge_length_d1)))];
                            double _in_p_patch_edges_inv_primal_edge_length_0 = p_patch_edges_inv_primal_edge_length[((_loop_it_19 - offset_p_patch_edges_inv_primal_edge_length_d0) + (p_patch_edges_inv_primal_edge_length_d0 * (_loop_it_17 - offset_p_patch_edges_inv_primal_edge_length_d1)))];
                            double _in_p_patch_edges_tangent_orientation_0 = p_patch_edges_tangent_orientation[((_loop_it_19 - offset_p_patch_edges_tangent_orientation_d0) + (p_patch_edges_tangent_orientation_d0 * (_loop_it_17 - offset_p_patch_edges_tangent_orientation_d1)))];
                            double _in_p_prog_w_0 = p_prog_w[((((- offset_p_prog_w_d0) + p_patch_edges_cell_idx_at32) + ((p_prog_w_d0 * p_prog_w_d1) * ((- offset_p_prog_w_d2) + p_patch_edges_cell_blk_at33))) + (p_prog_w_d0 * (_loop_it_18 - offset_p_prog_w_d1)))];
                            double _in_p_prog_w_1 = p_prog_w[((((- offset_p_prog_w_d0) + p_patch_edges_cell_idx_at34) + ((p_prog_w_d0 * p_prog_w_d1) * ((- offset_p_prog_w_d2) + p_patch_edges_cell_blk_at35))) + (p_prog_w_d0 * (_loop_it_18 - offset_p_prog_w_d1)))];
                            double _in_z_vt_ie_0 = z_vt_ie[(((_loop_it_19 + ((z_vt_ie_d0 * z_vt_ie_d1) * (_loop_it_17 - 1))) + (z_vt_ie_d0 * (_loop_it_18 - 1))) - 1)];
                            double _in_z_w_v_0 = z_w_v[(((((nproma * p_patch_nlevp1) * (p_patch_edges_vertex_blk_at37 - 1)) + (nproma * (_loop_it_18 - 1))) + p_patch_edges_vertex_idx_at36) - 1)];
                            double _in_z_w_v_1 = z_w_v[(((((nproma * p_patch_nlevp1) * (p_patch_edges_vertex_blk_at39 - 1)) + (nproma * (_loop_it_18 - 1))) + p_patch_edges_vertex_idx_at38) - 1)];
                            double _out_z_v_grad_w;

                            ///////////////////
                            // Tasklet code (t_0)
                            _out_z_v_grad_w = (((_in_p_diag_vn_ie_0 * _in_p_patch_edges_inv_dual_edge_length_0) * (_in_p_prog_w_0 - _in_p_prog_w_1)) + (((_in_z_vt_ie_0 * _in_p_patch_edges_inv_primal_edge_length_0) * _in_p_patch_edges_tangent_orientation_0) * (_in_z_w_v_0 - _in_z_w_v_1)));
                            ///////////////////

                            z_v_grad_w[(((_loop_it_19 + ((nproma * p_patch_nlev) * (_loop_it_17 - 1))) + (nproma * (_loop_it_18 - 1))) - 1)] = _out_z_v_grad_w;
                        }

                    }

                }



            }



        }


    }


    if_cond_168 = ((lvn_only != true) && ldeepatmo);


    if (if_cond_168) {

        for (_loop_it_20 = i_startblk; (_loop_it_20 < (i_endblk + 1)); _loop_it_20 = (_loop_it_20 + 1)) {

            f__assoc_scalar_12 = 7;
            {

                {
                    int _out;

                    ///////////////////
                    // Tasklet code (set_f__assoc_scalar_13)
                    _out = -9;
                    ///////////////////

                    f__assoc_scalar_13 = _out;
                }
                {
                    int _in_p_patch_edges_start_index_0 = p_patch_edges_start_index[(f__assoc_scalar_12 - offset_p_patch_edges_start_index_d0)];
                    int _out_i_startidx_in_inl2;

                    ///////////////////
                    // Tasklet code (t_173)
                    _out_i_startidx_in_inl2 = _in_p_patch_edges_start_index_0;
                    ///////////////////

                    i_startidx_in_inl2 = _out_i_startidx_in_inl2;
                }

            }
            irl_end_inl2 = f__assoc_scalar_13;
            {

                {
                    int _in_p_patch_edges_end_index_0 = p_patch_edges_end_index[(irl_end_inl2 + 10)];
                    int _out_i_endidx_in_inl2;

                    ///////////////////
                    // Tasklet code (t_175)
                    _out_i_endidx_in_inl2 = _in_p_patch_edges_end_index_0;
                    ///////////////////

                    i_endidx_in_inl2 = _out_i_endidx_in_inl2;
                }

            }
            i_startidx = ((_loop_it_20 != i_startblk) ? 1 : ((i_startidx_in_inl2 < 1) ? 1 : i_startidx_in_inl2));

            i_endidx = ((_loop_it_20 != i_endblk) ? nproma : i_endidx_in_inl2);


            for (_loop_it_21 = 1; (_loop_it_21 < (nlev + 1)); _loop_it_21 = (_loop_it_21 + 1)) {

                for (_loop_it_22 = i_startidx; (_loop_it_22 < (i_endidx + 1)); _loop_it_22 = (_loop_it_22 + 1)) {
                    {

                        {
                            double _in_p_diag_vn_ie_0 = p_diag_vn_ie[(((_loop_it_22 - offset_p_diag_vn_ie_d0) + ((p_diag_vn_ie_d0 * p_diag_vn_ie_d1) * (_loop_it_20 - offset_p_diag_vn_ie_d2))) + (p_diag_vn_ie_d0 * (_loop_it_21 - offset_p_diag_vn_ie_d1)))];
                            double _in_p_diag_vn_ie_1 = p_diag_vn_ie[(((_loop_it_22 - offset_p_diag_vn_ie_d0) + ((p_diag_vn_ie_d0 * p_diag_vn_ie_d1) * (_loop_it_20 - offset_p_diag_vn_ie_d2))) + (p_diag_vn_ie_d0 * (_loop_it_21 - offset_p_diag_vn_ie_d1)))];
                            double _in_p_metrics_deepatmo_gradh_ifc_0 = p_metrics_deepatmo_gradh_ifc[(_loop_it_21 - offset_p_metrics_deepatmo_gradh_ifc_d0)];
                            double _in_p_metrics_deepatmo_invr_ifc_0 = p_metrics_deepatmo_invr_ifc[(_loop_it_21 - offset_p_metrics_deepatmo_invr_ifc_d0)];
                            double _in_p_metrics_deepatmo_invr_ifc_1 = p_metrics_deepatmo_invr_ifc[(_loop_it_21 - offset_p_metrics_deepatmo_invr_ifc_d0)];
                            double _in_p_patch_edges_fn_e_0 = p_patch_edges_fn_e[((_loop_it_22 - offset_p_patch_edges_fn_e_d0) + (p_patch_edges_fn_e_d0 * (_loop_it_20 - offset_p_patch_edges_fn_e_d1)))];
                            double _in_p_patch_edges_ft_e_0 = p_patch_edges_ft_e[((_loop_it_22 - offset_p_patch_edges_ft_e_d0) + (p_patch_edges_ft_e_d0 * (_loop_it_20 - offset_p_patch_edges_ft_e_d1)))];
                            double _in_z_v_grad_w_0 = z_v_grad_w[(((_loop_it_22 + ((nproma * p_patch_nlev) * (_loop_it_20 - 1))) + (nproma * (_loop_it_21 - 1))) - 1)];
                            double _in_z_vt_ie_0 = z_vt_ie[(((_loop_it_22 + ((z_vt_ie_d0 * z_vt_ie_d1) * (_loop_it_20 - 1))) + (z_vt_ie_d0 * (_loop_it_21 - 1))) - 1)];
                            double _in_z_vt_ie_1 = z_vt_ie[(((_loop_it_22 + ((z_vt_ie_d0 * z_vt_ie_d1) * (_loop_it_20 - 1))) + (z_vt_ie_d0 * (_loop_it_21 - 1))) - 1)];
                            double _out_z_v_grad_w;

                            ///////////////////
                            // Tasklet code (t_0)
                            _out_z_v_grad_w = (((_in_z_v_grad_w_0 * _in_p_metrics_deepatmo_gradh_ifc_0) + (_in_p_diag_vn_ie_0 * ((_in_p_diag_vn_ie_1 * _in_p_metrics_deepatmo_invr_ifc_0) - _in_p_patch_edges_ft_e_0))) + (_in_z_vt_ie_0 * ((_in_z_vt_ie_1 * _in_p_metrics_deepatmo_invr_ifc_1) + _in_p_patch_edges_fn_e_0)));
                            ///////////////////

                            z_v_grad_w[(((_loop_it_22 + ((nproma * p_patch_nlev) * (_loop_it_20 - 1))) + (nproma * (_loop_it_21 - 1))) - 1)] = _out_z_v_grad_w;
                        }

                    }

                }



            }



        }


    }

    {

        {
            int _out;

            ///////////////////
            // Tasklet code (set_rl_start)
            _out = 4;
            ///////////////////

            rl_start = _out;
        }
        {
            int _out;

            ///////////////////
            // Tasklet code (set_rl_end)
            _out = -5;
            ///////////////////

            rl_end = _out;
        }

    }
    i_startblk = p_patch_cells_start_block[(4 - offset_p_patch_cells_start_block_d0)];

    i_endblk = p_patch_cells_end_block[0];
    {
        int rl_end_2;
        int rl_start_2;

        {
            int _out;

            ///////////////////
            // Tasklet code (set_rl_start_2)
            _out = 5;
            ///////////////////

            rl_start_2 = _out;
        }
        {
            int _out;

            ///////////////////
            // Tasklet code (set_rl_end_2)
            _out = -4;
            ///////////////////

            rl_end_2 = _out;
        }

    }
    i_startblk_2 = p_patch_cells_start_block[(5 - offset_p_patch_cells_start_block_d0)];

    i_endblk_2 = p_patch_cells_end_block[1];


    for (_loop_it_23 = i_startblk; (_loop_it_23 < (i_endblk + 1)); _loop_it_23 = (_loop_it_23 + 1)) {

        f__assoc_scalar_14 = 4;
        {

            {
                int _out;

                ///////////////////
                // Tasklet code (set_f__assoc_scalar_15)
                _out = -5;
                ///////////////////

                f__assoc_scalar_15 = _out;
            }
            {
                int _in_p_patch_cells_start_index_0 = p_patch_cells_start_index[(f__assoc_scalar_14 - offset_p_patch_cells_start_index_d0)];
                int _out_get_indices_c_i_startidx_in;

                ///////////////////
                // Tasklet code (t_188)
                _out_get_indices_c_i_startidx_in = _in_p_patch_cells_start_index_0;
                ///////////////////

                get_indices_c_i_startidx_in = _out_get_indices_c_i_startidx_in;
            }

        }
        get_indices_c_irl_end = f__assoc_scalar_15;
        {

            {
                int _in_p_patch_cells_end_index_0 = p_patch_cells_end_index[(get_indices_c_irl_end + 5)];
                int _out_get_indices_c_i_endidx_in;

                ///////////////////
                // Tasklet code (t_190)
                _out_get_indices_c_i_endidx_in = _in_p_patch_cells_end_index_0;
                ///////////////////

                get_indices_c_i_endidx_in = _out_get_indices_c_i_endidx_in;
            }

        }
        if_cond_191 = (_loop_it_23 == i_startblk);

        if_cond_192 = (_loop_it_23 == i_endblk);


        if (if_cond_191) {

            i_startidx = ((get_indices_c_i_startidx_in < 1) ? 1 : get_indices_c_i_startidx_in);

            i_endidx = nproma;

            if_cond_197 = (_loop_it_23 == i_endblk);


            if (if_cond_197) {

                i_endidx = get_indices_c_i_endidx_in;

            }

        } else if (if_cond_192) {

            i_startidx = 1;

            i_endidx = get_indices_c_i_endidx_in;

        } else {

            i_startidx = 1;

            i_endidx = nproma;

        }


        for (_loop_it_24 = 1; (_loop_it_24 < (nlev + 1)); _loop_it_24 = (_loop_it_24 + 1)) {

            for (_loop_it_25 = i_startidx; (_loop_it_25 < (i_endidx + 1)); _loop_it_25 = (_loop_it_25 + 1)) {

                p_patch_cells_edge_idx_at40 = p_patch_cells_edge_idx[(((_loop_it_25 - offset_p_patch_cells_edge_idx_d0) + ((p_patch_cells_edge_idx_d0 * p_patch_cells_edge_idx_d1) * (1 - offset_p_patch_cells_edge_idx_d2))) + (p_patch_cells_edge_idx_d0 * (_loop_it_23 - offset_p_patch_cells_edge_idx_d1)))];

                p_patch_cells_edge_blk_at41 = p_patch_cells_edge_blk[(((_loop_it_25 - offset_p_patch_cells_edge_blk_d0) + ((p_patch_cells_edge_blk_d0 * p_patch_cells_edge_blk_d1) * (1 - offset_p_patch_cells_edge_blk_d2))) + (p_patch_cells_edge_blk_d0 * (_loop_it_23 - offset_p_patch_cells_edge_blk_d1)))];

                p_patch_cells_edge_idx_at42 = p_patch_cells_edge_idx[(((_loop_it_25 - offset_p_patch_cells_edge_idx_d0) + ((p_patch_cells_edge_idx_d0 * p_patch_cells_edge_idx_d1) * (2 - offset_p_patch_cells_edge_idx_d2))) + (p_patch_cells_edge_idx_d0 * (_loop_it_23 - offset_p_patch_cells_edge_idx_d1)))];

                p_patch_cells_edge_blk_at43 = p_patch_cells_edge_blk[(((_loop_it_25 - offset_p_patch_cells_edge_blk_d0) + ((p_patch_cells_edge_blk_d0 * p_patch_cells_edge_blk_d1) * (2 - offset_p_patch_cells_edge_blk_d2))) + (p_patch_cells_edge_blk_d0 * (_loop_it_23 - offset_p_patch_cells_edge_blk_d1)))];

                p_patch_cells_edge_idx_at44 = p_patch_cells_edge_idx[(((_loop_it_25 - offset_p_patch_cells_edge_idx_d0) + ((p_patch_cells_edge_idx_d0 * p_patch_cells_edge_idx_d1) * (3 - offset_p_patch_cells_edge_idx_d2))) + (p_patch_cells_edge_idx_d0 * (_loop_it_23 - offset_p_patch_cells_edge_idx_d1)))];

                p_patch_cells_edge_blk_at45 = p_patch_cells_edge_blk[(((_loop_it_25 - offset_p_patch_cells_edge_blk_d0) + ((p_patch_cells_edge_blk_d0 * p_patch_cells_edge_blk_d1) * (3 - offset_p_patch_cells_edge_blk_d2))) + (p_patch_cells_edge_blk_d0 * (_loop_it_23 - offset_p_patch_cells_edge_blk_d1)))];

                {

                    {
                        double _in_p_int_e_bln_c_s_0 = p_int_e_bln_c_s[(((_loop_it_25 - offset_p_int_e_bln_c_s_d0) + ((p_int_e_bln_c_s_d0 * p_int_e_bln_c_s_d1) * (_loop_it_23 - offset_p_int_e_bln_c_s_d2))) + (p_int_e_bln_c_s_d0 * (1 - offset_p_int_e_bln_c_s_d1)))];
                        double _in_p_int_e_bln_c_s_1 = p_int_e_bln_c_s[(((_loop_it_25 - offset_p_int_e_bln_c_s_d0) + ((p_int_e_bln_c_s_d0 * p_int_e_bln_c_s_d1) * (_loop_it_23 - offset_p_int_e_bln_c_s_d2))) + (p_int_e_bln_c_s_d0 * (2 - offset_p_int_e_bln_c_s_d1)))];
                        double _in_p_int_e_bln_c_s_2 = p_int_e_bln_c_s[(((_loop_it_25 - offset_p_int_e_bln_c_s_d0) + ((p_int_e_bln_c_s_d0 * p_int_e_bln_c_s_d1) * (_loop_it_23 - offset_p_int_e_bln_c_s_d2))) + (p_int_e_bln_c_s_d0 * (3 - offset_p_int_e_bln_c_s_d1)))];
                        double _in_z_kin_hor_e_0 = z_kin_hor_e[(((p_patch_cells_edge_idx_at40 + ((z_kin_hor_e_d0 * z_kin_hor_e_d1) * (p_patch_cells_edge_blk_at41 - 1))) + (z_kin_hor_e_d0 * (_loop_it_24 - 1))) - 1)];
                        double _in_z_kin_hor_e_1 = z_kin_hor_e[(((p_patch_cells_edge_idx_at42 + ((z_kin_hor_e_d0 * z_kin_hor_e_d1) * (p_patch_cells_edge_blk_at43 - 1))) + (z_kin_hor_e_d0 * (_loop_it_24 - 1))) - 1)];
                        double _in_z_kin_hor_e_2 = z_kin_hor_e[(((p_patch_cells_edge_idx_at44 + ((z_kin_hor_e_d0 * z_kin_hor_e_d1) * (p_patch_cells_edge_blk_at45 - 1))) + (z_kin_hor_e_d0 * (_loop_it_24 - 1))) - 1)];
                        double _out_z_ekinh;

                        ///////////////////
                        // Tasklet code (t_0)
                        _out_z_ekinh = (((_in_p_int_e_bln_c_s_0 * _in_z_kin_hor_e_0) + (_in_p_int_e_bln_c_s_1 * _in_z_kin_hor_e_1)) + (_in_p_int_e_bln_c_s_2 * _in_z_kin_hor_e_2));
                        ///////////////////

                        z_ekinh[(((_loop_it_25 + ((nproma * p_patch_nlev) * (_loop_it_23 - 1))) + (nproma * (_loop_it_24 - 1))) - 1)] = _out_z_ekinh;
                    }

                }

            }



        }



        if_cond_217 = (istep == 1);


        if (if_cond_217) {

            for (_loop_it_26 = nflatlev_jg; (_loop_it_26 < (nlev + 1)); _loop_it_26 = (_loop_it_26 + 1)) {

                for (_loop_it_27 = i_startidx; (_loop_it_27 < (i_endidx + 1)); _loop_it_27 = (_loop_it_27 + 1)) {

                    p_patch_cells_edge_idx_at46 = p_patch_cells_edge_idx[(((_loop_it_27 - offset_p_patch_cells_edge_idx_d0) + ((p_patch_cells_edge_idx_d0 * p_patch_cells_edge_idx_d1) * (1 - offset_p_patch_cells_edge_idx_d2))) + (p_patch_cells_edge_idx_d0 * (_loop_it_23 - offset_p_patch_cells_edge_idx_d1)))];

                    p_patch_cells_edge_blk_at47 = p_patch_cells_edge_blk[(((_loop_it_27 - offset_p_patch_cells_edge_blk_d0) + ((p_patch_cells_edge_blk_d0 * p_patch_cells_edge_blk_d1) * (1 - offset_p_patch_cells_edge_blk_d2))) + (p_patch_cells_edge_blk_d0 * (_loop_it_23 - offset_p_patch_cells_edge_blk_d1)))];

                    p_patch_cells_edge_idx_at48 = p_patch_cells_edge_idx[(((_loop_it_27 - offset_p_patch_cells_edge_idx_d0) + ((p_patch_cells_edge_idx_d0 * p_patch_cells_edge_idx_d1) * (2 - offset_p_patch_cells_edge_idx_d2))) + (p_patch_cells_edge_idx_d0 * (_loop_it_23 - offset_p_patch_cells_edge_idx_d1)))];

                    p_patch_cells_edge_blk_at49 = p_patch_cells_edge_blk[(((_loop_it_27 - offset_p_patch_cells_edge_blk_d0) + ((p_patch_cells_edge_blk_d0 * p_patch_cells_edge_blk_d1) * (2 - offset_p_patch_cells_edge_blk_d2))) + (p_patch_cells_edge_blk_d0 * (_loop_it_23 - offset_p_patch_cells_edge_blk_d1)))];

                    p_patch_cells_edge_idx_at50 = p_patch_cells_edge_idx[(((_loop_it_27 - offset_p_patch_cells_edge_idx_d0) + ((p_patch_cells_edge_idx_d0 * p_patch_cells_edge_idx_d1) * (3 - offset_p_patch_cells_edge_idx_d2))) + (p_patch_cells_edge_idx_d0 * (_loop_it_23 - offset_p_patch_cells_edge_idx_d1)))];

                    p_patch_cells_edge_blk_at51 = p_patch_cells_edge_blk[(((_loop_it_27 - offset_p_patch_cells_edge_blk_d0) + ((p_patch_cells_edge_blk_d0 * p_patch_cells_edge_blk_d1) * (3 - offset_p_patch_cells_edge_blk_d2))) + (p_patch_cells_edge_blk_d0 * (_loop_it_23 - offset_p_patch_cells_edge_blk_d1)))];

                    {

                        {
                            double _in_p_int_e_bln_c_s_0 = p_int_e_bln_c_s[(((_loop_it_27 - offset_p_int_e_bln_c_s_d0) + ((p_int_e_bln_c_s_d0 * p_int_e_bln_c_s_d1) * (_loop_it_23 - offset_p_int_e_bln_c_s_d2))) + (p_int_e_bln_c_s_d0 * (1 - offset_p_int_e_bln_c_s_d1)))];
                            double _in_p_int_e_bln_c_s_1 = p_int_e_bln_c_s[(((_loop_it_27 - offset_p_int_e_bln_c_s_d0) + ((p_int_e_bln_c_s_d0 * p_int_e_bln_c_s_d1) * (_loop_it_23 - offset_p_int_e_bln_c_s_d2))) + (p_int_e_bln_c_s_d0 * (2 - offset_p_int_e_bln_c_s_d1)))];
                            double _in_p_int_e_bln_c_s_2 = p_int_e_bln_c_s[(((_loop_it_27 - offset_p_int_e_bln_c_s_d0) + ((p_int_e_bln_c_s_d0 * p_int_e_bln_c_s_d1) * (_loop_it_23 - offset_p_int_e_bln_c_s_d2))) + (p_int_e_bln_c_s_d0 * (3 - offset_p_int_e_bln_c_s_d1)))];
                            double _in_z_w_concorr_me_0 = z_w_concorr_me[(((p_patch_cells_edge_idx_at46 + ((z_w_concorr_me_d0 * z_w_concorr_me_d1) * (p_patch_cells_edge_blk_at47 - 1))) + (z_w_concorr_me_d0 * (_loop_it_26 - 1))) - 1)];
                            double _in_z_w_concorr_me_1 = z_w_concorr_me[(((p_patch_cells_edge_idx_at48 + ((z_w_concorr_me_d0 * z_w_concorr_me_d1) * (p_patch_cells_edge_blk_at49 - 1))) + (z_w_concorr_me_d0 * (_loop_it_26 - 1))) - 1)];
                            double _in_z_w_concorr_me_2 = z_w_concorr_me[(((p_patch_cells_edge_idx_at50 + ((z_w_concorr_me_d0 * z_w_concorr_me_d1) * (p_patch_cells_edge_blk_at51 - 1))) + (z_w_concorr_me_d0 * (_loop_it_26 - 1))) - 1)];
                            double _out_z_w_concorr_mc;

                            ///////////////////
                            // Tasklet code (t_0)
                            _out_z_w_concorr_mc = (((_in_p_int_e_bln_c_s_0 * _in_z_w_concorr_me_0) + (_in_p_int_e_bln_c_s_1 * _in_z_w_concorr_me_1)) + (_in_p_int_e_bln_c_s_2 * _in_z_w_concorr_me_2));
                            ///////////////////

                            z_w_concorr_mc[((_loop_it_27 + (nproma * (_loop_it_26 - 1))) - 1)] = _out_z_w_concorr_mc;
                        }

                    }

                }



            }



            loopbegin_228 = (nflatlev_jg + 1);


            for (_loop_it_28 = loopbegin_228; (_loop_it_28 < (nlev + 1)); _loop_it_28 = (_loop_it_28 + 1)) {

                for (_loop_it_29 = i_startidx; (_loop_it_29 < (i_endidx + 1)); _loop_it_29 = (_loop_it_29 + 1)) {
                    {

                        {
                            double _in_p_metrics_wgtfac_c_0 = p_metrics_wgtfac_c[(((_loop_it_29 - offset_p_metrics_wgtfac_c_d0) + ((p_metrics_wgtfac_c_d0 * p_metrics_wgtfac_c_d1) * (_loop_it_23 - offset_p_metrics_wgtfac_c_d2))) + (p_metrics_wgtfac_c_d0 * (_loop_it_28 - offset_p_metrics_wgtfac_c_d1)))];
                            double _in_p_metrics_wgtfac_c_1 = p_metrics_wgtfac_c[(((_loop_it_29 - offset_p_metrics_wgtfac_c_d0) + ((p_metrics_wgtfac_c_d0 * p_metrics_wgtfac_c_d1) * (_loop_it_23 - offset_p_metrics_wgtfac_c_d2))) + (p_metrics_wgtfac_c_d0 * (_loop_it_28 - offset_p_metrics_wgtfac_c_d1)))];
                            double _in_z_w_concorr_mc_0 = z_w_concorr_mc[((_loop_it_29 + (nproma * (_loop_it_28 - 1))) - 1)];
                            double _in_z_w_concorr_mc_1 = z_w_concorr_mc[((_loop_it_29 + (nproma * (_loop_it_28 - 2))) - 1)];
                            double _out_p_diag_w_concorr_c;

                            ///////////////////
                            // Tasklet code (t_0)
                            _out_p_diag_w_concorr_c = ((_in_p_metrics_wgtfac_c_0 * _in_z_w_concorr_mc_0) + ((1.0 - _in_p_metrics_wgtfac_c_1) * _in_z_w_concorr_mc_1));
                            ///////////////////

                            p_diag_w_concorr_c[(((_loop_it_29 - offset_p_diag_w_concorr_c_d0) + ((p_diag_w_concorr_c_d0 * p_diag_w_concorr_c_d1) * (_loop_it_23 - offset_p_diag_w_concorr_c_d2))) + (p_diag_w_concorr_c_d0 * (_loop_it_28 - offset_p_diag_w_concorr_c_d1)))] = _out_p_diag_w_concorr_c;
                        }

                    }

                }



            }


        }


        for (_loop_it_30 = 1; (_loop_it_30 < (nlev + 1)); _loop_it_30 = (_loop_it_30 + 1)) {

            for (_loop_it_31 = i_startidx; (_loop_it_31 < (i_endidx + 1)); _loop_it_31 = (_loop_it_31 + 1)) {
                {

                    {
                        double _in_p_prog_w_0 = p_prog_w[(((_loop_it_31 - offset_p_prog_w_d0) + ((p_prog_w_d0 * p_prog_w_d1) * (_loop_it_23 - offset_p_prog_w_d2))) + (p_prog_w_d0 * (_loop_it_30 - offset_p_prog_w_d1)))];
                        double _out_z_w_con_c;

                        ///////////////////
                        // Tasklet code (t_0)
                        _out_z_w_con_c = _in_p_prog_w_0;
                        ///////////////////

                        z_w_con_c[((_loop_it_31 + (nproma * (_loop_it_30 - 1))) - 1)] = _out_z_w_con_c;
                    }

                }

            }



        }



        for (_loop_it_32 = i_startidx; (_loop_it_32 < (i_endidx + 1)); _loop_it_32 = (_loop_it_32 + 1)) {
            {

                {
                    double _out_z_w_con_c;

                    ///////////////////
                    // Tasklet code (t_0)
                    _out_z_w_con_c = 0.0;
                    ///////////////////

                    z_w_con_c[((_loop_it_32 + (nproma * (nlevp1 - 1))) - 1)] = _out_z_w_con_c;
                }

            }

        }



        loopend_235 = (nflatlev_jg + 1);


        for (_loop_it_33 = nlev; (_loop_it_33 >= loopend_235); _loop_it_33 = (_loop_it_33 - 1)) {

            for (_loop_it_34 = i_startidx; (_loop_it_34 < (i_endidx + 1)); _loop_it_34 = (_loop_it_34 + 1)) {
                {

                    {
                        double _in_p_diag_w_concorr_c_0 = p_diag_w_concorr_c[(((_loop_it_34 - offset_p_diag_w_concorr_c_d0) + ((p_diag_w_concorr_c_d0 * p_diag_w_concorr_c_d1) * (_loop_it_23 - offset_p_diag_w_concorr_c_d2))) + (p_diag_w_concorr_c_d0 * (_loop_it_33 - offset_p_diag_w_concorr_c_d1)))];
                        double _in_z_w_con_c_0 = z_w_con_c[((_loop_it_34 + (nproma * (_loop_it_33 - 1))) - 1)];
                        double _out_z_w_con_c;

                        ///////////////////
                        // Tasklet code (t_0)
                        _out_z_w_con_c = (_in_z_w_con_c_0 - _in_p_diag_w_concorr_c_0);
                        ///////////////////

                        z_w_con_c[((_loop_it_34 + (nproma * (_loop_it_33 - 1))) - 1)] = _out_z_w_con_c;
                    }

                }

            }



        }



        loopend_239 = (nlev - 3);

        loopbegin_241 = max(3, (nrdmax_jg - 2));


        for (_loop_it_35 = loopbegin_241; (_loop_it_35 < (loopend_239 + 1)); _loop_it_35 = (_loop_it_35 + 1)) {
            {

                {
                    bool _out_levmask;

                    ///////////////////
                    // Tasklet code (t_0)
                    _out_levmask = 0;
                    ///////////////////

                    levmask[((_loop_it_23 + (p_patch_nblks_c * (_loop_it_35 - 1))) - 1)] = _out_levmask;
                }

            }

        }


        {

            {
                double _out;

                ///////////////////
                // Tasklet code (set_maxvcfl)
                _out = 0.0;
                ///////////////////

                maxvcfl = _out;
            }

        }
        loopend_244 = (nlev - 3);

        loopbegin_245 = max(3, (nrdmax_jg - 2));


        for (_loop_it_36 = loopbegin_245; (_loop_it_36 < (loopend_244 + 1)); _loop_it_36 = (_loop_it_36 + 1)) {

            clip_count = 0;


            for (_loop_it_37 = i_startidx; (_loop_it_37 < (i_endidx + 1)); _loop_it_37 = (_loop_it_37 + 1)) {
                {

                    {
                        double _in_p_metrics_ddqz_z_half_0 = p_metrics_ddqz_z_half[(((_loop_it_37 - offset_p_metrics_ddqz_z_half_d0) + ((p_metrics_ddqz_z_half_d0 * p_metrics_ddqz_z_half_d1) * (_loop_it_23 - offset_p_metrics_ddqz_z_half_d2))) + (p_metrics_ddqz_z_half_d0 * (_loop_it_36 - offset_p_metrics_ddqz_z_half_d1)))];
                        double _in_z_w_con_c_0 = z_w_con_c[((_loop_it_37 + (nproma * (_loop_it_36 - 1))) - 1)];
                        double _in_cfl_w_limit = cfl_w_limit;
                        bool _out_cfl_clipping;

                        ///////////////////
                        // Tasklet code (t_251)
                        _out_cfl_clipping = (abs(_in_z_w_con_c_0) > (_in_cfl_w_limit * _in_p_metrics_ddqz_z_half_0));
                        ///////////////////

                        cfl_clipping[((_loop_it_37 + (nproma * (_loop_it_36 - 1))) - 1)] = _out_cfl_clipping;
                    }

                }
                {

                    {
                        bool _in_cfl_clipping_0 = cfl_clipping[((_loop_it_37 + (nproma * (_loop_it_36 - 1))) - 1)];
                        int64_t _out_if_cond_252;

                        ///////////////////
                        // Tasklet code (t_253)
                        _out_if_cond_252 = _in_cfl_clipping_0;
                        ///////////////////

                        if_cond_252 = _out_if_cond_252;
                    }

                }

                if (if_cond_252) {

                    clip_count = (clip_count + 1);

                }


            }



            if_cond_258 = (clip_count != 0);


            if (if_cond_258) {

                for (_loop_it_38 = i_startidx; (_loop_it_38 < (i_endidx + 1)); _loop_it_38 = (_loop_it_38 + 1)) {

                    {

                        {
                            bool _in_cfl_clipping_0 = cfl_clipping[((_loop_it_38 + (nproma * (_loop_it_36 - 1))) - 1)];
                            int64_t _out_if_cond_262;

                            ///////////////////
                            // Tasklet code (t_263)
                            _out_if_cond_262 = _in_cfl_clipping_0;
                            ///////////////////

                            if_cond_262 = _out_if_cond_262;
                        }

                    }

                    if (if_cond_262) {
                        {

                            {
                                bool _out_levmask;

                                ///////////////////
                                // Tasklet code (t_266)
                                _out_levmask = -1;
                                ///////////////////

                                levmask[((_loop_it_23 + (p_patch_nblks_c * (_loop_it_36 - 1))) - 1)] = _out_levmask;
                            }
                            {
                                double _in_p_metrics_ddqz_z_half_0 = p_metrics_ddqz_z_half[(((_loop_it_38 - offset_p_metrics_ddqz_z_half_d0) + ((p_metrics_ddqz_z_half_d0 * p_metrics_ddqz_z_half_d1) * (_loop_it_23 - offset_p_metrics_ddqz_z_half_d2))) + (p_metrics_ddqz_z_half_d0 * (_loop_it_36 - offset_p_metrics_ddqz_z_half_d1)))];
                                double _in_z_w_con_c_0 = z_w_con_c[((_loop_it_38 + (nproma * (_loop_it_36 - 1))) - 1)];
                                double _in_dtime = dtime;
                                double _out_vcfl;

                                ///////////////////
                                // Tasklet code (t_267)
                                _out_vcfl = ((_in_z_w_con_c_0 * _in_dtime) / _in_p_metrics_ddqz_z_half_0);
                                ///////////////////

                                vcfl = _out_vcfl;
                            }
                            {
                                double _in_maxvcfl = maxvcfl;
                                double _in_vcfl = vcfl;
                                double _out;

                                ///////////////////
                                // Tasklet code (set_maxvcfl)
                                _out = max(_in_maxvcfl, abs(_in_vcfl));
                                ///////////////////

                                maxvcfl = _out;
                            }

                        }
                        if_cond_268 = (vcfl < -0.85);

                        if_cond_269 = (vcfl > 0.85);


                        if (if_cond_268) {
                            {

                                {
                                    double _in_p_metrics_ddqz_z_half_0 = p_metrics_ddqz_z_half[(((_loop_it_38 - offset_p_metrics_ddqz_z_half_d0) + ((p_metrics_ddqz_z_half_d0 * p_metrics_ddqz_z_half_d1) * (_loop_it_23 - offset_p_metrics_ddqz_z_half_d2))) + (p_metrics_ddqz_z_half_d0 * (_loop_it_36 - offset_p_metrics_ddqz_z_half_d1)))];
                                    double _in_dtime = dtime;
                                    double _out_z_w_con_c;

                                    ///////////////////
                                    // Tasklet code (t_272)
                                    _out_z_w_con_c = (- ((_in_p_metrics_ddqz_z_half_0 * 0.85) / _in_dtime));
                                    ///////////////////

                                    z_w_con_c[((_loop_it_38 + (nproma * (_loop_it_36 - 1))) - 1)] = _out_z_w_con_c;
                                }

                            }
                        } else if (if_cond_269) {
                            {

                                {
                                    double _in_p_metrics_ddqz_z_half_0 = p_metrics_ddqz_z_half[(((_loop_it_38 - offset_p_metrics_ddqz_z_half_d0) + ((p_metrics_ddqz_z_half_d0 * p_metrics_ddqz_z_half_d1) * (_loop_it_23 - offset_p_metrics_ddqz_z_half_d2))) + (p_metrics_ddqz_z_half_d0 * (_loop_it_36 - offset_p_metrics_ddqz_z_half_d1)))];
                                    double _in_dtime = dtime;
                                    double _out_z_w_con_c;

                                    ///////////////////
                                    // Tasklet code (t_274)
                                    _out_z_w_con_c = ((_in_p_metrics_ddqz_z_half_0 * 0.85) / _in_dtime);
                                    ///////////////////

                                    z_w_con_c[((_loop_it_38 + (nproma * (_loop_it_36 - 1))) - 1)] = _out_z_w_con_c;
                                }

                            }
                        }

                    }


                }


            }


        }



        for (_loop_it_39 = 1; (_loop_it_39 < (nlev + 1)); _loop_it_39 = (_loop_it_39 + 1)) {

            for (_loop_it_40 = i_startidx; (_loop_it_40 < (i_endidx + 1)); _loop_it_40 = (_loop_it_40 + 1)) {
                {

                    {
                        double _in_z_w_con_c_0 = z_w_con_c[((_loop_it_40 + (nproma * (_loop_it_39 - 1))) - 1)];
                        double _in_z_w_con_c_1 = z_w_con_c[(((_loop_it_39 * nproma) + _loop_it_40) - 1)];
                        double _out_z_w_con_c_full;

                        ///////////////////
                        // Tasklet code (t_0)
                        _out_z_w_con_c_full = ((_in_z_w_con_c_0 + _in_z_w_con_c_1) * 0.5);
                        ///////////////////

                        z_w_con_c_full[(((_loop_it_40 + ((nproma * p_patch_nlev) * (_loop_it_23 - 1))) + (nproma * (_loop_it_39 - 1))) - 1)] = _out_z_w_con_c_full;
                    }

                }

            }



        }


        {

            {
                double _in_maxvcfl = maxvcfl;
                double _out_vcflmax;

                ///////////////////
                // Tasklet code (t_278)
                _out_vcflmax = _in_maxvcfl;
                ///////////////////

                vcflmax[(_loop_it_23 - 1)] = _out_vcflmax;
            }

        }
        if_cond_279 = (lvn_only != true);


        if (if_cond_279) {

            if_cond_282 = (((_loop_it_23 < i_startblk_2) || (_loop_it_23 > i_endblk_2)) != true);


            if (if_cond_282) {

                f__assoc_scalar_16 = 5;
                {

                    {
                        int _out;

                        ///////////////////
                        // Tasklet code (set_f__assoc_scalar_17)
                        _out = -4;
                        ///////////////////

                        f__assoc_scalar_17 = _out;
                    }
                    {
                        int _in_p_patch_cells_start_index_0 = p_patch_cells_start_index[(f__assoc_scalar_16 - offset_p_patch_cells_start_index_d0)];
                        int _out_get_indices_c_i_startidx_in_inl1;

                        ///////////////////
                        // Tasklet code (t_286)
                        _out_get_indices_c_i_startidx_in_inl1 = _in_p_patch_cells_start_index_0;
                        ///////////////////

                        get_indices_c_i_startidx_in_inl1 = _out_get_indices_c_i_startidx_in_inl1;
                    }

                }
                get_indices_c_irl_end_inl1 = f__assoc_scalar_17;
                {

                    {
                        int _in_p_patch_cells_end_index_0 = p_patch_cells_end_index[(get_indices_c_irl_end_inl1 + 5)];
                        int _out_get_indices_c_i_endidx_in_inl1;

                        ///////////////////
                        // Tasklet code (t_288)
                        _out_get_indices_c_i_endidx_in_inl1 = _in_p_patch_cells_end_index_0;
                        ///////////////////

                        get_indices_c_i_endidx_in_inl1 = _out_get_indices_c_i_endidx_in_inl1;
                    }

                }
                if_cond_289 = (_loop_it_23 == i_startblk_2);

                if_cond_290 = (_loop_it_23 == i_endblk_2);


                if (if_cond_289) {

                    i_startidx_2 = ((get_indices_c_i_startidx_in_inl1 < 1) ? 1 : get_indices_c_i_startidx_in_inl1);

                    i_endidx_2 = nproma;

                    if_cond_295 = (_loop_it_23 == i_endblk_2);


                    if (if_cond_295) {

                        i_endidx_2 = get_indices_c_i_endidx_in_inl1;

                    }

                } else if (if_cond_290) {

                    i_startidx_2 = 1;

                    i_endidx_2 = get_indices_c_i_endidx_in_inl1;

                } else {

                    i_startidx_2 = 1;

                    i_endidx_2 = nproma;

                }


                for (_loop_it_41 = 2; (_loop_it_41 < (nlev + 1)); _loop_it_41 = (_loop_it_41 + 1)) {

                    for (_loop_it_42 = i_startidx_2; (_loop_it_42 < (i_endidx_2 + 1)); _loop_it_42 = (_loop_it_42 + 1)) {
                        {

                            {
                                double _in_p_metrics_coeff1_dwdz_0 = p_metrics_coeff1_dwdz[(((_loop_it_42 - offset_p_metrics_coeff1_dwdz_d0) + ((p_metrics_coeff1_dwdz_d0 * p_metrics_coeff1_dwdz_d1) * (_loop_it_23 - offset_p_metrics_coeff1_dwdz_d2))) + (p_metrics_coeff1_dwdz_d0 * (_loop_it_41 - offset_p_metrics_coeff1_dwdz_d1)))];
                                double _in_p_metrics_coeff1_dwdz_1 = p_metrics_coeff1_dwdz[(((_loop_it_42 - offset_p_metrics_coeff1_dwdz_d0) + ((p_metrics_coeff1_dwdz_d0 * p_metrics_coeff1_dwdz_d1) * (_loop_it_23 - offset_p_metrics_coeff1_dwdz_d2))) + (p_metrics_coeff1_dwdz_d0 * (_loop_it_41 - offset_p_metrics_coeff1_dwdz_d1)))];
                                double _in_p_metrics_coeff2_dwdz_0 = p_metrics_coeff2_dwdz[(((_loop_it_42 - offset_p_metrics_coeff2_dwdz_d0) + ((p_metrics_coeff2_dwdz_d0 * p_metrics_coeff2_dwdz_d1) * (_loop_it_23 - offset_p_metrics_coeff2_dwdz_d2))) + (p_metrics_coeff2_dwdz_d0 * (_loop_it_41 - offset_p_metrics_coeff2_dwdz_d1)))];
                                double _in_p_metrics_coeff2_dwdz_1 = p_metrics_coeff2_dwdz[(((_loop_it_42 - offset_p_metrics_coeff2_dwdz_d0) + ((p_metrics_coeff2_dwdz_d0 * p_metrics_coeff2_dwdz_d1) * (_loop_it_23 - offset_p_metrics_coeff2_dwdz_d2))) + (p_metrics_coeff2_dwdz_d0 * (_loop_it_41 - offset_p_metrics_coeff2_dwdz_d1)))];
                                double _in_p_prog_w_0 = p_prog_w[(((_loop_it_42 - offset_p_prog_w_d0) + ((p_prog_w_d0 * p_prog_w_d1) * (_loop_it_23 - offset_p_prog_w_d2))) + (p_prog_w_d0 * ((_loop_it_41 - offset_p_prog_w_d1) - 1)))];
                                double _in_p_prog_w_1 = p_prog_w[(((_loop_it_42 - offset_p_prog_w_d0) + ((p_prog_w_d0 * p_prog_w_d1) * (_loop_it_23 - offset_p_prog_w_d2))) + (p_prog_w_d0 * ((_loop_it_41 - offset_p_prog_w_d1) + 1)))];
                                double _in_p_prog_w_2 = p_prog_w[(((_loop_it_42 - offset_p_prog_w_d0) + ((p_prog_w_d0 * p_prog_w_d1) * (_loop_it_23 - offset_p_prog_w_d2))) + (p_prog_w_d0 * (_loop_it_41 - offset_p_prog_w_d1)))];
                                double _in_z_w_con_c_0 = z_w_con_c[((_loop_it_42 + (nproma * (_loop_it_41 - 1))) - 1)];
                                double _out_p_diag_ddt_w_adv_pc;

                                ///////////////////
                                // Tasklet code (t_0)
                                _out_p_diag_ddt_w_adv_pc = (- (_in_z_w_con_c_0 * (((_in_p_prog_w_0 * _in_p_metrics_coeff1_dwdz_0) - (_in_p_prog_w_1 * _in_p_metrics_coeff2_dwdz_0)) + (_in_p_prog_w_2 * (_in_p_metrics_coeff2_dwdz_1 - _in_p_metrics_coeff1_dwdz_1)))));
                                ///////////////////

                                p_diag_ddt_w_adv_pc[((((_loop_it_42 - offset_p_diag_ddt_w_adv_pc_d0) + (((p_diag_ddt_w_adv_pc_d0 * p_diag_ddt_w_adv_pc_d1) * p_diag_ddt_w_adv_pc_d2) * (ntnd - offset_p_diag_ddt_w_adv_pc_d3))) + ((p_diag_ddt_w_adv_pc_d0 * p_diag_ddt_w_adv_pc_d1) * (_loop_it_23 - offset_p_diag_ddt_w_adv_pc_d2))) + (p_diag_ddt_w_adv_pc_d0 * (_loop_it_41 - offset_p_diag_ddt_w_adv_pc_d1)))] = _out_p_diag_ddt_w_adv_pc;
                            }

                        }

                    }



                }



                for (_loop_it_43 = 2; (_loop_it_43 < (nlev + 1)); _loop_it_43 = (_loop_it_43 + 1)) {

                    for (_loop_it_44 = i_startidx_2; (_loop_it_44 < (i_endidx_2 + 1)); _loop_it_44 = (_loop_it_44 + 1)) {

                        p_patch_cells_edge_idx_at52 = p_patch_cells_edge_idx[(((_loop_it_44 - offset_p_patch_cells_edge_idx_d0) + ((p_patch_cells_edge_idx_d0 * p_patch_cells_edge_idx_d1) * (1 - offset_p_patch_cells_edge_idx_d2))) + (p_patch_cells_edge_idx_d0 * (_loop_it_23 - offset_p_patch_cells_edge_idx_d1)))];

                        p_patch_cells_edge_blk_at53 = p_patch_cells_edge_blk[(((_loop_it_44 - offset_p_patch_cells_edge_blk_d0) + ((p_patch_cells_edge_blk_d0 * p_patch_cells_edge_blk_d1) * (1 - offset_p_patch_cells_edge_blk_d2))) + (p_patch_cells_edge_blk_d0 * (_loop_it_23 - offset_p_patch_cells_edge_blk_d1)))];

                        p_patch_cells_edge_idx_at54 = p_patch_cells_edge_idx[(((_loop_it_44 - offset_p_patch_cells_edge_idx_d0) + ((p_patch_cells_edge_idx_d0 * p_patch_cells_edge_idx_d1) * (2 - offset_p_patch_cells_edge_idx_d2))) + (p_patch_cells_edge_idx_d0 * (_loop_it_23 - offset_p_patch_cells_edge_idx_d1)))];

                        p_patch_cells_edge_blk_at55 = p_patch_cells_edge_blk[(((_loop_it_44 - offset_p_patch_cells_edge_blk_d0) + ((p_patch_cells_edge_blk_d0 * p_patch_cells_edge_blk_d1) * (2 - offset_p_patch_cells_edge_blk_d2))) + (p_patch_cells_edge_blk_d0 * (_loop_it_23 - offset_p_patch_cells_edge_blk_d1)))];

                        p_patch_cells_edge_idx_at56 = p_patch_cells_edge_idx[(((_loop_it_44 - offset_p_patch_cells_edge_idx_d0) + ((p_patch_cells_edge_idx_d0 * p_patch_cells_edge_idx_d1) * (3 - offset_p_patch_cells_edge_idx_d2))) + (p_patch_cells_edge_idx_d0 * (_loop_it_23 - offset_p_patch_cells_edge_idx_d1)))];

                        p_patch_cells_edge_blk_at57 = p_patch_cells_edge_blk[(((_loop_it_44 - offset_p_patch_cells_edge_blk_d0) + ((p_patch_cells_edge_blk_d0 * p_patch_cells_edge_blk_d1) * (3 - offset_p_patch_cells_edge_blk_d2))) + (p_patch_cells_edge_blk_d0 * (_loop_it_23 - offset_p_patch_cells_edge_blk_d1)))];

                        {

                            {
                                double _in_p_diag_ddt_w_adv_pc_0 = p_diag_ddt_w_adv_pc[((((_loop_it_44 - offset_p_diag_ddt_w_adv_pc_d0) + (((p_diag_ddt_w_adv_pc_d0 * p_diag_ddt_w_adv_pc_d1) * p_diag_ddt_w_adv_pc_d2) * (ntnd - offset_p_diag_ddt_w_adv_pc_d3))) + ((p_diag_ddt_w_adv_pc_d0 * p_diag_ddt_w_adv_pc_d1) * (_loop_it_23 - offset_p_diag_ddt_w_adv_pc_d2))) + (p_diag_ddt_w_adv_pc_d0 * (_loop_it_43 - offset_p_diag_ddt_w_adv_pc_d1)))];
                                double _in_p_int_e_bln_c_s_0 = p_int_e_bln_c_s[(((_loop_it_44 - offset_p_int_e_bln_c_s_d0) + ((p_int_e_bln_c_s_d0 * p_int_e_bln_c_s_d1) * (_loop_it_23 - offset_p_int_e_bln_c_s_d2))) + (p_int_e_bln_c_s_d0 * (1 - offset_p_int_e_bln_c_s_d1)))];
                                double _in_p_int_e_bln_c_s_1 = p_int_e_bln_c_s[(((_loop_it_44 - offset_p_int_e_bln_c_s_d0) + ((p_int_e_bln_c_s_d0 * p_int_e_bln_c_s_d1) * (_loop_it_23 - offset_p_int_e_bln_c_s_d2))) + (p_int_e_bln_c_s_d0 * (2 - offset_p_int_e_bln_c_s_d1)))];
                                double _in_p_int_e_bln_c_s_2 = p_int_e_bln_c_s[(((_loop_it_44 - offset_p_int_e_bln_c_s_d0) + ((p_int_e_bln_c_s_d0 * p_int_e_bln_c_s_d1) * (_loop_it_23 - offset_p_int_e_bln_c_s_d2))) + (p_int_e_bln_c_s_d0 * (3 - offset_p_int_e_bln_c_s_d1)))];
                                double _in_z_v_grad_w_0 = z_v_grad_w[(((((nproma * p_patch_nlev) * (p_patch_cells_edge_blk_at53 - 1)) + (nproma * (_loop_it_43 - 1))) + p_patch_cells_edge_idx_at52) - 1)];
                                double _in_z_v_grad_w_1 = z_v_grad_w[(((((nproma * p_patch_nlev) * (p_patch_cells_edge_blk_at55 - 1)) + (nproma * (_loop_it_43 - 1))) + p_patch_cells_edge_idx_at54) - 1)];
                                double _in_z_v_grad_w_2 = z_v_grad_w[(((((nproma * p_patch_nlev) * (p_patch_cells_edge_blk_at57 - 1)) + (nproma * (_loop_it_43 - 1))) + p_patch_cells_edge_idx_at56) - 1)];
                                double _out_p_diag_ddt_w_adv_pc;

                                ///////////////////
                                // Tasklet code (t_0)
                                _out_p_diag_ddt_w_adv_pc = (((_in_p_diag_ddt_w_adv_pc_0 + (_in_p_int_e_bln_c_s_0 * _in_z_v_grad_w_0)) + (_in_p_int_e_bln_c_s_1 * _in_z_v_grad_w_1)) + (_in_p_int_e_bln_c_s_2 * _in_z_v_grad_w_2));
                                ///////////////////

                                p_diag_ddt_w_adv_pc[((((_loop_it_44 - offset_p_diag_ddt_w_adv_pc_d0) + (((p_diag_ddt_w_adv_pc_d0 * p_diag_ddt_w_adv_pc_d1) * p_diag_ddt_w_adv_pc_d2) * (ntnd - offset_p_diag_ddt_w_adv_pc_d3))) + ((p_diag_ddt_w_adv_pc_d0 * p_diag_ddt_w_adv_pc_d1) * (_loop_it_23 - offset_p_diag_ddt_w_adv_pc_d2))) + (p_diag_ddt_w_adv_pc_d0 * (_loop_it_43 - offset_p_diag_ddt_w_adv_pc_d1)))] = _out_p_diag_ddt_w_adv_pc;
                            }

                        }

                    }



                }



                if_cond_317 = scal_lextra_diffu;


                if (if_cond_317) {

                    loopend_319 = (nlev - 3);

                    loopbegin_321 = max(3, (nrdmax_jg - 2));


                    for (_loop_it_45 = loopbegin_321; (_loop_it_45 < (loopend_319 + 1)); _loop_it_45 = (_loop_it_45 + 1)) {

                        {

                            {
                                bool _in_levmask_0 = levmask[((_loop_it_23 + (p_patch_nblks_c * (_loop_it_45 - 1))) - 1)];
                                int64_t _out_if_cond_324;

                                ///////////////////
                                // Tasklet code (t_325)
                                _out_if_cond_324 = _in_levmask_0;
                                ///////////////////

                                if_cond_324 = _out_if_cond_324;
                            }

                        }

                        if (if_cond_324) {

                            for (_loop_it_46 = i_startidx_2; (_loop_it_46 < (i_endidx_2 + 1)); _loop_it_46 = (_loop_it_46 + 1)) {

                                {

                                    {
                                        bool _in_cfl_clipping_0 = cfl_clipping[((_loop_it_46 + (nproma * (_loop_it_45 - 1))) - 1)];
                                        bool _in_p_patch_cells_decomp_info_owner_mask_0 = p_patch_cells_decomp_info_owner_mask[((_loop_it_46 - offset_p_patch_cells_decomp_info_owner_mask_d0) + (p_patch_cells_decomp_info_owner_mask_d0 * (_loop_it_23 - offset_p_patch_cells_decomp_info_owner_mask_d1)))];
                                        int64_t _out_if_cond_329;

                                        ///////////////////
                                        // Tasklet code (t_330)
                                        _out_if_cond_329 = (_in_cfl_clipping_0 && _in_p_patch_cells_decomp_info_owner_mask_0);
                                        ///////////////////

                                        if_cond_329 = _out_if_cond_329;
                                    }

                                }

                                if (if_cond_329) {
                                    {

                                        {
                                            double _in_p_metrics_ddqz_z_half_0 = p_metrics_ddqz_z_half[(((_loop_it_46 - offset_p_metrics_ddqz_z_half_d0) + ((p_metrics_ddqz_z_half_d0 * p_metrics_ddqz_z_half_d1) * (_loop_it_23 - offset_p_metrics_ddqz_z_half_d2))) + (p_metrics_ddqz_z_half_d0 * (_loop_it_45 - offset_p_metrics_ddqz_z_half_d1)))];
                                            double _in_z_w_con_c_0 = z_w_con_c[((_loop_it_46 + (nproma * (_loop_it_45 - 1))) - 1)];
                                            double _in_cfl_w_limit = cfl_w_limit;
                                            double _in_dtime = dtime;
                                            double _in_scalfac_exdiff = scalfac_exdiff;
                                            double _out_difcoef;

                                            ///////////////////
                                            // Tasklet code (t_333)
                                            _out_difcoef = (_in_scalfac_exdiff * min((0.85 - (_in_cfl_w_limit * _in_dtime)), (((abs(_in_z_w_con_c_0) * _in_dtime) / _in_p_metrics_ddqz_z_half_0) - (_in_cfl_w_limit * _in_dtime))));
                                            ///////////////////

                                            difcoef = _out_difcoef;
                                        }

                                    }
                                    p_patch_cells_neighbor_idx_at58 = p_patch_cells_neighbor_idx[(((_loop_it_46 - offset_p_patch_cells_neighbor_idx_d0) + ((p_patch_cells_neighbor_idx_d0 * p_patch_cells_neighbor_idx_d1) * (1 - offset_p_patch_cells_neighbor_idx_d2))) + (p_patch_cells_neighbor_idx_d0 * (_loop_it_23 - offset_p_patch_cells_neighbor_idx_d1)))];

                                    p_patch_cells_neighbor_blk_at59 = p_patch_cells_neighbor_blk[(((_loop_it_46 - offset_p_patch_cells_neighbor_blk_d0) + ((p_patch_cells_neighbor_blk_d0 * p_patch_cells_neighbor_blk_d1) * (1 - offset_p_patch_cells_neighbor_blk_d2))) + (p_patch_cells_neighbor_blk_d0 * (_loop_it_23 - offset_p_patch_cells_neighbor_blk_d1)))];

                                    p_patch_cells_neighbor_idx_at60 = p_patch_cells_neighbor_idx[(((_loop_it_46 - offset_p_patch_cells_neighbor_idx_d0) + ((p_patch_cells_neighbor_idx_d0 * p_patch_cells_neighbor_idx_d1) * (2 - offset_p_patch_cells_neighbor_idx_d2))) + (p_patch_cells_neighbor_idx_d0 * (_loop_it_23 - offset_p_patch_cells_neighbor_idx_d1)))];

                                    p_patch_cells_neighbor_blk_at61 = p_patch_cells_neighbor_blk[(((_loop_it_46 - offset_p_patch_cells_neighbor_blk_d0) + ((p_patch_cells_neighbor_blk_d0 * p_patch_cells_neighbor_blk_d1) * (2 - offset_p_patch_cells_neighbor_blk_d2))) + (p_patch_cells_neighbor_blk_d0 * (_loop_it_23 - offset_p_patch_cells_neighbor_blk_d1)))];

                                    p_patch_cells_neighbor_idx_at62 = p_patch_cells_neighbor_idx[(((_loop_it_46 - offset_p_patch_cells_neighbor_idx_d0) + ((p_patch_cells_neighbor_idx_d0 * p_patch_cells_neighbor_idx_d1) * (3 - offset_p_patch_cells_neighbor_idx_d2))) + (p_patch_cells_neighbor_idx_d0 * (_loop_it_23 - offset_p_patch_cells_neighbor_idx_d1)))];

                                    p_patch_cells_neighbor_blk_at63 = p_patch_cells_neighbor_blk[(((_loop_it_46 - offset_p_patch_cells_neighbor_blk_d0) + ((p_patch_cells_neighbor_blk_d0 * p_patch_cells_neighbor_blk_d1) * (3 - offset_p_patch_cells_neighbor_blk_d2))) + (p_patch_cells_neighbor_blk_d0 * (_loop_it_23 - offset_p_patch_cells_neighbor_blk_d1)))];
                                    {

                                        {
                                            double _in_p_diag_ddt_w_adv_pc_0 = p_diag_ddt_w_adv_pc[((((_loop_it_46 - offset_p_diag_ddt_w_adv_pc_d0) + (((p_diag_ddt_w_adv_pc_d0 * p_diag_ddt_w_adv_pc_d1) * p_diag_ddt_w_adv_pc_d2) * (ntnd - offset_p_diag_ddt_w_adv_pc_d3))) + ((p_diag_ddt_w_adv_pc_d0 * p_diag_ddt_w_adv_pc_d1) * (_loop_it_23 - offset_p_diag_ddt_w_adv_pc_d2))) + (p_diag_ddt_w_adv_pc_d0 * (_loop_it_45 - offset_p_diag_ddt_w_adv_pc_d1)))];
                                            double _in_p_int_geofac_n2s_0 = p_int_geofac_n2s[(((_loop_it_46 - offset_p_int_geofac_n2s_d0) + ((p_int_geofac_n2s_d0 * p_int_geofac_n2s_d1) * (_loop_it_23 - offset_p_int_geofac_n2s_d2))) + (p_int_geofac_n2s_d0 * (1 - offset_p_int_geofac_n2s_d1)))];
                                            double _in_p_int_geofac_n2s_1 = p_int_geofac_n2s[(((_loop_it_46 - offset_p_int_geofac_n2s_d0) + ((p_int_geofac_n2s_d0 * p_int_geofac_n2s_d1) * (_loop_it_23 - offset_p_int_geofac_n2s_d2))) + (p_int_geofac_n2s_d0 * (2 - offset_p_int_geofac_n2s_d1)))];
                                            double _in_p_int_geofac_n2s_2 = p_int_geofac_n2s[(((_loop_it_46 - offset_p_int_geofac_n2s_d0) + ((p_int_geofac_n2s_d0 * p_int_geofac_n2s_d1) * (_loop_it_23 - offset_p_int_geofac_n2s_d2))) + (p_int_geofac_n2s_d0 * (3 - offset_p_int_geofac_n2s_d1)))];
                                            double _in_p_int_geofac_n2s_3 = p_int_geofac_n2s[(((_loop_it_46 - offset_p_int_geofac_n2s_d0) + ((p_int_geofac_n2s_d0 * p_int_geofac_n2s_d1) * (_loop_it_23 - offset_p_int_geofac_n2s_d2))) + (p_int_geofac_n2s_d0 * (4 - offset_p_int_geofac_n2s_d1)))];
                                            double _in_p_patch_cells_area_0 = p_patch_cells_area[((_loop_it_46 - offset_p_patch_cells_area_d0) + (p_patch_cells_area_d0 * (_loop_it_23 - offset_p_patch_cells_area_d1)))];
                                            double _in_p_prog_w_0 = p_prog_w[(((_loop_it_46 - offset_p_prog_w_d0) + ((p_prog_w_d0 * p_prog_w_d1) * (_loop_it_23 - offset_p_prog_w_d2))) + (p_prog_w_d0 * (_loop_it_45 - offset_p_prog_w_d1)))];
                                            double _in_p_prog_w_1 = p_prog_w[((((- offset_p_prog_w_d0) + p_patch_cells_neighbor_idx_at58) + ((p_prog_w_d0 * p_prog_w_d1) * ((- offset_p_prog_w_d2) + p_patch_cells_neighbor_blk_at59))) + (p_prog_w_d0 * (_loop_it_45 - offset_p_prog_w_d1)))];
                                            double _in_p_prog_w_2 = p_prog_w[((((- offset_p_prog_w_d0) + p_patch_cells_neighbor_idx_at60) + ((p_prog_w_d0 * p_prog_w_d1) * ((- offset_p_prog_w_d2) + p_patch_cells_neighbor_blk_at61))) + (p_prog_w_d0 * (_loop_it_45 - offset_p_prog_w_d1)))];
                                            double _in_p_prog_w_3 = p_prog_w[((((- offset_p_prog_w_d0) + p_patch_cells_neighbor_idx_at62) + ((p_prog_w_d0 * p_prog_w_d1) * ((- offset_p_prog_w_d2) + p_patch_cells_neighbor_blk_at63))) + (p_prog_w_d0 * (_loop_it_45 - offset_p_prog_w_d1)))];
                                            double _in_difcoef = difcoef;
                                            double _out_p_diag_ddt_w_adv_pc;

                                            ///////////////////
                                            // Tasklet code (t_340)
                                            _out_p_diag_ddt_w_adv_pc = (_in_p_diag_ddt_w_adv_pc_0 + ((_in_difcoef * _in_p_patch_cells_area_0) * ((((_in_p_prog_w_0 * _in_p_int_geofac_n2s_0) + (_in_p_prog_w_1 * _in_p_int_geofac_n2s_1)) + (_in_p_prog_w_2 * _in_p_int_geofac_n2s_2)) + (_in_p_prog_w_3 * _in_p_int_geofac_n2s_3))));
                                            ///////////////////

                                            p_diag_ddt_w_adv_pc[((((_loop_it_46 - offset_p_diag_ddt_w_adv_pc_d0) + (((p_diag_ddt_w_adv_pc_d0 * p_diag_ddt_w_adv_pc_d1) * p_diag_ddt_w_adv_pc_d2) * (ntnd - offset_p_diag_ddt_w_adv_pc_d3))) + ((p_diag_ddt_w_adv_pc_d0 * p_diag_ddt_w_adv_pc_d1) * (_loop_it_23 - offset_p_diag_ddt_w_adv_pc_d2))) + (p_diag_ddt_w_adv_pc_d0 * (_loop_it_45 - offset_p_diag_ddt_w_adv_pc_d1)))] = _out_p_diag_ddt_w_adv_pc;
                                        }

                                    }
                                }


                            }


                        }


                    }


                }

            }

        }


    }



    loopend_341 = (nlev - 3);

    loopbegin_343 = max(3, (nrdmax_jg - 2));


    for (_loop_it_47 = loopbegin_343; (_loop_it_47 < (loopend_341 + 1)); _loop_it_47 = (_loop_it_47 + 1)) {
        {

            {
                bool _out_levelmask;

                ///////////////////
                // Tasklet code (t_346)
                _out_levelmask = false;
                ///////////////////

                levelmask[(_loop_it_47 - 1)] = _out_levelmask;
            }

        }

        for (_loop_it_48 = i_startblk; (_loop_it_48 < (i_endblk + 1)); _loop_it_48 = (_loop_it_48 + 1)) {
            {

                {
                    bool _in_levelmask_0 = levelmask[(_loop_it_47 - 1)];
                    bool _in_levmask_0 = levmask[((_loop_it_48 + (p_patch_nblks_c * (_loop_it_47 - 1))) - 1)];
                    bool _out_levelmask;

                    ///////////////////
                    // Tasklet code (t_0)
                    _out_levelmask = (_in_levelmask_0 || _in_levmask_0);
                    ///////////////////

                    levelmask[(_loop_it_47 - 1)] = _out_levelmask;
                }

            }

        }



    }


    {

        {
            int _out;

            ///////////////////
            // Tasklet code (set_rl_start)
            _out = 10;
            ///////////////////

            rl_start = _out;
        }
        {
            int _out;

            ///////////////////
            // Tasklet code (set_rl_end)
            _out = -8;
            ///////////////////

            rl_end = _out;
        }

    }
    i_startblk = p_patch_edges_start_block[(10 - offset_p_patch_edges_start_block_d0)];

    i_endblk = p_patch_edges_end_block[2];


    for (_loop_it_49 = i_startblk; (_loop_it_49 < (i_endblk + 1)); _loop_it_49 = (_loop_it_49 + 1)) {

        f__assoc_scalar_18 = 10;
        {

            {
                int _out;

                ///////////////////
                // Tasklet code (set_f__assoc_scalar_19)
                _out = -8;
                ///////////////////

                f__assoc_scalar_19 = _out;
            }
            {
                int _in_p_patch_edges_start_index_0 = p_patch_edges_start_index[(f__assoc_scalar_18 - offset_p_patch_edges_start_index_d0)];
                int _out_i_startidx_in_inl3;

                ///////////////////
                // Tasklet code (t_354)
                _out_i_startidx_in_inl3 = _in_p_patch_edges_start_index_0;
                ///////////////////

                i_startidx_in_inl3 = _out_i_startidx_in_inl3;
            }

        }
        irl_end_inl3 = f__assoc_scalar_19;
        {

            {
                int _in_p_patch_edges_end_index_0 = p_patch_edges_end_index[(irl_end_inl3 + 10)];
                int _out_i_endidx_in_inl3;

                ///////////////////
                // Tasklet code (t_356)
                _out_i_endidx_in_inl3 = _in_p_patch_edges_end_index_0;
                ///////////////////

                i_endidx_in_inl3 = _out_i_endidx_in_inl3;
            }

        }
        i_startidx = ((_loop_it_49 != i_startblk) ? 1 : ((i_startidx_in_inl3 < 1) ? 1 : i_startidx_in_inl3));

        i_endidx = ((_loop_it_49 != i_endblk) ? nproma : i_endidx_in_inl3);

        if_cond_359 = (ldeepatmo != true);


        if (if_cond_359) {

            for (_loop_it_50 = 1; (_loop_it_50 < (nlev + 1)); _loop_it_50 = (_loop_it_50 + 1)) {

                for (_loop_it_51 = i_startidx; (_loop_it_51 < (i_endidx + 1)); _loop_it_51 = (_loop_it_51 + 1)) {

                    p_patch_edges_cell_idx_at64 = p_patch_edges_cell_idx[(((_loop_it_51 - offset_p_patch_edges_cell_idx_d0) + ((p_patch_edges_cell_idx_d0 * p_patch_edges_cell_idx_d1) * (2 - offset_p_patch_edges_cell_idx_d2))) + (p_patch_edges_cell_idx_d0 * (_loop_it_49 - offset_p_patch_edges_cell_idx_d1)))];

                    p_patch_edges_cell_blk_at65 = p_patch_edges_cell_blk[(((_loop_it_51 - offset_p_patch_edges_cell_blk_d0) + ((p_patch_edges_cell_blk_d0 * p_patch_edges_cell_blk_d1) * (2 - offset_p_patch_edges_cell_blk_d2))) + (p_patch_edges_cell_blk_d0 * (_loop_it_49 - offset_p_patch_edges_cell_blk_d1)))];

                    p_patch_edges_cell_idx_at66 = p_patch_edges_cell_idx[(((_loop_it_51 - offset_p_patch_edges_cell_idx_d0) + ((p_patch_edges_cell_idx_d0 * p_patch_edges_cell_idx_d1) * (1 - offset_p_patch_edges_cell_idx_d2))) + (p_patch_edges_cell_idx_d0 * (_loop_it_49 - offset_p_patch_edges_cell_idx_d1)))];

                    p_patch_edges_cell_blk_at67 = p_patch_edges_cell_blk[(((_loop_it_51 - offset_p_patch_edges_cell_blk_d0) + ((p_patch_edges_cell_blk_d0 * p_patch_edges_cell_blk_d1) * (1 - offset_p_patch_edges_cell_blk_d2))) + (p_patch_edges_cell_blk_d0 * (_loop_it_49 - offset_p_patch_edges_cell_blk_d1)))];

                    p_patch_edges_vertex_idx_at68 = p_patch_edges_vertex_idx[(((_loop_it_51 - offset_p_patch_edges_vertex_idx_d0) + ((p_patch_edges_vertex_idx_d0 * p_patch_edges_vertex_idx_d1) * (1 - offset_p_patch_edges_vertex_idx_d2))) + (p_patch_edges_vertex_idx_d0 * (_loop_it_49 - offset_p_patch_edges_vertex_idx_d1)))];

                    p_patch_edges_vertex_blk_at69 = p_patch_edges_vertex_blk[(((_loop_it_51 - offset_p_patch_edges_vertex_blk_d0) + ((p_patch_edges_vertex_blk_d0 * p_patch_edges_vertex_blk_d1) * (1 - offset_p_patch_edges_vertex_blk_d2))) + (p_patch_edges_vertex_blk_d0 * (_loop_it_49 - offset_p_patch_edges_vertex_blk_d1)))];

                    p_patch_edges_vertex_idx_at70 = p_patch_edges_vertex_idx[(((_loop_it_51 - offset_p_patch_edges_vertex_idx_d0) + ((p_patch_edges_vertex_idx_d0 * p_patch_edges_vertex_idx_d1) * (2 - offset_p_patch_edges_vertex_idx_d2))) + (p_patch_edges_vertex_idx_d0 * (_loop_it_49 - offset_p_patch_edges_vertex_idx_d1)))];

                    p_patch_edges_vertex_blk_at71 = p_patch_edges_vertex_blk[(((_loop_it_51 - offset_p_patch_edges_vertex_blk_d0) + ((p_patch_edges_vertex_blk_d0 * p_patch_edges_vertex_blk_d1) * (2 - offset_p_patch_edges_vertex_blk_d2))) + (p_patch_edges_vertex_blk_d0 * (_loop_it_49 - offset_p_patch_edges_vertex_blk_d1)))];

                    {

                        {
                            double _in_p_diag_vn_ie_0 = p_diag_vn_ie[(((_loop_it_51 - offset_p_diag_vn_ie_d0) + ((p_diag_vn_ie_d0 * p_diag_vn_ie_d1) * (_loop_it_49 - offset_p_diag_vn_ie_d2))) + (p_diag_vn_ie_d0 * (_loop_it_50 - offset_p_diag_vn_ie_d1)))];
                            double _in_p_diag_vn_ie_1 = p_diag_vn_ie[(((_loop_it_51 - offset_p_diag_vn_ie_d0) + ((p_diag_vn_ie_d0 * p_diag_vn_ie_d1) * (_loop_it_49 - offset_p_diag_vn_ie_d2))) + (p_diag_vn_ie_d0 * ((_loop_it_50 - offset_p_diag_vn_ie_d1) + 1)))];
                            double _in_p_diag_vt_0 = p_diag_vt[(((_loop_it_51 - offset_p_diag_vt_d0) + ((p_diag_vt_d0 * p_diag_vt_d1) * (_loop_it_49 - offset_p_diag_vt_d2))) + (p_diag_vt_d0 * (_loop_it_50 - offset_p_diag_vt_d1)))];
                            double _in_p_int_c_lin_e_0 = p_int_c_lin_e[(((_loop_it_51 - offset_p_int_c_lin_e_d0) + ((p_int_c_lin_e_d0 * p_int_c_lin_e_d1) * (_loop_it_49 - offset_p_int_c_lin_e_d2))) + (p_int_c_lin_e_d0 * (1 - offset_p_int_c_lin_e_d1)))];
                            double _in_p_int_c_lin_e_1 = p_int_c_lin_e[(((_loop_it_51 - offset_p_int_c_lin_e_d0) + ((p_int_c_lin_e_d0 * p_int_c_lin_e_d1) * (_loop_it_49 - offset_p_int_c_lin_e_d2))) + (p_int_c_lin_e_d0 * (2 - offset_p_int_c_lin_e_d1)))];
                            double _in_p_metrics_coeff_gradekin_0 = p_metrics_coeff_gradekin[(((_loop_it_51 - offset_p_metrics_coeff_gradekin_d0) + ((p_metrics_coeff_gradekin_d0 * p_metrics_coeff_gradekin_d1) * (_loop_it_49 - offset_p_metrics_coeff_gradekin_d2))) + (p_metrics_coeff_gradekin_d0 * (1 - offset_p_metrics_coeff_gradekin_d1)))];
                            double _in_p_metrics_coeff_gradekin_1 = p_metrics_coeff_gradekin[(((_loop_it_51 - offset_p_metrics_coeff_gradekin_d0) + ((p_metrics_coeff_gradekin_d0 * p_metrics_coeff_gradekin_d1) * (_loop_it_49 - offset_p_metrics_coeff_gradekin_d2))) + (p_metrics_coeff_gradekin_d0 * (2 - offset_p_metrics_coeff_gradekin_d1)))];
                            double _in_p_metrics_coeff_gradekin_2 = p_metrics_coeff_gradekin[(((_loop_it_51 - offset_p_metrics_coeff_gradekin_d0) + ((p_metrics_coeff_gradekin_d0 * p_metrics_coeff_gradekin_d1) * (_loop_it_49 - offset_p_metrics_coeff_gradekin_d2))) + (p_metrics_coeff_gradekin_d0 * (2 - offset_p_metrics_coeff_gradekin_d1)))];
                            double _in_p_metrics_coeff_gradekin_3 = p_metrics_coeff_gradekin[(((_loop_it_51 - offset_p_metrics_coeff_gradekin_d0) + ((p_metrics_coeff_gradekin_d0 * p_metrics_coeff_gradekin_d1) * (_loop_it_49 - offset_p_metrics_coeff_gradekin_d2))) + (p_metrics_coeff_gradekin_d0 * (1 - offset_p_metrics_coeff_gradekin_d1)))];
                            double _in_p_metrics_ddqz_z_full_e_0 = p_metrics_ddqz_z_full_e[(((_loop_it_51 - offset_p_metrics_ddqz_z_full_e_d0) + ((p_metrics_ddqz_z_full_e_d0 * p_metrics_ddqz_z_full_e_d1) * (_loop_it_49 - offset_p_metrics_ddqz_z_full_e_d2))) + (p_metrics_ddqz_z_full_e_d0 * (_loop_it_50 - offset_p_metrics_ddqz_z_full_e_d1)))];
                            double _in_p_patch_edges_f_e_0 = p_patch_edges_f_e[((_loop_it_51 - offset_p_patch_edges_f_e_d0) + (p_patch_edges_f_e_d0 * (_loop_it_49 - offset_p_patch_edges_f_e_d1)))];
                            double _in_z_ekinh_0 = z_ekinh[(((((nproma * p_patch_nlev) * (p_patch_edges_cell_blk_at65 - 1)) + (nproma * (_loop_it_50 - 1))) + p_patch_edges_cell_idx_at64) - 1)];
                            double _in_z_ekinh_1 = z_ekinh[(((((nproma * p_patch_nlev) * (p_patch_edges_cell_blk_at67 - 1)) + (nproma * (_loop_it_50 - 1))) + p_patch_edges_cell_idx_at66) - 1)];
                            double _in_z_kin_hor_e_0 = z_kin_hor_e[(((_loop_it_51 + ((z_kin_hor_e_d0 * z_kin_hor_e_d1) * (_loop_it_49 - 1))) + (z_kin_hor_e_d0 * (_loop_it_50 - 1))) - 1)];
                            double _in_z_w_con_c_full_0 = z_w_con_c_full[(((((nproma * p_patch_nlev) * (p_patch_edges_cell_blk_at67 - 1)) + (nproma * (_loop_it_50 - 1))) + p_patch_edges_cell_idx_at66) - 1)];
                            double _in_z_w_con_c_full_1 = z_w_con_c_full[(((((nproma * p_patch_nlev) * (p_patch_edges_cell_blk_at65 - 1)) + (nproma * (_loop_it_50 - 1))) + p_patch_edges_cell_idx_at64) - 1)];
                            double _in_zeta_0 = zeta[(((((nproma * p_patch_nlev) * (p_patch_edges_vertex_blk_at69 - 1)) + (nproma * (_loop_it_50 - 1))) + p_patch_edges_vertex_idx_at68) - 1)];
                            double _in_zeta_1 = zeta[(((((nproma * p_patch_nlev) * (p_patch_edges_vertex_blk_at71 - 1)) + (nproma * (_loop_it_50 - 1))) + p_patch_edges_vertex_idx_at70) - 1)];
                            double _out_p_diag_ddt_vn_apc_pc;

                            ///////////////////
                            // Tasklet code (t_0)
                            _out_p_diag_ddt_vn_apc_pc = (- (((((_in_z_kin_hor_e_0 * (_in_p_metrics_coeff_gradekin_0 - _in_p_metrics_coeff_gradekin_1)) + (_in_p_metrics_coeff_gradekin_2 * _in_z_ekinh_0)) - (_in_p_metrics_coeff_gradekin_3 * _in_z_ekinh_1)) + (_in_p_diag_vt_0 * (_in_p_patch_edges_f_e_0 + ((_in_zeta_0 + _in_zeta_1) * 0.5)))) + ((((_in_p_int_c_lin_e_0 * _in_z_w_con_c_full_0) + (_in_p_int_c_lin_e_1 * _in_z_w_con_c_full_1)) * (_in_p_diag_vn_ie_0 - _in_p_diag_vn_ie_1)) / _in_p_metrics_ddqz_z_full_e_0)));
                            ///////////////////

                            p_diag_ddt_vn_apc_pc[((((_loop_it_51 - offset_p_diag_ddt_vn_apc_pc_d0) + (((p_diag_ddt_vn_apc_pc_d0 * p_diag_ddt_vn_apc_pc_d1) * p_diag_ddt_vn_apc_pc_d2) * (ntnd - offset_p_diag_ddt_vn_apc_pc_d3))) + ((p_diag_ddt_vn_apc_pc_d0 * p_diag_ddt_vn_apc_pc_d1) * (_loop_it_49 - offset_p_diag_ddt_vn_apc_pc_d2))) + (p_diag_ddt_vn_apc_pc_d0 * (_loop_it_50 - offset_p_diag_ddt_vn_apc_pc_d1)))] = _out_p_diag_ddt_vn_apc_pc;
                        }

                    }

                }



            }



            if_cond_373 = (scal_p_diag_ddt_vn_adv_is_associated || scal_p_diag_ddt_vn_cor_is_associated);


            if (if_cond_373) {

                for (_loop_it_52 = 1; (_loop_it_52 < (nlev + 1)); _loop_it_52 = (_loop_it_52 + 1)) {

                    for (_loop_it_53 = i_startidx; (_loop_it_53 < (i_endidx + 1)); _loop_it_53 = (_loop_it_53 + 1)) {
                        {

                            {
                                double _in_p_diag_vt_0 = p_diag_vt[(((_loop_it_53 - offset_p_diag_vt_d0) + ((p_diag_vt_d0 * p_diag_vt_d1) * (_loop_it_49 - offset_p_diag_vt_d2))) + (p_diag_vt_d0 * (_loop_it_52 - offset_p_diag_vt_d1)))];
                                double _in_p_patch_edges_f_e_0 = p_patch_edges_f_e[((_loop_it_53 - offset_p_patch_edges_f_e_d0) + (p_patch_edges_f_e_d0 * (_loop_it_49 - offset_p_patch_edges_f_e_d1)))];
                                double _out_p_diag_ddt_vn_cor_pc;

                                ///////////////////
                                // Tasklet code (t_0)
                                _out_p_diag_ddt_vn_cor_pc = (- (_in_p_diag_vt_0 * _in_p_patch_edges_f_e_0));
                                ///////////////////

                                p_diag_ddt_vn_cor_pc[((((_loop_it_53 - offset_p_diag_ddt_vn_cor_pc_d0) + (((p_diag_ddt_vn_cor_pc_d0 * p_diag_ddt_vn_cor_pc_d1) * p_diag_ddt_vn_cor_pc_d2) * (ntnd - offset_p_diag_ddt_vn_cor_pc_d3))) + ((p_diag_ddt_vn_cor_pc_d0 * p_diag_ddt_vn_cor_pc_d1) * (_loop_it_49 - offset_p_diag_ddt_vn_cor_pc_d2))) + (p_diag_ddt_vn_cor_pc_d0 * (_loop_it_52 - offset_p_diag_ddt_vn_cor_pc_d1)))] = _out_p_diag_ddt_vn_cor_pc;
                            }

                        }

                    }



                }


            }

        } else {

            for (_loop_it_54 = 1; (_loop_it_54 < (nlev + 1)); _loop_it_54 = (_loop_it_54 + 1)) {

                for (_loop_it_55 = i_startidx; (_loop_it_55 < (i_endidx + 1)); _loop_it_55 = (_loop_it_55 + 1)) {

                    p_patch_edges_cell_idx_at72 = p_patch_edges_cell_idx[(((_loop_it_55 - offset_p_patch_edges_cell_idx_d0) + ((p_patch_edges_cell_idx_d0 * p_patch_edges_cell_idx_d1) * (2 - offset_p_patch_edges_cell_idx_d2))) + (p_patch_edges_cell_idx_d0 * (_loop_it_49 - offset_p_patch_edges_cell_idx_d1)))];

                    p_patch_edges_cell_blk_at73 = p_patch_edges_cell_blk[(((_loop_it_55 - offset_p_patch_edges_cell_blk_d0) + ((p_patch_edges_cell_blk_d0 * p_patch_edges_cell_blk_d1) * (2 - offset_p_patch_edges_cell_blk_d2))) + (p_patch_edges_cell_blk_d0 * (_loop_it_49 - offset_p_patch_edges_cell_blk_d1)))];

                    p_patch_edges_cell_idx_at74 = p_patch_edges_cell_idx[(((_loop_it_55 - offset_p_patch_edges_cell_idx_d0) + ((p_patch_edges_cell_idx_d0 * p_patch_edges_cell_idx_d1) * (1 - offset_p_patch_edges_cell_idx_d2))) + (p_patch_edges_cell_idx_d0 * (_loop_it_49 - offset_p_patch_edges_cell_idx_d1)))];

                    p_patch_edges_cell_blk_at75 = p_patch_edges_cell_blk[(((_loop_it_55 - offset_p_patch_edges_cell_blk_d0) + ((p_patch_edges_cell_blk_d0 * p_patch_edges_cell_blk_d1) * (1 - offset_p_patch_edges_cell_blk_d2))) + (p_patch_edges_cell_blk_d0 * (_loop_it_49 - offset_p_patch_edges_cell_blk_d1)))];

                    p_patch_edges_vertex_idx_at76 = p_patch_edges_vertex_idx[(((_loop_it_55 - offset_p_patch_edges_vertex_idx_d0) + ((p_patch_edges_vertex_idx_d0 * p_patch_edges_vertex_idx_d1) * (1 - offset_p_patch_edges_vertex_idx_d2))) + (p_patch_edges_vertex_idx_d0 * (_loop_it_49 - offset_p_patch_edges_vertex_idx_d1)))];

                    p_patch_edges_vertex_blk_at77 = p_patch_edges_vertex_blk[(((_loop_it_55 - offset_p_patch_edges_vertex_blk_d0) + ((p_patch_edges_vertex_blk_d0 * p_patch_edges_vertex_blk_d1) * (1 - offset_p_patch_edges_vertex_blk_d2))) + (p_patch_edges_vertex_blk_d0 * (_loop_it_49 - offset_p_patch_edges_vertex_blk_d1)))];

                    p_patch_edges_vertex_idx_at78 = p_patch_edges_vertex_idx[(((_loop_it_55 - offset_p_patch_edges_vertex_idx_d0) + ((p_patch_edges_vertex_idx_d0 * p_patch_edges_vertex_idx_d1) * (2 - offset_p_patch_edges_vertex_idx_d2))) + (p_patch_edges_vertex_idx_d0 * (_loop_it_49 - offset_p_patch_edges_vertex_idx_d1)))];

                    p_patch_edges_vertex_blk_at79 = p_patch_edges_vertex_blk[(((_loop_it_55 - offset_p_patch_edges_vertex_blk_d0) + ((p_patch_edges_vertex_blk_d0 * p_patch_edges_vertex_blk_d1) * (2 - offset_p_patch_edges_vertex_blk_d2))) + (p_patch_edges_vertex_blk_d0 * (_loop_it_49 - offset_p_patch_edges_vertex_blk_d1)))];

                    {

                        {
                            double _in_p_diag_vn_ie_0 = p_diag_vn_ie[(((_loop_it_55 - offset_p_diag_vn_ie_d0) + ((p_diag_vn_ie_d0 * p_diag_vn_ie_d1) * (_loop_it_49 - offset_p_diag_vn_ie_d2))) + (p_diag_vn_ie_d0 * (_loop_it_54 - offset_p_diag_vn_ie_d1)))];
                            double _in_p_diag_vn_ie_1 = p_diag_vn_ie[(((_loop_it_55 - offset_p_diag_vn_ie_d0) + ((p_diag_vn_ie_d0 * p_diag_vn_ie_d1) * (_loop_it_49 - offset_p_diag_vn_ie_d2))) + (p_diag_vn_ie_d0 * ((_loop_it_54 - offset_p_diag_vn_ie_d1) + 1)))];
                            double _in_p_diag_vt_0 = p_diag_vt[(((_loop_it_55 - offset_p_diag_vt_d0) + ((p_diag_vt_d0 * p_diag_vt_d1) * (_loop_it_49 - offset_p_diag_vt_d2))) + (p_diag_vt_d0 * (_loop_it_54 - offset_p_diag_vt_d1)))];
                            double _in_p_int_c_lin_e_0 = p_int_c_lin_e[(((_loop_it_55 - offset_p_int_c_lin_e_d0) + ((p_int_c_lin_e_d0 * p_int_c_lin_e_d1) * (_loop_it_49 - offset_p_int_c_lin_e_d2))) + (p_int_c_lin_e_d0 * (1 - offset_p_int_c_lin_e_d1)))];
                            double _in_p_int_c_lin_e_1 = p_int_c_lin_e[(((_loop_it_55 - offset_p_int_c_lin_e_d0) + ((p_int_c_lin_e_d0 * p_int_c_lin_e_d1) * (_loop_it_49 - offset_p_int_c_lin_e_d2))) + (p_int_c_lin_e_d0 * (2 - offset_p_int_c_lin_e_d1)))];
                            double _in_p_metrics_coeff_gradekin_0 = p_metrics_coeff_gradekin[(((_loop_it_55 - offset_p_metrics_coeff_gradekin_d0) + ((p_metrics_coeff_gradekin_d0 * p_metrics_coeff_gradekin_d1) * (_loop_it_49 - offset_p_metrics_coeff_gradekin_d2))) + (p_metrics_coeff_gradekin_d0 * (1 - offset_p_metrics_coeff_gradekin_d1)))];
                            double _in_p_metrics_coeff_gradekin_1 = p_metrics_coeff_gradekin[(((_loop_it_55 - offset_p_metrics_coeff_gradekin_d0) + ((p_metrics_coeff_gradekin_d0 * p_metrics_coeff_gradekin_d1) * (_loop_it_49 - offset_p_metrics_coeff_gradekin_d2))) + (p_metrics_coeff_gradekin_d0 * (2 - offset_p_metrics_coeff_gradekin_d1)))];
                            double _in_p_metrics_coeff_gradekin_2 = p_metrics_coeff_gradekin[(((_loop_it_55 - offset_p_metrics_coeff_gradekin_d0) + ((p_metrics_coeff_gradekin_d0 * p_metrics_coeff_gradekin_d1) * (_loop_it_49 - offset_p_metrics_coeff_gradekin_d2))) + (p_metrics_coeff_gradekin_d0 * (2 - offset_p_metrics_coeff_gradekin_d1)))];
                            double _in_p_metrics_coeff_gradekin_3 = p_metrics_coeff_gradekin[(((_loop_it_55 - offset_p_metrics_coeff_gradekin_d0) + ((p_metrics_coeff_gradekin_d0 * p_metrics_coeff_gradekin_d1) * (_loop_it_49 - offset_p_metrics_coeff_gradekin_d2))) + (p_metrics_coeff_gradekin_d0 * (1 - offset_p_metrics_coeff_gradekin_d1)))];
                            double _in_p_metrics_ddqz_z_full_e_0 = p_metrics_ddqz_z_full_e[(((_loop_it_55 - offset_p_metrics_ddqz_z_full_e_d0) + ((p_metrics_ddqz_z_full_e_d0 * p_metrics_ddqz_z_full_e_d1) * (_loop_it_49 - offset_p_metrics_ddqz_z_full_e_d2))) + (p_metrics_ddqz_z_full_e_d0 * (_loop_it_54 - offset_p_metrics_ddqz_z_full_e_d1)))];
                            double _in_p_metrics_deepatmo_gradh_mc_0 = p_metrics_deepatmo_gradh_mc[(_loop_it_54 - offset_p_metrics_deepatmo_gradh_mc_d0)];
                            double _in_p_metrics_deepatmo_gradh_mc_1 = p_metrics_deepatmo_gradh_mc[(_loop_it_54 - offset_p_metrics_deepatmo_gradh_mc_d0)];
                            double _in_p_metrics_deepatmo_invr_mc_0 = p_metrics_deepatmo_invr_mc[(_loop_it_54 - offset_p_metrics_deepatmo_invr_mc_d0)];
                            double _in_p_patch_edges_f_e_0 = p_patch_edges_f_e[((_loop_it_55 - offset_p_patch_edges_f_e_d0) + (p_patch_edges_f_e_d0 * (_loop_it_49 - offset_p_patch_edges_f_e_d1)))];
                            double _in_p_patch_edges_ft_e_0 = p_patch_edges_ft_e[((_loop_it_55 - offset_p_patch_edges_ft_e_d0) + (p_patch_edges_ft_e_d0 * (_loop_it_49 - offset_p_patch_edges_ft_e_d1)))];
                            double _in_p_prog_vn_0 = p_prog_vn[(((_loop_it_55 - offset_p_prog_vn_d0) + ((p_prog_vn_d0 * p_prog_vn_d1) * (_loop_it_49 - offset_p_prog_vn_d2))) + (p_prog_vn_d0 * (_loop_it_54 - offset_p_prog_vn_d1)))];
                            double _in_z_ekinh_0 = z_ekinh[(((((nproma * p_patch_nlev) * (p_patch_edges_cell_blk_at73 - 1)) + (nproma * (_loop_it_54 - 1))) + p_patch_edges_cell_idx_at72) - 1)];
                            double _in_z_ekinh_1 = z_ekinh[(((((nproma * p_patch_nlev) * (p_patch_edges_cell_blk_at75 - 1)) + (nproma * (_loop_it_54 - 1))) + p_patch_edges_cell_idx_at74) - 1)];
                            double _in_z_kin_hor_e_0 = z_kin_hor_e[(((_loop_it_55 + ((z_kin_hor_e_d0 * z_kin_hor_e_d1) * (_loop_it_49 - 1))) + (z_kin_hor_e_d0 * (_loop_it_54 - 1))) - 1)];
                            double _in_z_w_con_c_full_0 = z_w_con_c_full[(((((nproma * p_patch_nlev) * (p_patch_edges_cell_blk_at75 - 1)) + (nproma * (_loop_it_54 - 1))) + p_patch_edges_cell_idx_at74) - 1)];
                            double _in_z_w_con_c_full_1 = z_w_con_c_full[(((((nproma * p_patch_nlev) * (p_patch_edges_cell_blk_at73 - 1)) + (nproma * (_loop_it_54 - 1))) + p_patch_edges_cell_idx_at72) - 1)];
                            double _in_zeta_0 = zeta[(((((nproma * p_patch_nlev) * (p_patch_edges_vertex_blk_at77 - 1)) + (nproma * (_loop_it_54 - 1))) + p_patch_edges_vertex_idx_at76) - 1)];
                            double _in_zeta_1 = zeta[(((((nproma * p_patch_nlev) * (p_patch_edges_vertex_blk_at79 - 1)) + (nproma * (_loop_it_54 - 1))) + p_patch_edges_vertex_idx_at78) - 1)];
                            double _out_p_diag_ddt_vn_apc_pc;

                            ///////////////////
                            // Tasklet code (t_0)
                            _out_p_diag_ddt_vn_apc_pc = (- ((((((_in_z_kin_hor_e_0 * (_in_p_metrics_coeff_gradekin_0 - _in_p_metrics_coeff_gradekin_1)) + (_in_p_metrics_coeff_gradekin_2 * _in_z_ekinh_0)) - (_in_p_metrics_coeff_gradekin_3 * _in_z_ekinh_1)) * _in_p_metrics_deepatmo_gradh_mc_0) + (_in_p_diag_vt_0 * (_in_p_patch_edges_f_e_0 + (((_in_zeta_0 + _in_zeta_1) * 0.5) * _in_p_metrics_deepatmo_gradh_mc_1)))) + (((_in_p_int_c_lin_e_0 * _in_z_w_con_c_full_0) + (_in_p_int_c_lin_e_1 * _in_z_w_con_c_full_1)) * ((((_in_p_diag_vn_ie_0 - _in_p_diag_vn_ie_1) / _in_p_metrics_ddqz_z_full_e_0) + (_in_p_prog_vn_0 * _in_p_metrics_deepatmo_invr_mc_0)) - _in_p_patch_edges_ft_e_0))));
                            ///////////////////

                            p_diag_ddt_vn_apc_pc[((((_loop_it_55 - offset_p_diag_ddt_vn_apc_pc_d0) + (((p_diag_ddt_vn_apc_pc_d0 * p_diag_ddt_vn_apc_pc_d1) * p_diag_ddt_vn_apc_pc_d2) * (ntnd - offset_p_diag_ddt_vn_apc_pc_d3))) + ((p_diag_ddt_vn_apc_pc_d0 * p_diag_ddt_vn_apc_pc_d1) * (_loop_it_49 - offset_p_diag_ddt_vn_apc_pc_d2))) + (p_diag_ddt_vn_apc_pc_d0 * (_loop_it_54 - offset_p_diag_ddt_vn_apc_pc_d1)))] = _out_p_diag_ddt_vn_apc_pc;
                        }

                    }

                }



            }



            if_cond_389 = (scal_p_diag_ddt_vn_adv_is_associated || scal_p_diag_ddt_vn_cor_is_associated);


            if (if_cond_389) {

                for (_loop_it_56 = 1; (_loop_it_56 < (nlev + 1)); _loop_it_56 = (_loop_it_56 + 1)) {

                    for (_loop_it_57 = i_startidx; (_loop_it_57 < (i_endidx + 1)); _loop_it_57 = (_loop_it_57 + 1)) {

                        p_patch_edges_cell_idx_at80 = p_patch_edges_cell_idx[(((_loop_it_57 - offset_p_patch_edges_cell_idx_d0) + ((p_patch_edges_cell_idx_d0 * p_patch_edges_cell_idx_d1) * (1 - offset_p_patch_edges_cell_idx_d2))) + (p_patch_edges_cell_idx_d0 * (_loop_it_49 - offset_p_patch_edges_cell_idx_d1)))];

                        p_patch_edges_cell_blk_at81 = p_patch_edges_cell_blk[(((_loop_it_57 - offset_p_patch_edges_cell_blk_d0) + ((p_patch_edges_cell_blk_d0 * p_patch_edges_cell_blk_d1) * (1 - offset_p_patch_edges_cell_blk_d2))) + (p_patch_edges_cell_blk_d0 * (_loop_it_49 - offset_p_patch_edges_cell_blk_d1)))];

                        p_patch_edges_cell_idx_at82 = p_patch_edges_cell_idx[(((_loop_it_57 - offset_p_patch_edges_cell_idx_d0) + ((p_patch_edges_cell_idx_d0 * p_patch_edges_cell_idx_d1) * (2 - offset_p_patch_edges_cell_idx_d2))) + (p_patch_edges_cell_idx_d0 * (_loop_it_49 - offset_p_patch_edges_cell_idx_d1)))];

                        p_patch_edges_cell_blk_at83 = p_patch_edges_cell_blk[(((_loop_it_57 - offset_p_patch_edges_cell_blk_d0) + ((p_patch_edges_cell_blk_d0 * p_patch_edges_cell_blk_d1) * (2 - offset_p_patch_edges_cell_blk_d2))) + (p_patch_edges_cell_blk_d0 * (_loop_it_49 - offset_p_patch_edges_cell_blk_d1)))];

                        {

                            {
                                double _in_p_diag_vt_0 = p_diag_vt[(((_loop_it_57 - offset_p_diag_vt_d0) + ((p_diag_vt_d0 * p_diag_vt_d1) * (_loop_it_49 - offset_p_diag_vt_d2))) + (p_diag_vt_d0 * (_loop_it_56 - offset_p_diag_vt_d1)))];
                                double _in_p_int_c_lin_e_0 = p_int_c_lin_e[(((_loop_it_57 - offset_p_int_c_lin_e_d0) + ((p_int_c_lin_e_d0 * p_int_c_lin_e_d1) * (_loop_it_49 - offset_p_int_c_lin_e_d2))) + (p_int_c_lin_e_d0 * (1 - offset_p_int_c_lin_e_d1)))];
                                double _in_p_int_c_lin_e_1 = p_int_c_lin_e[(((_loop_it_57 - offset_p_int_c_lin_e_d0) + ((p_int_c_lin_e_d0 * p_int_c_lin_e_d1) * (_loop_it_49 - offset_p_int_c_lin_e_d2))) + (p_int_c_lin_e_d0 * (2 - offset_p_int_c_lin_e_d1)))];
                                double _in_p_patch_edges_f_e_0 = p_patch_edges_f_e[((_loop_it_57 - offset_p_patch_edges_f_e_d0) + (p_patch_edges_f_e_d0 * (_loop_it_49 - offset_p_patch_edges_f_e_d1)))];
                                double _in_p_patch_edges_ft_e_0 = p_patch_edges_ft_e[((_loop_it_57 - offset_p_patch_edges_ft_e_d0) + (p_patch_edges_ft_e_d0 * (_loop_it_49 - offset_p_patch_edges_ft_e_d1)))];
                                double _in_z_w_con_c_full_0 = z_w_con_c_full[(((((nproma * p_patch_nlev) * (p_patch_edges_cell_blk_at81 - 1)) + (nproma * (_loop_it_56 - 1))) + p_patch_edges_cell_idx_at80) - 1)];
                                double _in_z_w_con_c_full_1 = z_w_con_c_full[(((((nproma * p_patch_nlev) * (p_patch_edges_cell_blk_at83 - 1)) + (nproma * (_loop_it_56 - 1))) + p_patch_edges_cell_idx_at82) - 1)];
                                double _out_p_diag_ddt_vn_cor_pc;

                                ///////////////////
                                // Tasklet code (t_0)
                                _out_p_diag_ddt_vn_cor_pc = (- ((_in_p_diag_vt_0 * _in_p_patch_edges_f_e_0) + (((_in_p_int_c_lin_e_0 * _in_z_w_con_c_full_0) + (_in_p_int_c_lin_e_1 * _in_z_w_con_c_full_1)) * (- _in_p_patch_edges_ft_e_0))));
                                ///////////////////

                                p_diag_ddt_vn_cor_pc[((((_loop_it_57 - offset_p_diag_ddt_vn_cor_pc_d0) + (((p_diag_ddt_vn_cor_pc_d0 * p_diag_ddt_vn_cor_pc_d1) * p_diag_ddt_vn_cor_pc_d2) * (ntnd - offset_p_diag_ddt_vn_cor_pc_d3))) + ((p_diag_ddt_vn_cor_pc_d0 * p_diag_ddt_vn_cor_pc_d1) * (_loop_it_49 - offset_p_diag_ddt_vn_cor_pc_d2))) + (p_diag_ddt_vn_cor_pc_d0 * (_loop_it_56 - offset_p_diag_ddt_vn_cor_pc_d1)))] = _out_p_diag_ddt_vn_cor_pc;
                            }

                        }

                    }



                }


            }

        }


        if_cond_399 = scal_lextra_diffu;


        if (if_cond_399) {
            {

                {
                    int _out;

                    ///////////////////
                    // Tasklet code (set_ie)
                    _out = 0;
                    ///////////////////

                    ie = _out;
                }

            }
            loopend_402 = (nlev - 4);

            loopbegin_403 = max(3, (nrdmax_jg - 2));


            for (_loop_it_58 = loopbegin_403; (_loop_it_58 < (loopend_402 + 1)); _loop_it_58 = (_loop_it_58 + 1)) {

                {

                    {
                        bool _in_levelmask_0 = levelmask[(_loop_it_58 - 1)];
                        bool _in_levelmask_1 = levelmask[_loop_it_58];
                        int64_t _out_if_cond_406;

                        ///////////////////
                        // Tasklet code (t_407)
                        _out_if_cond_406 = (_in_levelmask_0 || _in_levelmask_1);
                        ///////////////////

                        if_cond_406 = _out_if_cond_406;
                    }

                }

                if (if_cond_406) {

                    for (_loop_it_59 = i_startidx; (_loop_it_59 < (i_endidx + 1)); _loop_it_59 = (_loop_it_59 + 1)) {

                        p_patch_edges_cell_idx_at84 = p_patch_edges_cell_idx[(((_loop_it_59 - offset_p_patch_edges_cell_idx_d0) + ((p_patch_edges_cell_idx_d0 * p_patch_edges_cell_idx_d1) * (1 - offset_p_patch_edges_cell_idx_d2))) + (p_patch_edges_cell_idx_d0 * (_loop_it_49 - offset_p_patch_edges_cell_idx_d1)))];

                        p_patch_edges_cell_blk_at85 = p_patch_edges_cell_blk[(((_loop_it_59 - offset_p_patch_edges_cell_blk_d0) + ((p_patch_edges_cell_blk_d0 * p_patch_edges_cell_blk_d1) * (1 - offset_p_patch_edges_cell_blk_d2))) + (p_patch_edges_cell_blk_d0 * (_loop_it_49 - offset_p_patch_edges_cell_blk_d1)))];

                        p_patch_edges_cell_idx_at86 = p_patch_edges_cell_idx[(((_loop_it_59 - offset_p_patch_edges_cell_idx_d0) + ((p_patch_edges_cell_idx_d0 * p_patch_edges_cell_idx_d1) * (2 - offset_p_patch_edges_cell_idx_d2))) + (p_patch_edges_cell_idx_d0 * (_loop_it_49 - offset_p_patch_edges_cell_idx_d1)))];

                        p_patch_edges_cell_blk_at87 = p_patch_edges_cell_blk[(((_loop_it_59 - offset_p_patch_edges_cell_blk_d0) + ((p_patch_edges_cell_blk_d0 * p_patch_edges_cell_blk_d1) * (2 - offset_p_patch_edges_cell_blk_d2))) + (p_patch_edges_cell_blk_d0 * (_loop_it_49 - offset_p_patch_edges_cell_blk_d1)))];
                        {

                            {
                                double _in_p_int_c_lin_e_0 = p_int_c_lin_e[(((_loop_it_59 - offset_p_int_c_lin_e_d0) + ((p_int_c_lin_e_d0 * p_int_c_lin_e_d1) * (_loop_it_49 - offset_p_int_c_lin_e_d2))) + (p_int_c_lin_e_d0 * (1 - offset_p_int_c_lin_e_d1)))];
                                double _in_p_int_c_lin_e_1 = p_int_c_lin_e[(((_loop_it_59 - offset_p_int_c_lin_e_d0) + ((p_int_c_lin_e_d0 * p_int_c_lin_e_d1) * (_loop_it_49 - offset_p_int_c_lin_e_d2))) + (p_int_c_lin_e_d0 * (2 - offset_p_int_c_lin_e_d1)))];
                                double _in_z_w_con_c_full_0 = z_w_con_c_full[(((((nproma * p_patch_nlev) * (p_patch_edges_cell_blk_at85 - 1)) + (nproma * (_loop_it_58 - 1))) + p_patch_edges_cell_idx_at84) - 1)];
                                double _in_z_w_con_c_full_1 = z_w_con_c_full[(((((nproma * p_patch_nlev) * (p_patch_edges_cell_blk_at87 - 1)) + (nproma * (_loop_it_58 - 1))) + p_patch_edges_cell_idx_at86) - 1)];
                                double _out_w_con_e;

                                ///////////////////
                                // Tasklet code (t_415)
                                _out_w_con_e = ((_in_p_int_c_lin_e_0 * _in_z_w_con_c_full_0) + (_in_p_int_c_lin_e_1 * _in_z_w_con_c_full_1));
                                ///////////////////

                                w_con_e = _out_w_con_e;
                            }

                        }
                        {

                            {
                                double _in_p_metrics_ddqz_z_full_e_0 = p_metrics_ddqz_z_full_e[(((_loop_it_59 - offset_p_metrics_ddqz_z_full_e_d0) + ((p_metrics_ddqz_z_full_e_d0 * p_metrics_ddqz_z_full_e_d1) * (_loop_it_49 - offset_p_metrics_ddqz_z_full_e_d2))) + (p_metrics_ddqz_z_full_e_d0 * (_loop_it_58 - offset_p_metrics_ddqz_z_full_e_d1)))];
                                double _in_cfl_w_limit = cfl_w_limit;
                                double _in_w_con_e = w_con_e;
                                int64_t _out_if_cond_416;

                                ///////////////////
                                // Tasklet code (t_417)
                                _out_if_cond_416 = (abs(_in_w_con_e) > (_in_cfl_w_limit * _in_p_metrics_ddqz_z_full_e_0));
                                ///////////////////

                                if_cond_416 = _out_if_cond_416;
                            }

                        }

                        if (if_cond_416) {
                            {

                                {
                                    double _in_p_metrics_ddqz_z_full_e_0 = p_metrics_ddqz_z_full_e[(((_loop_it_59 - offset_p_metrics_ddqz_z_full_e_d0) + ((p_metrics_ddqz_z_full_e_d0 * p_metrics_ddqz_z_full_e_d1) * (_loop_it_49 - offset_p_metrics_ddqz_z_full_e_d2))) + (p_metrics_ddqz_z_full_e_d0 * (_loop_it_58 - offset_p_metrics_ddqz_z_full_e_d1)))];
                                    double _in_cfl_w_limit = cfl_w_limit;
                                    double _in_dtime = dtime;
                                    double _in_scalfac_exdiff = scalfac_exdiff;
                                    double _in_w_con_e = w_con_e;
                                    double _out_difcoef;

                                    ///////////////////
                                    // Tasklet code (t_420)
                                    _out_difcoef = (_in_scalfac_exdiff * min((0.85 - (_in_cfl_w_limit * _in_dtime)), (((abs(_in_w_con_e) * _in_dtime) / _in_p_metrics_ddqz_z_full_e_0) - (_in_cfl_w_limit * _in_dtime))));
                                    ///////////////////

                                    difcoef = _out_difcoef;
                                }

                            }
                            p_patch_edges_quad_idx_at88 = p_patch_edges_quad_idx[(((_loop_it_59 - offset_p_patch_edges_quad_idx_d0) + ((p_patch_edges_quad_idx_d0 * p_patch_edges_quad_idx_d1) * (1 - offset_p_patch_edges_quad_idx_d2))) + (p_patch_edges_quad_idx_d0 * (_loop_it_49 - offset_p_patch_edges_quad_idx_d1)))];

                            p_patch_edges_quad_blk_at89 = p_patch_edges_quad_blk[(((_loop_it_59 - offset_p_patch_edges_quad_blk_d0) + ((p_patch_edges_quad_blk_d0 * p_patch_edges_quad_blk_d1) * (1 - offset_p_patch_edges_quad_blk_d2))) + (p_patch_edges_quad_blk_d0 * (_loop_it_49 - offset_p_patch_edges_quad_blk_d1)))];

                            p_patch_edges_quad_idx_at90 = p_patch_edges_quad_idx[(((_loop_it_59 - offset_p_patch_edges_quad_idx_d0) + ((p_patch_edges_quad_idx_d0 * p_patch_edges_quad_idx_d1) * (2 - offset_p_patch_edges_quad_idx_d2))) + (p_patch_edges_quad_idx_d0 * (_loop_it_49 - offset_p_patch_edges_quad_idx_d1)))];

                            p_patch_edges_quad_blk_at91 = p_patch_edges_quad_blk[(((_loop_it_59 - offset_p_patch_edges_quad_blk_d0) + ((p_patch_edges_quad_blk_d0 * p_patch_edges_quad_blk_d1) * (2 - offset_p_patch_edges_quad_blk_d2))) + (p_patch_edges_quad_blk_d0 * (_loop_it_49 - offset_p_patch_edges_quad_blk_d1)))];

                            p_patch_edges_quad_idx_at92 = p_patch_edges_quad_idx[(((_loop_it_59 - offset_p_patch_edges_quad_idx_d0) + ((p_patch_edges_quad_idx_d0 * p_patch_edges_quad_idx_d1) * (3 - offset_p_patch_edges_quad_idx_d2))) + (p_patch_edges_quad_idx_d0 * (_loop_it_49 - offset_p_patch_edges_quad_idx_d1)))];

                            p_patch_edges_quad_blk_at93 = p_patch_edges_quad_blk[(((_loop_it_59 - offset_p_patch_edges_quad_blk_d0) + ((p_patch_edges_quad_blk_d0 * p_patch_edges_quad_blk_d1) * (3 - offset_p_patch_edges_quad_blk_d2))) + (p_patch_edges_quad_blk_d0 * (_loop_it_49 - offset_p_patch_edges_quad_blk_d1)))];

                            p_patch_edges_quad_idx_at94 = p_patch_edges_quad_idx[(((_loop_it_59 - offset_p_patch_edges_quad_idx_d0) + ((p_patch_edges_quad_idx_d0 * p_patch_edges_quad_idx_d1) * (4 - offset_p_patch_edges_quad_idx_d2))) + (p_patch_edges_quad_idx_d0 * (_loop_it_49 - offset_p_patch_edges_quad_idx_d1)))];

                            p_patch_edges_quad_blk_at95 = p_patch_edges_quad_blk[(((_loop_it_59 - offset_p_patch_edges_quad_blk_d0) + ((p_patch_edges_quad_blk_d0 * p_patch_edges_quad_blk_d1) * (4 - offset_p_patch_edges_quad_blk_d2))) + (p_patch_edges_quad_blk_d0 * (_loop_it_49 - offset_p_patch_edges_quad_blk_d1)))];

                            p_patch_edges_vertex_idx_at96 = p_patch_edges_vertex_idx[(((_loop_it_59 - offset_p_patch_edges_vertex_idx_d0) + ((p_patch_edges_vertex_idx_d0 * p_patch_edges_vertex_idx_d1) * (2 - offset_p_patch_edges_vertex_idx_d2))) + (p_patch_edges_vertex_idx_d0 * (_loop_it_49 - offset_p_patch_edges_vertex_idx_d1)))];

                            p_patch_edges_vertex_blk_at97 = p_patch_edges_vertex_blk[(((_loop_it_59 - offset_p_patch_edges_vertex_blk_d0) + ((p_patch_edges_vertex_blk_d0 * p_patch_edges_vertex_blk_d1) * (2 - offset_p_patch_edges_vertex_blk_d2))) + (p_patch_edges_vertex_blk_d0 * (_loop_it_49 - offset_p_patch_edges_vertex_blk_d1)))];

                            p_patch_edges_vertex_idx_at98 = p_patch_edges_vertex_idx[(((_loop_it_59 - offset_p_patch_edges_vertex_idx_d0) + ((p_patch_edges_vertex_idx_d0 * p_patch_edges_vertex_idx_d1) * (1 - offset_p_patch_edges_vertex_idx_d2))) + (p_patch_edges_vertex_idx_d0 * (_loop_it_49 - offset_p_patch_edges_vertex_idx_d1)))];

                            p_patch_edges_vertex_blk_at99 = p_patch_edges_vertex_blk[(((_loop_it_59 - offset_p_patch_edges_vertex_blk_d0) + ((p_patch_edges_vertex_blk_d0 * p_patch_edges_vertex_blk_d1) * (1 - offset_p_patch_edges_vertex_blk_d2))) + (p_patch_edges_vertex_blk_d0 * (_loop_it_49 - offset_p_patch_edges_vertex_blk_d1)))];
                            {

                                {
                                    double _in_p_diag_ddt_vn_apc_pc_0 = p_diag_ddt_vn_apc_pc[((((_loop_it_59 - offset_p_diag_ddt_vn_apc_pc_d0) + (((p_diag_ddt_vn_apc_pc_d0 * p_diag_ddt_vn_apc_pc_d1) * p_diag_ddt_vn_apc_pc_d2) * (ntnd - offset_p_diag_ddt_vn_apc_pc_d3))) + ((p_diag_ddt_vn_apc_pc_d0 * p_diag_ddt_vn_apc_pc_d1) * (_loop_it_49 - offset_p_diag_ddt_vn_apc_pc_d2))) + (p_diag_ddt_vn_apc_pc_d0 * (_loop_it_58 - offset_p_diag_ddt_vn_apc_pc_d1)))];
                                    double _in_p_int_geofac_grdiv_0 = p_int_geofac_grdiv[(((_loop_it_59 - offset_p_int_geofac_grdiv_d0) + ((p_int_geofac_grdiv_d0 * p_int_geofac_grdiv_d1) * (_loop_it_49 - offset_p_int_geofac_grdiv_d2))) + (p_int_geofac_grdiv_d0 * (1 - offset_p_int_geofac_grdiv_d1)))];
                                    double _in_p_int_geofac_grdiv_1 = p_int_geofac_grdiv[(((_loop_it_59 - offset_p_int_geofac_grdiv_d0) + ((p_int_geofac_grdiv_d0 * p_int_geofac_grdiv_d1) * (_loop_it_49 - offset_p_int_geofac_grdiv_d2))) + (p_int_geofac_grdiv_d0 * (2 - offset_p_int_geofac_grdiv_d1)))];
                                    double _in_p_int_geofac_grdiv_2 = p_int_geofac_grdiv[(((_loop_it_59 - offset_p_int_geofac_grdiv_d0) + ((p_int_geofac_grdiv_d0 * p_int_geofac_grdiv_d1) * (_loop_it_49 - offset_p_int_geofac_grdiv_d2))) + (p_int_geofac_grdiv_d0 * (3 - offset_p_int_geofac_grdiv_d1)))];
                                    double _in_p_int_geofac_grdiv_3 = p_int_geofac_grdiv[(((_loop_it_59 - offset_p_int_geofac_grdiv_d0) + ((p_int_geofac_grdiv_d0 * p_int_geofac_grdiv_d1) * (_loop_it_49 - offset_p_int_geofac_grdiv_d2))) + (p_int_geofac_grdiv_d0 * (4 - offset_p_int_geofac_grdiv_d1)))];
                                    double _in_p_int_geofac_grdiv_4 = p_int_geofac_grdiv[(((_loop_it_59 - offset_p_int_geofac_grdiv_d0) + ((p_int_geofac_grdiv_d0 * p_int_geofac_grdiv_d1) * (_loop_it_49 - offset_p_int_geofac_grdiv_d2))) + (p_int_geofac_grdiv_d0 * (5 - offset_p_int_geofac_grdiv_d1)))];
                                    double _in_p_patch_edges_area_edge_0 = p_patch_edges_area_edge[((_loop_it_59 - offset_p_patch_edges_area_edge_d0) + (p_patch_edges_area_edge_d0 * (_loop_it_49 - offset_p_patch_edges_area_edge_d1)))];
                                    double _in_p_patch_edges_inv_primal_edge_length_0 = p_patch_edges_inv_primal_edge_length[((_loop_it_59 - offset_p_patch_edges_inv_primal_edge_length_d0) + (p_patch_edges_inv_primal_edge_length_d0 * (_loop_it_49 - offset_p_patch_edges_inv_primal_edge_length_d1)))];
                                    double _in_p_patch_edges_tangent_orientation_0 = p_patch_edges_tangent_orientation[((_loop_it_59 - offset_p_patch_edges_tangent_orientation_d0) + (p_patch_edges_tangent_orientation_d0 * (_loop_it_49 - offset_p_patch_edges_tangent_orientation_d1)))];
                                    double _in_p_prog_vn_0 = p_prog_vn[(((_loop_it_59 - offset_p_prog_vn_d0) + ((p_prog_vn_d0 * p_prog_vn_d1) * (_loop_it_49 - offset_p_prog_vn_d2))) + (p_prog_vn_d0 * (_loop_it_58 - offset_p_prog_vn_d1)))];
                                    double _in_p_prog_vn_1 = p_prog_vn[((((- offset_p_prog_vn_d0) + p_patch_edges_quad_idx_at88) + ((p_prog_vn_d0 * p_prog_vn_d1) * ((- offset_p_prog_vn_d2) + p_patch_edges_quad_blk_at89))) + (p_prog_vn_d0 * (_loop_it_58 - offset_p_prog_vn_d1)))];
                                    double _in_p_prog_vn_2 = p_prog_vn[((((- offset_p_prog_vn_d0) + p_patch_edges_quad_idx_at90) + ((p_prog_vn_d0 * p_prog_vn_d1) * ((- offset_p_prog_vn_d2) + p_patch_edges_quad_blk_at91))) + (p_prog_vn_d0 * (_loop_it_58 - offset_p_prog_vn_d1)))];
                                    double _in_p_prog_vn_3 = p_prog_vn[((((- offset_p_prog_vn_d0) + p_patch_edges_quad_idx_at92) + ((p_prog_vn_d0 * p_prog_vn_d1) * ((- offset_p_prog_vn_d2) + p_patch_edges_quad_blk_at93))) + (p_prog_vn_d0 * (_loop_it_58 - offset_p_prog_vn_d1)))];
                                    double _in_p_prog_vn_4 = p_prog_vn[((((- offset_p_prog_vn_d0) + p_patch_edges_quad_idx_at94) + ((p_prog_vn_d0 * p_prog_vn_d1) * ((- offset_p_prog_vn_d2) + p_patch_edges_quad_blk_at95))) + (p_prog_vn_d0 * (_loop_it_58 - offset_p_prog_vn_d1)))];
                                    double _in_zeta_0 = zeta[(((((nproma * p_patch_nlev) * (p_patch_edges_vertex_blk_at97 - 1)) + (nproma * (_loop_it_58 - 1))) + p_patch_edges_vertex_idx_at96) - 1)];
                                    double _in_zeta_1 = zeta[(((((nproma * p_patch_nlev) * (p_patch_edges_vertex_blk_at99 - 1)) + (nproma * (_loop_it_58 - 1))) + p_patch_edges_vertex_idx_at98) - 1)];
                                    double _in_difcoef = difcoef;
                                    double _out_p_diag_ddt_vn_apc_pc;

                                    ///////////////////
                                    // Tasklet code (t_433)
                                    _out_p_diag_ddt_vn_apc_pc = (_in_p_diag_ddt_vn_apc_pc_0 + ((_in_difcoef * _in_p_patch_edges_area_edge_0) * ((((((_in_p_int_geofac_grdiv_0 * _in_p_prog_vn_0) + (_in_p_int_geofac_grdiv_1 * _in_p_prog_vn_1)) + (_in_p_int_geofac_grdiv_2 * _in_p_prog_vn_2)) + (_in_p_int_geofac_grdiv_3 * _in_p_prog_vn_3)) + (_in_p_int_geofac_grdiv_4 * _in_p_prog_vn_4)) + ((_in_p_patch_edges_tangent_orientation_0 * _in_p_patch_edges_inv_primal_edge_length_0) * (_in_zeta_0 - _in_zeta_1)))));
                                    ///////////////////

                                    p_diag_ddt_vn_apc_pc[((((_loop_it_59 - offset_p_diag_ddt_vn_apc_pc_d0) + (((p_diag_ddt_vn_apc_pc_d0 * p_diag_ddt_vn_apc_pc_d1) * p_diag_ddt_vn_apc_pc_d2) * (ntnd - offset_p_diag_ddt_vn_apc_pc_d3))) + ((p_diag_ddt_vn_apc_pc_d0 * p_diag_ddt_vn_apc_pc_d1) * (_loop_it_49 - offset_p_diag_ddt_vn_apc_pc_d2))) + (p_diag_ddt_vn_apc_pc_d0 * (_loop_it_58 - offset_p_diag_ddt_vn_apc_pc_d1)))] = _out_p_diag_ddt_vn_apc_pc;
                                }

                            }
                        }


                    }


                }


            }


        }


    }


    i_startblk = p_patch_cells_start_block[(4 - offset_p_patch_cells_start_block_d0)];

    i_endblk = p_patch_cells_end_block[1];
    {

        {
            double _out;

            ///////////////////
            // Tasklet code (set__QQred_lift_0)
            _out = (- INFINITY);
            ///////////////////

            _QQred_lift_0 = _out;
        }

    }

    for (_loop_it_60 = i_startblk; (_loop_it_60 < (i_endblk + 1)); _loop_it_60 = (_loop_it_60 + 1)) {
        {

            {
                double _in_vcflmax_0 = vcflmax[(_loop_it_60 - 1)];
                double _in__QQred_lift_0 = _QQred_lift_0;
                double _out__QQred_lift_0;

                ///////////////////
                // Tasklet code (t_0)
                _out__QQred_lift_0 = max(_in__QQred_lift_0, _in_vcflmax_0);
                ///////////////////

                _QQred_lift_0 = _out__QQred_lift_0;
            }

        }

    }

    {
        double max_vcfl_dyn;

        {
            double _in_p_diag_max_vcfl_dyn = scal_p_diag_max_vcfl_dyn;
            double _in__QQred_lift_0 = _QQred_lift_0;
            double _out;

            ///////////////////
            // Tasklet code (set_max_vcfl_dyn)
            _out = max(_in_p_diag_max_vcfl_dyn, _in__QQred_lift_0);
            ///////////////////

            max_vcfl_dyn = _out;
        }
        {
            double _in_max_vcfl_dyn = max_vcfl_dyn;
            double _out;

            ///////////////////
            // Tasklet code (set_p_diag_max_vcfl_dyn)
            _out = _in_max_vcfl_dyn;
            ///////////////////

            scal_p_diag_max_vcfl_dyn = _out;
        }

    }
    if_cond_439 = (timers_level > 5);


    if (if_cond_439) {

    }

    {

        {
            double _cpy_in = scal_p_diag_max_vcfl_dyn;
            double _cpy_out;

            ///////////////////
            // Tasklet code (copy_scal_p_diag_max_vcfl_dyn_to_p_diag_max_vcfl_dyn)
            _cpy_out = _cpy_in;
            ///////////////////

            p_diag_max_vcfl_dyn[0] = _cpy_out;
        }

    }
    static_assert(std::is_trivially_destructible<bool>::value, "aligned heap deallocation skips destructors");
    ::operator delete[](cfl_clipping, std::align_val_t(64));
    static_assert(std::is_trivially_destructible<bool>::value, "aligned heap deallocation skips destructors");
    ::operator delete[](levelmask, std::align_val_t(64));
    static_assert(std::is_trivially_destructible<bool>::value, "aligned heap deallocation skips destructors");
    ::operator delete[](levmask, std::align_val_t(64));
    static_assert(std::is_trivially_destructible<double>::value, "aligned heap deallocation skips destructors");
    ::operator delete[](vcflmax, std::align_val_t(64));
    static_assert(std::is_trivially_destructible<double>::value, "aligned heap deallocation skips destructors");
    ::operator delete[](z_ekinh, std::align_val_t(64));
    static_assert(std::is_trivially_destructible<double>::value, "aligned heap deallocation skips destructors");
    ::operator delete[](z_v_grad_w, std::align_val_t(64));
    static_assert(std::is_trivially_destructible<double>::value, "aligned heap deallocation skips destructors");
    ::operator delete[](z_w_con_c, std::align_val_t(64));
    static_assert(std::is_trivially_destructible<double>::value, "aligned heap deallocation skips destructors");
    ::operator delete[](z_w_con_c_full, std::align_val_t(64));
    static_assert(std::is_trivially_destructible<double>::value, "aligned heap deallocation skips destructors");
    ::operator delete[](z_w_concorr_mc, std::align_val_t(64));
    static_assert(std::is_trivially_destructible<double>::value, "aligned heap deallocation skips destructors");
    ::operator delete[](z_w_v, std::align_val_t(64));
    static_assert(std::is_trivially_destructible<double>::value, "aligned heap deallocation skips destructors");
    ::operator delete[](zeta, std::align_val_t(64));
}

DACE_EXPORTED void __program_velocity_tendencies(velocity_tendencies_state_t *__state, bool * __restrict__ i_am_accel_node, bool * __restrict__ lextra_diffu, bool * __restrict__ lvert_nest, int * __restrict__ nflatlev, int * __restrict__ nrdmax, bool * __restrict__ p_diag_ddt_vn_adv_is_associated, double * __restrict__ p_diag_ddt_vn_apc_pc, bool * __restrict__ p_diag_ddt_vn_cor_is_associated, double * __restrict__ p_diag_ddt_vn_cor_pc, double * __restrict__ p_diag_ddt_w_adv_pc, double * __restrict__ p_diag_max_vcfl_dyn, double * __restrict__ p_diag_vn_ie, double * __restrict__ p_diag_vn_ie_ubc, double * __restrict__ p_diag_vt, double * __restrict__ p_diag_w_concorr_c, double * __restrict__ p_int_c_lin_e, double * __restrict__ p_int_cells_aw_verts, double * __restrict__ p_int_e_bln_c_s, double * __restrict__ p_int_geofac_grdiv, double * __restrict__ p_int_geofac_n2s, double * __restrict__ p_int_geofac_rot, double * __restrict__ p_int_rbf_vec_coeff_e, double * __restrict__ p_metrics_coeff1_dwdz, double * __restrict__ p_metrics_coeff2_dwdz, double * __restrict__ p_metrics_coeff_gradekin, double * __restrict__ p_metrics_ddqz_z_full_e, double * __restrict__ p_metrics_ddqz_z_half, double * __restrict__ p_metrics_ddxn_z_full, double * __restrict__ p_metrics_ddxt_z_full, double * __restrict__ p_metrics_deepatmo_gradh_ifc, double * __restrict__ p_metrics_deepatmo_gradh_mc, double * __restrict__ p_metrics_deepatmo_invr_ifc, double * __restrict__ p_metrics_deepatmo_invr_mc, double * __restrict__ p_metrics_wgtfac_c, double * __restrict__ p_metrics_wgtfac_e, double * __restrict__ p_metrics_wgtfacq_e, double * __restrict__ p_patch_cells_area, bool * __restrict__ p_patch_cells_decomp_info_owner_mask, int * __restrict__ p_patch_cells_edge_blk, int * __restrict__ p_patch_cells_edge_idx, int * __restrict__ p_patch_cells_end_block, int * __restrict__ p_patch_cells_end_index, int * __restrict__ p_patch_cells_neighbor_blk, int * __restrict__ p_patch_cells_neighbor_idx, int * __restrict__ p_patch_cells_start_block, int * __restrict__ p_patch_cells_start_index, double * __restrict__ p_patch_edges_area_edge, int * __restrict__ p_patch_edges_cell_blk, int * __restrict__ p_patch_edges_cell_idx, int * __restrict__ p_patch_edges_end_block, int * __restrict__ p_patch_edges_end_index, double * __restrict__ p_patch_edges_f_e, double * __restrict__ p_patch_edges_fn_e, double * __restrict__ p_patch_edges_ft_e, double * __restrict__ p_patch_edges_inv_dual_edge_length, double * __restrict__ p_patch_edges_inv_primal_edge_length, int * __restrict__ p_patch_edges_quad_blk, int * __restrict__ p_patch_edges_quad_idx, int * __restrict__ p_patch_edges_start_block, int * __restrict__ p_patch_edges_start_index, double * __restrict__ p_patch_edges_tangent_orientation, int * __restrict__ p_patch_edges_vertex_blk, int * __restrict__ p_patch_edges_vertex_idx, int * __restrict__ p_patch_id, int * __restrict__ p_patch_verts_cell_blk, int * __restrict__ p_patch_verts_cell_idx, int * __restrict__ p_patch_verts_edge_blk, int * __restrict__ p_patch_verts_edge_idx, int * __restrict__ p_patch_verts_end_block, int * __restrict__ p_patch_verts_end_index, int * __restrict__ p_patch_verts_start_block, int * __restrict__ p_patch_verts_start_index, double * __restrict__ p_prog_vn, double * __restrict__ p_prog_w, int * __restrict__ timer_intp, int * __restrict__ timer_solve_nh_veltend, double * __restrict__ z_kin_hor_e, double * __restrict__ z_vt_ie, double * __restrict__ z_w_concorr_me, double dt_linintp_ubc, double dtime, int istep, bool ldeepatmo, bool lvn_only, int nproma, int ntnd, int64_t offset_p_diag_ddt_vn_apc_pc_d0, int64_t offset_p_diag_ddt_vn_apc_pc_d1, int64_t offset_p_diag_ddt_vn_apc_pc_d2, int64_t offset_p_diag_ddt_vn_apc_pc_d3, int64_t offset_p_diag_ddt_vn_cor_pc_d0, int64_t offset_p_diag_ddt_vn_cor_pc_d1, int64_t offset_p_diag_ddt_vn_cor_pc_d2, int64_t offset_p_diag_ddt_vn_cor_pc_d3, int64_t offset_p_diag_ddt_w_adv_pc_d0, int64_t offset_p_diag_ddt_w_adv_pc_d1, int64_t offset_p_diag_ddt_w_adv_pc_d2, int64_t offset_p_diag_ddt_w_adv_pc_d3, int64_t offset_p_diag_vn_ie_d0, int64_t offset_p_diag_vn_ie_d1, int64_t offset_p_diag_vn_ie_d2, int64_t offset_p_diag_vn_ie_ubc_d0, int64_t offset_p_diag_vn_ie_ubc_d1, int64_t offset_p_diag_vn_ie_ubc_d2, int64_t offset_p_diag_vt_d0, int64_t offset_p_diag_vt_d1, int64_t offset_p_diag_vt_d2, int64_t offset_p_diag_w_concorr_c_d0, int64_t offset_p_diag_w_concorr_c_d1, int64_t offset_p_diag_w_concorr_c_d2, int64_t offset_p_int_c_lin_e_d0, int64_t offset_p_int_c_lin_e_d1, int64_t offset_p_int_c_lin_e_d2, int64_t offset_p_int_cells_aw_verts_d0, int64_t offset_p_int_cells_aw_verts_d1, int64_t offset_p_int_cells_aw_verts_d2, int64_t offset_p_int_e_bln_c_s_d0, int64_t offset_p_int_e_bln_c_s_d1, int64_t offset_p_int_e_bln_c_s_d2, int64_t offset_p_int_geofac_grdiv_d0, int64_t offset_p_int_geofac_grdiv_d1, int64_t offset_p_int_geofac_grdiv_d2, int64_t offset_p_int_geofac_n2s_d0, int64_t offset_p_int_geofac_n2s_d1, int64_t offset_p_int_geofac_n2s_d2, int64_t offset_p_int_geofac_rot_d0, int64_t offset_p_int_geofac_rot_d1, int64_t offset_p_int_geofac_rot_d2, int64_t offset_p_int_rbf_vec_coeff_e_d0, int64_t offset_p_int_rbf_vec_coeff_e_d1, int64_t offset_p_int_rbf_vec_coeff_e_d2, int64_t offset_p_metrics_coeff1_dwdz_d0, int64_t offset_p_metrics_coeff1_dwdz_d1, int64_t offset_p_metrics_coeff1_dwdz_d2, int64_t offset_p_metrics_coeff2_dwdz_d0, int64_t offset_p_metrics_coeff2_dwdz_d1, int64_t offset_p_metrics_coeff2_dwdz_d2, int64_t offset_p_metrics_coeff_gradekin_d0, int64_t offset_p_metrics_coeff_gradekin_d1, int64_t offset_p_metrics_coeff_gradekin_d2, int64_t offset_p_metrics_ddqz_z_full_e_d0, int64_t offset_p_metrics_ddqz_z_full_e_d1, int64_t offset_p_metrics_ddqz_z_full_e_d2, int64_t offset_p_metrics_ddqz_z_half_d0, int64_t offset_p_metrics_ddqz_z_half_d1, int64_t offset_p_metrics_ddqz_z_half_d2, int64_t offset_p_metrics_ddxn_z_full_d0, int64_t offset_p_metrics_ddxn_z_full_d1, int64_t offset_p_metrics_ddxn_z_full_d2, int64_t offset_p_metrics_ddxt_z_full_d0, int64_t offset_p_metrics_ddxt_z_full_d1, int64_t offset_p_metrics_ddxt_z_full_d2, int64_t offset_p_metrics_deepatmo_gradh_ifc_d0, int64_t offset_p_metrics_deepatmo_gradh_mc_d0, int64_t offset_p_metrics_deepatmo_invr_ifc_d0, int64_t offset_p_metrics_deepatmo_invr_mc_d0, int64_t offset_p_metrics_wgtfac_c_d0, int64_t offset_p_metrics_wgtfac_c_d1, int64_t offset_p_metrics_wgtfac_c_d2, int64_t offset_p_metrics_wgtfac_e_d0, int64_t offset_p_metrics_wgtfac_e_d1, int64_t offset_p_metrics_wgtfac_e_d2, int64_t offset_p_metrics_wgtfacq_e_d0, int64_t offset_p_metrics_wgtfacq_e_d1, int64_t offset_p_metrics_wgtfacq_e_d2, int64_t offset_p_patch_cells_area_d0, int64_t offset_p_patch_cells_area_d1, int64_t offset_p_patch_cells_decomp_info_owner_mask_d0, int64_t offset_p_patch_cells_decomp_info_owner_mask_d1, int64_t offset_p_patch_cells_edge_blk_d0, int64_t offset_p_patch_cells_edge_blk_d1, int64_t offset_p_patch_cells_edge_blk_d2, int64_t offset_p_patch_cells_edge_idx_d0, int64_t offset_p_patch_cells_edge_idx_d1, int64_t offset_p_patch_cells_edge_idx_d2, int64_t offset_p_patch_cells_neighbor_blk_d0, int64_t offset_p_patch_cells_neighbor_blk_d1, int64_t offset_p_patch_cells_neighbor_blk_d2, int64_t offset_p_patch_cells_neighbor_idx_d0, int64_t offset_p_patch_cells_neighbor_idx_d1, int64_t offset_p_patch_cells_neighbor_idx_d2, int64_t offset_p_patch_cells_start_block_d0, int64_t offset_p_patch_cells_start_index_d0, int64_t offset_p_patch_edges_area_edge_d0, int64_t offset_p_patch_edges_area_edge_d1, int64_t offset_p_patch_edges_cell_blk_d0, int64_t offset_p_patch_edges_cell_blk_d1, int64_t offset_p_patch_edges_cell_blk_d2, int64_t offset_p_patch_edges_cell_idx_d0, int64_t offset_p_patch_edges_cell_idx_d1, int64_t offset_p_patch_edges_cell_idx_d2, int64_t offset_p_patch_edges_f_e_d0, int64_t offset_p_patch_edges_f_e_d1, int64_t offset_p_patch_edges_fn_e_d0, int64_t offset_p_patch_edges_fn_e_d1, int64_t offset_p_patch_edges_ft_e_d0, int64_t offset_p_patch_edges_ft_e_d1, int64_t offset_p_patch_edges_inv_dual_edge_length_d0, int64_t offset_p_patch_edges_inv_dual_edge_length_d1, int64_t offset_p_patch_edges_inv_primal_edge_length_d0, int64_t offset_p_patch_edges_inv_primal_edge_length_d1, int64_t offset_p_patch_edges_quad_blk_d0, int64_t offset_p_patch_edges_quad_blk_d1, int64_t offset_p_patch_edges_quad_blk_d2, int64_t offset_p_patch_edges_quad_idx_d0, int64_t offset_p_patch_edges_quad_idx_d1, int64_t offset_p_patch_edges_quad_idx_d2, int64_t offset_p_patch_edges_start_block_d0, int64_t offset_p_patch_edges_start_index_d0, int64_t offset_p_patch_edges_tangent_orientation_d0, int64_t offset_p_patch_edges_tangent_orientation_d1, int64_t offset_p_patch_edges_vertex_blk_d0, int64_t offset_p_patch_edges_vertex_blk_d1, int64_t offset_p_patch_edges_vertex_blk_d2, int64_t offset_p_patch_edges_vertex_idx_d0, int64_t offset_p_patch_edges_vertex_idx_d1, int64_t offset_p_patch_edges_vertex_idx_d2, int64_t offset_p_patch_verts_cell_blk_d0, int64_t offset_p_patch_verts_cell_blk_d1, int64_t offset_p_patch_verts_cell_blk_d2, int64_t offset_p_patch_verts_cell_idx_d0, int64_t offset_p_patch_verts_cell_idx_d1, int64_t offset_p_patch_verts_cell_idx_d2, int64_t offset_p_patch_verts_edge_blk_d0, int64_t offset_p_patch_verts_edge_blk_d1, int64_t offset_p_patch_verts_edge_blk_d2, int64_t offset_p_patch_verts_edge_idx_d0, int64_t offset_p_patch_verts_edge_idx_d1, int64_t offset_p_patch_verts_edge_idx_d2, int64_t offset_p_patch_verts_start_block_d0, int64_t offset_p_patch_verts_start_index_d0, int64_t offset_p_prog_vn_d0, int64_t offset_p_prog_vn_d1, int64_t offset_p_prog_vn_d2, int64_t offset_p_prog_w_d0, int64_t offset_p_prog_w_d1, int64_t offset_p_prog_w_d2, int64_t p_diag_ddt_vn_apc_pc_d0, int64_t p_diag_ddt_vn_apc_pc_d1, int64_t p_diag_ddt_vn_apc_pc_d2, int64_t p_diag_ddt_vn_cor_pc_d0, int64_t p_diag_ddt_vn_cor_pc_d1, int64_t p_diag_ddt_vn_cor_pc_d2, int64_t p_diag_ddt_w_adv_pc_d0, int64_t p_diag_ddt_w_adv_pc_d1, int64_t p_diag_ddt_w_adv_pc_d2, int64_t p_diag_vn_ie_d0, int64_t p_diag_vn_ie_d1, int64_t p_diag_vn_ie_ubc_d0, int64_t p_diag_vn_ie_ubc_d1, int64_t p_diag_vt_d0, int64_t p_diag_vt_d1, int64_t p_diag_w_concorr_c_d0, int64_t p_diag_w_concorr_c_d1, int64_t p_int_c_lin_e_d0, int64_t p_int_c_lin_e_d1, int64_t p_int_cells_aw_verts_d0, int64_t p_int_cells_aw_verts_d1, int64_t p_int_e_bln_c_s_d0, int64_t p_int_e_bln_c_s_d1, int64_t p_int_geofac_grdiv_d0, int64_t p_int_geofac_grdiv_d1, int64_t p_int_geofac_n2s_d0, int64_t p_int_geofac_n2s_d1, int64_t p_int_geofac_rot_d0, int64_t p_int_geofac_rot_d1, int64_t p_int_rbf_vec_coeff_e_d0, int64_t p_int_rbf_vec_coeff_e_d1, int64_t p_metrics_coeff1_dwdz_d0, int64_t p_metrics_coeff1_dwdz_d1, int64_t p_metrics_coeff2_dwdz_d0, int64_t p_metrics_coeff2_dwdz_d1, int64_t p_metrics_coeff_gradekin_d0, int64_t p_metrics_coeff_gradekin_d1, int64_t p_metrics_ddqz_z_full_e_d0, int64_t p_metrics_ddqz_z_full_e_d1, int64_t p_metrics_ddqz_z_half_d0, int64_t p_metrics_ddqz_z_half_d1, int64_t p_metrics_ddxn_z_full_d0, int64_t p_metrics_ddxn_z_full_d1, int64_t p_metrics_ddxt_z_full_d0, int64_t p_metrics_ddxt_z_full_d1, int64_t p_metrics_wgtfac_c_d0, int64_t p_metrics_wgtfac_c_d1, int64_t p_metrics_wgtfac_e_d0, int64_t p_metrics_wgtfac_e_d1, int64_t p_metrics_wgtfacq_e_d0, int64_t p_metrics_wgtfacq_e_d1, int64_t p_patch_cells_area_d0, int64_t p_patch_cells_decomp_info_owner_mask_d0, int64_t p_patch_cells_edge_blk_d0, int64_t p_patch_cells_edge_blk_d1, int64_t p_patch_cells_edge_idx_d0, int64_t p_patch_cells_edge_idx_d1, int64_t p_patch_cells_neighbor_blk_d0, int64_t p_patch_cells_neighbor_blk_d1, int64_t p_patch_cells_neighbor_idx_d0, int64_t p_patch_cells_neighbor_idx_d1, int64_t p_patch_edges_area_edge_d0, int64_t p_patch_edges_cell_blk_d0, int64_t p_patch_edges_cell_blk_d1, int64_t p_patch_edges_cell_idx_d0, int64_t p_patch_edges_cell_idx_d1, int64_t p_patch_edges_f_e_d0, int64_t p_patch_edges_fn_e_d0, int64_t p_patch_edges_ft_e_d0, int64_t p_patch_edges_inv_dual_edge_length_d0, int64_t p_patch_edges_inv_primal_edge_length_d0, int64_t p_patch_edges_quad_blk_d0, int64_t p_patch_edges_quad_blk_d1, int64_t p_patch_edges_quad_idx_d0, int64_t p_patch_edges_quad_idx_d1, int64_t p_patch_edges_tangent_orientation_d0, int64_t p_patch_edges_vertex_blk_d0, int64_t p_patch_edges_vertex_blk_d1, int64_t p_patch_edges_vertex_idx_d0, int64_t p_patch_edges_vertex_idx_d1, int p_patch_nblks_c, int p_patch_nblks_e, int p_patch_nblks_v, int p_patch_nlev, int p_patch_nlevp1, int p_patch_nshift, int64_t p_patch_verts_cell_blk_d0, int64_t p_patch_verts_cell_blk_d1, int64_t p_patch_verts_cell_idx_d0, int64_t p_patch_verts_cell_idx_d1, int64_t p_patch_verts_edge_blk_d0, int64_t p_patch_verts_edge_blk_d1, int64_t p_patch_verts_edge_idx_d0, int64_t p_patch_verts_edge_idx_d1, int64_t p_prog_vn_d0, int64_t p_prog_vn_d1, int64_t p_prog_w_d0, int64_t p_prog_w_d1, int timers_level, int64_t z_kin_hor_e_d0, int64_t z_kin_hor_e_d1, int64_t z_vt_ie_d0, int64_t z_vt_ie_d1, int64_t z_w_concorr_me_d0, int64_t z_w_concorr_me_d1)
{
    __program_velocity_tendencies_internal(__state, i_am_accel_node, lextra_diffu, lvert_nest, nflatlev, nrdmax, p_diag_ddt_vn_adv_is_associated, p_diag_ddt_vn_apc_pc, p_diag_ddt_vn_cor_is_associated, p_diag_ddt_vn_cor_pc, p_diag_ddt_w_adv_pc, p_diag_max_vcfl_dyn, p_diag_vn_ie, p_diag_vn_ie_ubc, p_diag_vt, p_diag_w_concorr_c, p_int_c_lin_e, p_int_cells_aw_verts, p_int_e_bln_c_s, p_int_geofac_grdiv, p_int_geofac_n2s, p_int_geofac_rot, p_int_rbf_vec_coeff_e, p_metrics_coeff1_dwdz, p_metrics_coeff2_dwdz, p_metrics_coeff_gradekin, p_metrics_ddqz_z_full_e, p_metrics_ddqz_z_half, p_metrics_ddxn_z_full, p_metrics_ddxt_z_full, p_metrics_deepatmo_gradh_ifc, p_metrics_deepatmo_gradh_mc, p_metrics_deepatmo_invr_ifc, p_metrics_deepatmo_invr_mc, p_metrics_wgtfac_c, p_metrics_wgtfac_e, p_metrics_wgtfacq_e, p_patch_cells_area, p_patch_cells_decomp_info_owner_mask, p_patch_cells_edge_blk, p_patch_cells_edge_idx, p_patch_cells_end_block, p_patch_cells_end_index, p_patch_cells_neighbor_blk, p_patch_cells_neighbor_idx, p_patch_cells_start_block, p_patch_cells_start_index, p_patch_edges_area_edge, p_patch_edges_cell_blk, p_patch_edges_cell_idx, p_patch_edges_end_block, p_patch_edges_end_index, p_patch_edges_f_e, p_patch_edges_fn_e, p_patch_edges_ft_e, p_patch_edges_inv_dual_edge_length, p_patch_edges_inv_primal_edge_length, p_patch_edges_quad_blk, p_patch_edges_quad_idx, p_patch_edges_start_block, p_patch_edges_start_index, p_patch_edges_tangent_orientation, p_patch_edges_vertex_blk, p_patch_edges_vertex_idx, p_patch_id, p_patch_verts_cell_blk, p_patch_verts_cell_idx, p_patch_verts_edge_blk, p_patch_verts_edge_idx, p_patch_verts_end_block, p_patch_verts_end_index, p_patch_verts_start_block, p_patch_verts_start_index, p_prog_vn, p_prog_w, timer_intp, timer_solve_nh_veltend, z_kin_hor_e, z_vt_ie, z_w_concorr_me, dt_linintp_ubc, dtime, istep, ldeepatmo, lvn_only, nproma, ntnd, offset_p_diag_ddt_vn_apc_pc_d0, offset_p_diag_ddt_vn_apc_pc_d1, offset_p_diag_ddt_vn_apc_pc_d2, offset_p_diag_ddt_vn_apc_pc_d3, offset_p_diag_ddt_vn_cor_pc_d0, offset_p_diag_ddt_vn_cor_pc_d1, offset_p_diag_ddt_vn_cor_pc_d2, offset_p_diag_ddt_vn_cor_pc_d3, offset_p_diag_ddt_w_adv_pc_d0, offset_p_diag_ddt_w_adv_pc_d1, offset_p_diag_ddt_w_adv_pc_d2, offset_p_diag_ddt_w_adv_pc_d3, offset_p_diag_vn_ie_d0, offset_p_diag_vn_ie_d1, offset_p_diag_vn_ie_d2, offset_p_diag_vn_ie_ubc_d0, offset_p_diag_vn_ie_ubc_d1, offset_p_diag_vn_ie_ubc_d2, offset_p_diag_vt_d0, offset_p_diag_vt_d1, offset_p_diag_vt_d2, offset_p_diag_w_concorr_c_d0, offset_p_diag_w_concorr_c_d1, offset_p_diag_w_concorr_c_d2, offset_p_int_c_lin_e_d0, offset_p_int_c_lin_e_d1, offset_p_int_c_lin_e_d2, offset_p_int_cells_aw_verts_d0, offset_p_int_cells_aw_verts_d1, offset_p_int_cells_aw_verts_d2, offset_p_int_e_bln_c_s_d0, offset_p_int_e_bln_c_s_d1, offset_p_int_e_bln_c_s_d2, offset_p_int_geofac_grdiv_d0, offset_p_int_geofac_grdiv_d1, offset_p_int_geofac_grdiv_d2, offset_p_int_geofac_n2s_d0, offset_p_int_geofac_n2s_d1, offset_p_int_geofac_n2s_d2, offset_p_int_geofac_rot_d0, offset_p_int_geofac_rot_d1, offset_p_int_geofac_rot_d2, offset_p_int_rbf_vec_coeff_e_d0, offset_p_int_rbf_vec_coeff_e_d1, offset_p_int_rbf_vec_coeff_e_d2, offset_p_metrics_coeff1_dwdz_d0, offset_p_metrics_coeff1_dwdz_d1, offset_p_metrics_coeff1_dwdz_d2, offset_p_metrics_coeff2_dwdz_d0, offset_p_metrics_coeff2_dwdz_d1, offset_p_metrics_coeff2_dwdz_d2, offset_p_metrics_coeff_gradekin_d0, offset_p_metrics_coeff_gradekin_d1, offset_p_metrics_coeff_gradekin_d2, offset_p_metrics_ddqz_z_full_e_d0, offset_p_metrics_ddqz_z_full_e_d1, offset_p_metrics_ddqz_z_full_e_d2, offset_p_metrics_ddqz_z_half_d0, offset_p_metrics_ddqz_z_half_d1, offset_p_metrics_ddqz_z_half_d2, offset_p_metrics_ddxn_z_full_d0, offset_p_metrics_ddxn_z_full_d1, offset_p_metrics_ddxn_z_full_d2, offset_p_metrics_ddxt_z_full_d0, offset_p_metrics_ddxt_z_full_d1, offset_p_metrics_ddxt_z_full_d2, offset_p_metrics_deepatmo_gradh_ifc_d0, offset_p_metrics_deepatmo_gradh_mc_d0, offset_p_metrics_deepatmo_invr_ifc_d0, offset_p_metrics_deepatmo_invr_mc_d0, offset_p_metrics_wgtfac_c_d0, offset_p_metrics_wgtfac_c_d1, offset_p_metrics_wgtfac_c_d2, offset_p_metrics_wgtfac_e_d0, offset_p_metrics_wgtfac_e_d1, offset_p_metrics_wgtfac_e_d2, offset_p_metrics_wgtfacq_e_d0, offset_p_metrics_wgtfacq_e_d1, offset_p_metrics_wgtfacq_e_d2, offset_p_patch_cells_area_d0, offset_p_patch_cells_area_d1, offset_p_patch_cells_decomp_info_owner_mask_d0, offset_p_patch_cells_decomp_info_owner_mask_d1, offset_p_patch_cells_edge_blk_d0, offset_p_patch_cells_edge_blk_d1, offset_p_patch_cells_edge_blk_d2, offset_p_patch_cells_edge_idx_d0, offset_p_patch_cells_edge_idx_d1, offset_p_patch_cells_edge_idx_d2, offset_p_patch_cells_neighbor_blk_d0, offset_p_patch_cells_neighbor_blk_d1, offset_p_patch_cells_neighbor_blk_d2, offset_p_patch_cells_neighbor_idx_d0, offset_p_patch_cells_neighbor_idx_d1, offset_p_patch_cells_neighbor_idx_d2, offset_p_patch_cells_start_block_d0, offset_p_patch_cells_start_index_d0, offset_p_patch_edges_area_edge_d0, offset_p_patch_edges_area_edge_d1, offset_p_patch_edges_cell_blk_d0, offset_p_patch_edges_cell_blk_d1, offset_p_patch_edges_cell_blk_d2, offset_p_patch_edges_cell_idx_d0, offset_p_patch_edges_cell_idx_d1, offset_p_patch_edges_cell_idx_d2, offset_p_patch_edges_f_e_d0, offset_p_patch_edges_f_e_d1, offset_p_patch_edges_fn_e_d0, offset_p_patch_edges_fn_e_d1, offset_p_patch_edges_ft_e_d0, offset_p_patch_edges_ft_e_d1, offset_p_patch_edges_inv_dual_edge_length_d0, offset_p_patch_edges_inv_dual_edge_length_d1, offset_p_patch_edges_inv_primal_edge_length_d0, offset_p_patch_edges_inv_primal_edge_length_d1, offset_p_patch_edges_quad_blk_d0, offset_p_patch_edges_quad_blk_d1, offset_p_patch_edges_quad_blk_d2, offset_p_patch_edges_quad_idx_d0, offset_p_patch_edges_quad_idx_d1, offset_p_patch_edges_quad_idx_d2, offset_p_patch_edges_start_block_d0, offset_p_patch_edges_start_index_d0, offset_p_patch_edges_tangent_orientation_d0, offset_p_patch_edges_tangent_orientation_d1, offset_p_patch_edges_vertex_blk_d0, offset_p_patch_edges_vertex_blk_d1, offset_p_patch_edges_vertex_blk_d2, offset_p_patch_edges_vertex_idx_d0, offset_p_patch_edges_vertex_idx_d1, offset_p_patch_edges_vertex_idx_d2, offset_p_patch_verts_cell_blk_d0, offset_p_patch_verts_cell_blk_d1, offset_p_patch_verts_cell_blk_d2, offset_p_patch_verts_cell_idx_d0, offset_p_patch_verts_cell_idx_d1, offset_p_patch_verts_cell_idx_d2, offset_p_patch_verts_edge_blk_d0, offset_p_patch_verts_edge_blk_d1, offset_p_patch_verts_edge_blk_d2, offset_p_patch_verts_edge_idx_d0, offset_p_patch_verts_edge_idx_d1, offset_p_patch_verts_edge_idx_d2, offset_p_patch_verts_start_block_d0, offset_p_patch_verts_start_index_d0, offset_p_prog_vn_d0, offset_p_prog_vn_d1, offset_p_prog_vn_d2, offset_p_prog_w_d0, offset_p_prog_w_d1, offset_p_prog_w_d2, p_diag_ddt_vn_apc_pc_d0, p_diag_ddt_vn_apc_pc_d1, p_diag_ddt_vn_apc_pc_d2, p_diag_ddt_vn_cor_pc_d0, p_diag_ddt_vn_cor_pc_d1, p_diag_ddt_vn_cor_pc_d2, p_diag_ddt_w_adv_pc_d0, p_diag_ddt_w_adv_pc_d1, p_diag_ddt_w_adv_pc_d2, p_diag_vn_ie_d0, p_diag_vn_ie_d1, p_diag_vn_ie_ubc_d0, p_diag_vn_ie_ubc_d1, p_diag_vt_d0, p_diag_vt_d1, p_diag_w_concorr_c_d0, p_diag_w_concorr_c_d1, p_int_c_lin_e_d0, p_int_c_lin_e_d1, p_int_cells_aw_verts_d0, p_int_cells_aw_verts_d1, p_int_e_bln_c_s_d0, p_int_e_bln_c_s_d1, p_int_geofac_grdiv_d0, p_int_geofac_grdiv_d1, p_int_geofac_n2s_d0, p_int_geofac_n2s_d1, p_int_geofac_rot_d0, p_int_geofac_rot_d1, p_int_rbf_vec_coeff_e_d0, p_int_rbf_vec_coeff_e_d1, p_metrics_coeff1_dwdz_d0, p_metrics_coeff1_dwdz_d1, p_metrics_coeff2_dwdz_d0, p_metrics_coeff2_dwdz_d1, p_metrics_coeff_gradekin_d0, p_metrics_coeff_gradekin_d1, p_metrics_ddqz_z_full_e_d0, p_metrics_ddqz_z_full_e_d1, p_metrics_ddqz_z_half_d0, p_metrics_ddqz_z_half_d1, p_metrics_ddxn_z_full_d0, p_metrics_ddxn_z_full_d1, p_metrics_ddxt_z_full_d0, p_metrics_ddxt_z_full_d1, p_metrics_wgtfac_c_d0, p_metrics_wgtfac_c_d1, p_metrics_wgtfac_e_d0, p_metrics_wgtfac_e_d1, p_metrics_wgtfacq_e_d0, p_metrics_wgtfacq_e_d1, p_patch_cells_area_d0, p_patch_cells_decomp_info_owner_mask_d0, p_patch_cells_edge_blk_d0, p_patch_cells_edge_blk_d1, p_patch_cells_edge_idx_d0, p_patch_cells_edge_idx_d1, p_patch_cells_neighbor_blk_d0, p_patch_cells_neighbor_blk_d1, p_patch_cells_neighbor_idx_d0, p_patch_cells_neighbor_idx_d1, p_patch_edges_area_edge_d0, p_patch_edges_cell_blk_d0, p_patch_edges_cell_blk_d1, p_patch_edges_cell_idx_d0, p_patch_edges_cell_idx_d1, p_patch_edges_f_e_d0, p_patch_edges_fn_e_d0, p_patch_edges_ft_e_d0, p_patch_edges_inv_dual_edge_length_d0, p_patch_edges_inv_primal_edge_length_d0, p_patch_edges_quad_blk_d0, p_patch_edges_quad_blk_d1, p_patch_edges_quad_idx_d0, p_patch_edges_quad_idx_d1, p_patch_edges_tangent_orientation_d0, p_patch_edges_vertex_blk_d0, p_patch_edges_vertex_blk_d1, p_patch_edges_vertex_idx_d0, p_patch_edges_vertex_idx_d1, p_patch_nblks_c, p_patch_nblks_e, p_patch_nblks_v, p_patch_nlev, p_patch_nlevp1, p_patch_nshift, p_patch_verts_cell_blk_d0, p_patch_verts_cell_blk_d1, p_patch_verts_cell_idx_d0, p_patch_verts_cell_idx_d1, p_patch_verts_edge_blk_d0, p_patch_verts_edge_blk_d1, p_patch_verts_edge_idx_d0, p_patch_verts_edge_idx_d1, p_prog_vn_d0, p_prog_vn_d1, p_prog_w_d0, p_prog_w_d1, timers_level, z_kin_hor_e_d0, z_kin_hor_e_d1, z_vt_ie_d0, z_vt_ie_d1, z_w_concorr_me_d0, z_w_concorr_me_d1);
}

DACE_EXPORTED velocity_tendencies_state_t *__dace_init_velocity_tendencies(int istep, int nproma, int ntnd, int64_t offset_p_diag_ddt_vn_apc_pc_d0, int64_t offset_p_diag_ddt_vn_apc_pc_d1, int64_t offset_p_diag_ddt_vn_apc_pc_d2, int64_t offset_p_diag_ddt_vn_apc_pc_d3, int64_t offset_p_diag_ddt_vn_cor_pc_d0, int64_t offset_p_diag_ddt_vn_cor_pc_d1, int64_t offset_p_diag_ddt_vn_cor_pc_d2, int64_t offset_p_diag_ddt_vn_cor_pc_d3, int64_t offset_p_diag_ddt_w_adv_pc_d0, int64_t offset_p_diag_ddt_w_adv_pc_d1, int64_t offset_p_diag_ddt_w_adv_pc_d2, int64_t offset_p_diag_ddt_w_adv_pc_d3, int64_t offset_p_diag_vn_ie_d0, int64_t offset_p_diag_vn_ie_d1, int64_t offset_p_diag_vn_ie_d2, int64_t offset_p_diag_vn_ie_ubc_d0, int64_t offset_p_diag_vn_ie_ubc_d1, int64_t offset_p_diag_vn_ie_ubc_d2, int64_t offset_p_diag_vt_d0, int64_t offset_p_diag_vt_d1, int64_t offset_p_diag_vt_d2, int64_t offset_p_diag_w_concorr_c_d0, int64_t offset_p_diag_w_concorr_c_d1, int64_t offset_p_diag_w_concorr_c_d2, int64_t offset_p_int_c_lin_e_d0, int64_t offset_p_int_c_lin_e_d1, int64_t offset_p_int_c_lin_e_d2, int64_t offset_p_int_cells_aw_verts_d0, int64_t offset_p_int_cells_aw_verts_d1, int64_t offset_p_int_cells_aw_verts_d2, int64_t offset_p_int_e_bln_c_s_d0, int64_t offset_p_int_e_bln_c_s_d1, int64_t offset_p_int_e_bln_c_s_d2, int64_t offset_p_int_geofac_grdiv_d0, int64_t offset_p_int_geofac_grdiv_d1, int64_t offset_p_int_geofac_grdiv_d2, int64_t offset_p_int_geofac_n2s_d0, int64_t offset_p_int_geofac_n2s_d1, int64_t offset_p_int_geofac_n2s_d2, int64_t offset_p_int_geofac_rot_d0, int64_t offset_p_int_geofac_rot_d1, int64_t offset_p_int_geofac_rot_d2, int64_t offset_p_int_rbf_vec_coeff_e_d0, int64_t offset_p_int_rbf_vec_coeff_e_d1, int64_t offset_p_int_rbf_vec_coeff_e_d2, int64_t offset_p_metrics_coeff1_dwdz_d0, int64_t offset_p_metrics_coeff1_dwdz_d1, int64_t offset_p_metrics_coeff1_dwdz_d2, int64_t offset_p_metrics_coeff2_dwdz_d0, int64_t offset_p_metrics_coeff2_dwdz_d1, int64_t offset_p_metrics_coeff2_dwdz_d2, int64_t offset_p_metrics_coeff_gradekin_d0, int64_t offset_p_metrics_coeff_gradekin_d1, int64_t offset_p_metrics_coeff_gradekin_d2, int64_t offset_p_metrics_ddqz_z_full_e_d0, int64_t offset_p_metrics_ddqz_z_full_e_d1, int64_t offset_p_metrics_ddqz_z_full_e_d2, int64_t offset_p_metrics_ddqz_z_half_d0, int64_t offset_p_metrics_ddqz_z_half_d1, int64_t offset_p_metrics_ddqz_z_half_d2, int64_t offset_p_metrics_ddxn_z_full_d0, int64_t offset_p_metrics_ddxn_z_full_d1, int64_t offset_p_metrics_ddxn_z_full_d2, int64_t offset_p_metrics_ddxt_z_full_d0, int64_t offset_p_metrics_ddxt_z_full_d1, int64_t offset_p_metrics_ddxt_z_full_d2, int64_t offset_p_metrics_deepatmo_gradh_ifc_d0, int64_t offset_p_metrics_deepatmo_gradh_mc_d0, int64_t offset_p_metrics_deepatmo_invr_ifc_d0, int64_t offset_p_metrics_deepatmo_invr_mc_d0, int64_t offset_p_metrics_wgtfac_c_d0, int64_t offset_p_metrics_wgtfac_c_d1, int64_t offset_p_metrics_wgtfac_c_d2, int64_t offset_p_metrics_wgtfac_e_d0, int64_t offset_p_metrics_wgtfac_e_d1, int64_t offset_p_metrics_wgtfac_e_d2, int64_t offset_p_metrics_wgtfacq_e_d0, int64_t offset_p_metrics_wgtfacq_e_d1, int64_t offset_p_metrics_wgtfacq_e_d2, int64_t offset_p_patch_cells_area_d0, int64_t offset_p_patch_cells_area_d1, int64_t offset_p_patch_cells_decomp_info_owner_mask_d0, int64_t offset_p_patch_cells_decomp_info_owner_mask_d1, int64_t offset_p_patch_cells_edge_blk_d0, int64_t offset_p_patch_cells_edge_blk_d1, int64_t offset_p_patch_cells_edge_blk_d2, int64_t offset_p_patch_cells_edge_idx_d0, int64_t offset_p_patch_cells_edge_idx_d1, int64_t offset_p_patch_cells_edge_idx_d2, int64_t offset_p_patch_cells_neighbor_blk_d0, int64_t offset_p_patch_cells_neighbor_blk_d1, int64_t offset_p_patch_cells_neighbor_blk_d2, int64_t offset_p_patch_cells_neighbor_idx_d0, int64_t offset_p_patch_cells_neighbor_idx_d1, int64_t offset_p_patch_cells_neighbor_idx_d2, int64_t offset_p_patch_cells_start_block_d0, int64_t offset_p_patch_cells_start_index_d0, int64_t offset_p_patch_edges_area_edge_d0, int64_t offset_p_patch_edges_area_edge_d1, int64_t offset_p_patch_edges_cell_blk_d0, int64_t offset_p_patch_edges_cell_blk_d1, int64_t offset_p_patch_edges_cell_blk_d2, int64_t offset_p_patch_edges_cell_idx_d0, int64_t offset_p_patch_edges_cell_idx_d1, int64_t offset_p_patch_edges_cell_idx_d2, int64_t offset_p_patch_edges_f_e_d0, int64_t offset_p_patch_edges_f_e_d1, int64_t offset_p_patch_edges_fn_e_d0, int64_t offset_p_patch_edges_fn_e_d1, int64_t offset_p_patch_edges_ft_e_d0, int64_t offset_p_patch_edges_ft_e_d1, int64_t offset_p_patch_edges_inv_dual_edge_length_d0, int64_t offset_p_patch_edges_inv_dual_edge_length_d1, int64_t offset_p_patch_edges_inv_primal_edge_length_d0, int64_t offset_p_patch_edges_inv_primal_edge_length_d1, int64_t offset_p_patch_edges_quad_blk_d0, int64_t offset_p_patch_edges_quad_blk_d1, int64_t offset_p_patch_edges_quad_blk_d2, int64_t offset_p_patch_edges_quad_idx_d0, int64_t offset_p_patch_edges_quad_idx_d1, int64_t offset_p_patch_edges_quad_idx_d2, int64_t offset_p_patch_edges_start_block_d0, int64_t offset_p_patch_edges_start_index_d0, int64_t offset_p_patch_edges_tangent_orientation_d0, int64_t offset_p_patch_edges_tangent_orientation_d1, int64_t offset_p_patch_edges_vertex_blk_d0, int64_t offset_p_patch_edges_vertex_blk_d1, int64_t offset_p_patch_edges_vertex_blk_d2, int64_t offset_p_patch_edges_vertex_idx_d0, int64_t offset_p_patch_edges_vertex_idx_d1, int64_t offset_p_patch_edges_vertex_idx_d2, int64_t offset_p_patch_verts_cell_blk_d0, int64_t offset_p_patch_verts_cell_blk_d1, int64_t offset_p_patch_verts_cell_blk_d2, int64_t offset_p_patch_verts_cell_idx_d0, int64_t offset_p_patch_verts_cell_idx_d1, int64_t offset_p_patch_verts_cell_idx_d2, int64_t offset_p_patch_verts_edge_blk_d0, int64_t offset_p_patch_verts_edge_blk_d1, int64_t offset_p_patch_verts_edge_blk_d2, int64_t offset_p_patch_verts_edge_idx_d0, int64_t offset_p_patch_verts_edge_idx_d1, int64_t offset_p_patch_verts_edge_idx_d2, int64_t offset_p_patch_verts_start_block_d0, int64_t offset_p_patch_verts_start_index_d0, int64_t offset_p_prog_vn_d0, int64_t offset_p_prog_vn_d1, int64_t offset_p_prog_vn_d2, int64_t offset_p_prog_w_d0, int64_t offset_p_prog_w_d1, int64_t offset_p_prog_w_d2, int64_t p_diag_ddt_vn_apc_pc_d0, int64_t p_diag_ddt_vn_apc_pc_d1, int64_t p_diag_ddt_vn_apc_pc_d2, int64_t p_diag_ddt_vn_cor_pc_d0, int64_t p_diag_ddt_vn_cor_pc_d1, int64_t p_diag_ddt_vn_cor_pc_d2, int64_t p_diag_ddt_w_adv_pc_d0, int64_t p_diag_ddt_w_adv_pc_d1, int64_t p_diag_ddt_w_adv_pc_d2, int64_t p_diag_vn_ie_d0, int64_t p_diag_vn_ie_d1, int64_t p_diag_vn_ie_ubc_d0, int64_t p_diag_vn_ie_ubc_d1, int64_t p_diag_vt_d0, int64_t p_diag_vt_d1, int64_t p_diag_w_concorr_c_d0, int64_t p_diag_w_concorr_c_d1, int64_t p_int_c_lin_e_d0, int64_t p_int_c_lin_e_d1, int64_t p_int_cells_aw_verts_d0, int64_t p_int_cells_aw_verts_d1, int64_t p_int_e_bln_c_s_d0, int64_t p_int_e_bln_c_s_d1, int64_t p_int_geofac_grdiv_d0, int64_t p_int_geofac_grdiv_d1, int64_t p_int_geofac_n2s_d0, int64_t p_int_geofac_n2s_d1, int64_t p_int_geofac_rot_d0, int64_t p_int_geofac_rot_d1, int64_t p_int_rbf_vec_coeff_e_d0, int64_t p_int_rbf_vec_coeff_e_d1, int64_t p_metrics_coeff1_dwdz_d0, int64_t p_metrics_coeff1_dwdz_d1, int64_t p_metrics_coeff2_dwdz_d0, int64_t p_metrics_coeff2_dwdz_d1, int64_t p_metrics_coeff_gradekin_d0, int64_t p_metrics_coeff_gradekin_d1, int64_t p_metrics_ddqz_z_full_e_d0, int64_t p_metrics_ddqz_z_full_e_d1, int64_t p_metrics_ddqz_z_half_d0, int64_t p_metrics_ddqz_z_half_d1, int64_t p_metrics_ddxn_z_full_d0, int64_t p_metrics_ddxn_z_full_d1, int64_t p_metrics_ddxt_z_full_d0, int64_t p_metrics_ddxt_z_full_d1, int64_t p_metrics_wgtfac_c_d0, int64_t p_metrics_wgtfac_c_d1, int64_t p_metrics_wgtfac_e_d0, int64_t p_metrics_wgtfac_e_d1, int64_t p_metrics_wgtfacq_e_d0, int64_t p_metrics_wgtfacq_e_d1, int64_t p_patch_cells_area_d0, int64_t p_patch_cells_decomp_info_owner_mask_d0, int64_t p_patch_cells_edge_blk_d0, int64_t p_patch_cells_edge_blk_d1, int64_t p_patch_cells_edge_idx_d0, int64_t p_patch_cells_edge_idx_d1, int64_t p_patch_cells_neighbor_blk_d0, int64_t p_patch_cells_neighbor_blk_d1, int64_t p_patch_cells_neighbor_idx_d0, int64_t p_patch_cells_neighbor_idx_d1, int64_t p_patch_edges_area_edge_d0, int64_t p_patch_edges_cell_blk_d0, int64_t p_patch_edges_cell_blk_d1, int64_t p_patch_edges_cell_idx_d0, int64_t p_patch_edges_cell_idx_d1, int64_t p_patch_edges_f_e_d0, int64_t p_patch_edges_fn_e_d0, int64_t p_patch_edges_ft_e_d0, int64_t p_patch_edges_inv_dual_edge_length_d0, int64_t p_patch_edges_inv_primal_edge_length_d0, int64_t p_patch_edges_quad_blk_d0, int64_t p_patch_edges_quad_blk_d1, int64_t p_patch_edges_quad_idx_d0, int64_t p_patch_edges_quad_idx_d1, int64_t p_patch_edges_tangent_orientation_d0, int64_t p_patch_edges_vertex_blk_d0, int64_t p_patch_edges_vertex_blk_d1, int64_t p_patch_edges_vertex_idx_d0, int64_t p_patch_edges_vertex_idx_d1, int p_patch_nblks_c, int p_patch_nblks_e, int p_patch_nblks_v, int p_patch_nlev, int p_patch_nlevp1, int p_patch_nshift, int64_t p_patch_verts_cell_blk_d0, int64_t p_patch_verts_cell_blk_d1, int64_t p_patch_verts_cell_idx_d0, int64_t p_patch_verts_cell_idx_d1, int64_t p_patch_verts_edge_blk_d0, int64_t p_patch_verts_edge_blk_d1, int64_t p_patch_verts_edge_idx_d0, int64_t p_patch_verts_edge_idx_d1, int64_t p_prog_vn_d0, int64_t p_prog_vn_d1, int64_t p_prog_w_d0, int64_t p_prog_w_d1, int timers_level, int64_t z_kin_hor_e_d0, int64_t z_kin_hor_e_d1, int64_t z_vt_ie_d0, int64_t z_vt_ie_d1, int64_t z_w_concorr_me_d0, int64_t z_w_concorr_me_d1)
{

    int __result = 0;
    velocity_tendencies_state_t *__state = new velocity_tendencies_state_t();

    if (__result) {
        delete __state;
        return nullptr;
    }

    if (__result) {
        delete __state;
        return nullptr;
    }

    return __state;
}

DACE_EXPORTED int __dace_exit_velocity_tendencies(velocity_tendencies_state_t *__state)
{

    int __err = 0;
    delete __state;
    return __err;
}
