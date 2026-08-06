<!--
Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
SPDX-License-Identifier: GPL-3.0-or-later
-->
# SeisSol ADER-DG microkernels -- references & provenance

These two HPCAgent-Bench microkernels (`seissol_batched_gemm`,
`seissol_tensor_contraction`) are the element-local operators of **SeisSol**'s
ADER-DG seismic-wave solver. This file gives the rigorous, verified bibliography
for the kernel-generation and ADER-DG performance work they are drawn from.

## Provenance & licensing

- **SeisSol** -- github.com/SeisSol/SeisSol -- **BSD-3-Clause**. The kernel shapes,
  the elastic quantity ordering (6 stresses + 3 velocities), and the **real
  `star` sparsity pattern** (24 nonzeros, the stress<->velocity coupling) are
  transcribed from `codegen/matrices/star.xml`; the order-7 `kDivM` (stiffness x
  inverse-mass) sparsity from `codegen/matrices/matrices_84.xml`.
- **yateto** -- github.com/ThrudPrimrose/yateto -- **BSD-3-Clause**. The
  batched-tiny-GEMM framing and the loop-over-GEMM / sparse-tensor decomposition
  of the volume contraction.
- **gemmforge / TensorForge / chainforge** -- github.com/SeisSol/{gemmforge,
  TensorForge, chainforge}. The SeisSol GPU code generators for batched
  small-matrix GEMM and tensor contraction.
- The HPCAgent-Bench numpy ports + generators in this directory are **original**, under
  **GPL-3.0-or-later** (the HPCAgent-Bench license). No SeisSol/yateto source is copied;
  only the (factual) sparsity patterns and shapes are reproduced.

## Code generators & batched-GEMM kernels

**[1] Dorozhinskii, Brito Gadeschi & Bader (2024) -- fused batched GEMM (chainforge).**
Ravil Dorozhinskii, Gonzalo Brito Gadeschi, Michael Bader. *Fused GEMMs towards
an efficient GPU implementation of the ADER-DG method in SeisSol.* Concurrency and
Computation: Practice and Experience **36**(12), Article e8037, 2024.
DOI: [10.1002/cpe.8037](https://doi.org/10.1002/cpe.8037).
-> The state-of-the-art for the *batched small-matrix GEMM* at the heart of the
ADER-DG element update: fuses chains of subsequent batched GEMMs into one
shared-memory-resident GPU kernel. The direct method reference for
`seissol_batched_gemm`. (Electronic article; e8037 is the canonical locator, no
page range.)

**[2] Uphoff & Bader (2020) -- yateto.**
Carsten Uphoff, Michael Bader. *Yet Another Tensor Toolbox for Discontinuous
Galerkin Methods and Other Applications.* ACM Transactions on Mathematical
Software (TOMS) **46**(4), Article 34, 34:1-34:40, Oct. 2020.
DOI: [10.1145/3406835](https://doi.org/10.1145/3406835).
-> The tensor-contraction code-generation backend of SeisSol; compiles
Einstein-convention contractions into optimized small-matrix kernels. The direct
reference for the `'dkl,blq,dqp->bkp'` decomposition in
`seissol_tensor_contraction`.

**[3] Dorozhinskii & Bader (2021) -- CUDA codegen (gemmforge loop_level_reasoning).**
Ravil Dorozhinskii, Michael Bader. *SeisSol on Distributed Multi-GPU Systems:
CUDA Code Generation for the Modal Discontinuous Galerkin Method.* HPC Asia 2021
(Int. Conf. on High Performance Computing in Asia-Pacific Region), pp. 69-82, 2021.
DOI: [10.1145/3432261.3436753](https://doi.org/10.1145/3432261.3436753).
-> Foundational GPU generation of batched small matrix-multiply kernels (~2.5x over
cuBLAS batched); the citable companion to the **gemmforge** repository.

**[4] gemmforge / TensorForge / chainforge -- software.**
Ravil Dorozhinskii et al. (TU München / SeisSol project).
github.com/SeisSol/gemmforge (batched GEMM generator),
github.com/SeisSol/TensorForge (batched tensor-contraction generator),
github.com/SeisSol/chainforge (fused GEMM chains; see [1]).
-> The code-generation toolchain producing the GPU forms of exactly these two
kernels. Cite [1] and [3] for the underlying methods.

## SeisSol extreme-scale performance (the SIMD small-matrix kernels)

**[5] Heinecke et al. (2014) -- SC'14 (Gordon Bell finalist).**
A. Heinecke, A. Breuer, S. Rettenberger, M. Bader, A.-A. Gabriel, C. Pelties,
A. Bode, W. Barth, X.-K. Liao, K. Vaidyanathan, M. Smelyanskiy, P. Dubey.
*Petascale High Order Dynamic Rupture Earthquake Simulations on Heterogeneous
Supercomputers.* SC '14, pp. 3-14, 2014.
DOI: [10.1109/SC.2014.6](https://doi.org/10.1109/SC.2014.6).
-> Introduces the SIMD-vectorized batched small-matrix (sparse/dense) GEMM
microkernels (the LIBXSMM lineage) at the core of SeisSol's element-local operators.

**[6] Uphoff et al. (2017) -- SC'17 (Best Paper).**
C. Uphoff, S. Rettenberger, M. Bader, E. H. Madden, T. Ulrich, S. Wollherr,
A.-A. Gabriel. *Extreme scale multi-physics simulations of the tsunamigenic 2004
Sumatra megathrust earthquake.* SC '17, Article 21, 2017.
DOI: [10.1145/3126908.3126948](https://doi.org/10.1145/3126908.3126948).
-> The same vectorized batched small-matrix GEMM ADER-DG kernels at full-machine
scale with clustered local time-stepping.

**[7] Breuer et al. (2014) -- ISC'14, SuperMUC petascale.**
A. Breuer, A. Heinecke, S. Rettenberger, M. Bader, A.-A. Gabriel, C. Pelties.
*Sustained Petascale Performance of Seismic Simulations with SeisSol on SuperMUC.*
Supercomputing -- ISC 2014, LNCS **8488**, pp. 1-18, Springer, 2014.
DOI: [10.1007/978-3-319-07518-1_1](https://doi.org/10.1007/978-3-319-07518-1_1).
-> Production demonstration of code-generated SIMD small-matrix kernels driving
ADER-DG at petascale.

## ADER-DG method & small/sparse-matrix kernel foundations

**[8] Käser & Dumbser (2006) -- ADER-DG seismic, Part I (2D).**
M. Käser, M. Dumbser. *An arbitrary high-order discontinuous Galerkin method for
elastic waves on unstructured meshes -- I. The two-dimensional isotropic case with
external source terms.* Geophysical Journal International **166**(2), pp. 855-877,
2006. DOI: [10.1111/j.1365-246X.2006.03051.x](https://doi.org/10.1111/j.1365-246X.2006.03051.x).
-> The foundational ADER-DG scheme SeisSol implements; element updates as products
of small per-element matrices.

**[9] Dumbser & Käser (2006) -- ADER-DG seismic, Part II (3D).**
M. Dumbser, M. Käser. *An arbitrary high-order discontinuous Galerkin method for
elastic waves on unstructured meshes -- II. The three-dimensional isotropic case.*
Geophysical Journal International **167**(1), pp. 319-336, 2006.
DOI: [10.1111/j.1365-246X.2006.03120.x](https://doi.org/10.1111/j.1365-246X.2006.03120.x).
-> The 3D extension whose dense/sparse element operators (stiffness `kDivM`, flux
Jacobian `star`, ADER recurrence) are exactly the small matrices these kernels use.

**[10] Breuer, Heinecke, Bader & Pelties (2014) -- sparse-matrix vectorized codegen.**
A. Breuer, A. Heinecke, M. Bader, C. Pelties. *Accelerating SeisSol by Generating
Vectorized Code for Sparse Matrix Operators.* Parallel Computing: Accelerating
Computational Science and Engineering (ParCo 2013), Advances in Parallel Computing
**25**, pp. 347-356, IOS Press, 2014.
DOI: [10.3233/978-1-61499-381-0-347](https://doi.org/10.3233/978-1-61499-381-0-347).
-> The sparse counterpart of batched small-matrix GEMM: generates per-pattern
vectorized code for SeisSol's a-priori-known sparse element matrices -- the
rationale for keeping the real `star` / `kDivM` sparsity in these microkernels.

**[11] Breuer, Heinecke, Rannabauer & Bader (2015) -- energy/time-to-solution.**
A. Breuer, A. Heinecke, L. Rannabauer, M. Bader. *High-Order ADER-DG Minimizes
Energy- and Time-to-Solution of SeisSol.* High Performance Computing -- ISC 2015,
LNCS **9137**, pp. 340-357, Springer, 2015.
DOI: [10.1007/978-3-319-20119-1_25](https://doi.org/10.1007/978-3-319-20119-1_25).
-> Quantifies node-level gains from dense/sparse small-matrix kernel optimization
as the scheme order rises (2->7) -- the basis for choosing **order 7** as the
headline instance here.

## Related (not peer-reviewed)

- Y. K. Budanaz, *Improved GPU Kernel Generation for SeisSol using Loop-over-GEMM
  and Sparse-Matrix Operations* -- TUM SCCS colloquium talk in this forge
  ecosystem; the reference document the HPCAgent-Bench backlog cites for the order-6
  example shapes (56x9*9x9) and the loop-over-GEMM rationale. Not a peer-reviewed
  paper, listed for provenance only.

## Verification notes

Confirmed via a named primary/authoritative source (publisher page, DOI resolver,
dblp, dl.acm.org, Oxford Academic, IOS Press, TUM repositories):
[1] Crossref + TUM FIS (full author order, vol 36/12, e8037);
[2] dblp + ACM DL (vol 46/4, Article 34, 34:1-34:40);
[3] TUM FIS / ACM DL (pp. 69-82);
[5] dblp `conf/sc/HeineckeBRBGPBBLVSD14` + IEEE/ACM (12 authors, pp. 3-14);
[6] dblp + ACM DL (7 authors, Article 21);
[7] TUM FIS (LNCS 8488, pp. 1-18);
[8] Oxford Academic (166/2/855-877);
[9] Oxford Academic (167/1/319-336);
[10] IOS Press (pp. 347-356);
[11] TUM FIS (LNCS 9137, pp. 340-357);
[4] repos confirmed to exist.

Flagged / uncertain:
- **[1]** No conventional page range (electronic article; e8037 is the locator).
- **[6]** Article number 21 confirmed; the often-quoted "16 pages" length is
  unverified.
- A standalone paper "*Outsmarting the compiler...*" sometimes attributed to the
  forge generators **could not be located** -- do **not** cite it; [1] and [3] are
  the citable forge publications.
- **[10]** Conference ParCo 2013; proceedings published 2014 (year is the only
  soft field).
- Author-order corrections vs. the request: **[7]** is led by *Breuer* (not
  Heinecke); **[11]**'s third author is *Rannabauer* (not Käser). The CPE 2024
  title is "*Fused GEMMs towards an efficient GPU implementation of the ADER-DG
  method in SeisSol*".
