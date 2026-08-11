---
name: memory
description: Layout and traffic -- AoS/SoA, packing, padding, blocking for reuse.
---

Most kernels at scale are bound by memory traffic, not arithmetic. Count the bytes the
kernel must move, divide by the machine's bandwidth, and compare that to the measured
time -- if they are close, no amount of instruction-level tuning will help and the layout
is the problem.

- **AoS -> SoA.** When a loop touches one field of a struct across many elements, a
  struct-of-arrays layout turns a strided gather into a contiguous stream.
- **Pack** the tiles a blocked kernel reuses into a small contiguous buffer once, then read
  that buffer repeatedly. This is what makes blocked GEMM fast.
- **Pad** leading dimensions to break cache-set conflicts -- a power-of-two stride makes
  many rows collide in the same set. Padding by one element often recovers a large factor.
- **Transpose** when an axis is read column-wise more than it is written row-wise. Pay the
  transpose once, read contiguously many times.
- **Cut intermediates.** A temporary that is written then immediately read is a round trip
  through memory; fusing the producer into the consumer removes it.
- **Reuse.** Raise arithmetic intensity by keeping a block in registers/cache across as
  many uses as possible before evicting it.
