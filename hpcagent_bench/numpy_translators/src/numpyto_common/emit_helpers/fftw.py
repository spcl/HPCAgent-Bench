"""Decoding of the FFT library markers (``lib_nodes.FFT_LIBRARY_MARKER`` / ``FFTN_LIBRARY_MARKER``).

The lowering leaves a whole-array ``np.fft.*`` as a marker call; each native backend renders it as
FFTW3 plan/execute/destroy. FFTW is unnormalized in both directions, so the backend applies numpy's
``norm=`` divisor itself: :attr:`FftTransform.divides` says whether one is due and
:attr:`FftTransform.ortho` whether it is ``sqrt(n)`` rather than ``n``.
"""

import ast
import dataclasses
from typing import Any, cast

#: ``norm_kind`` encoding in the marker: 0 ``backward``, 1 ``forward``, 2 ``ortho``.
NORM_ORTHO = 2


@dataclasses.dataclass(frozen=True, slots=True)
class FftTransform:
    """The operands and direction shared by the 1-D and the N-D marker."""

    out: str
    src: str
    inverse: bool
    norm_kind: int

    @property
    def sign(self) -> str:
        return "FFTW_BACKWARD" if self.inverse else "FFTW_FORWARD"

    @property
    def ortho(self) -> bool:
        return self.norm_kind == NORM_ORTHO

    @property
    def divides(self) -> bool:
        """Whether numpy scales this direction: always under ortho, else the inverse under
        ``backward`` and the forward under ``forward``."""
        return self.ortho or (self.norm_kind == 0) == self.inverse


@dataclasses.dataclass(frozen=True, slots=True)
class FftNd:
    """The N-D marker: ``n_axes`` transform axes, LEADING or trailing the batch axes."""

    transform: FftTransform
    n_axes: int
    leading: bool
    extents: list[ast.expr]


def fftw_prefix(single: bool) -> str:
    """``fftwf`` for single precision, ``fftw`` for double."""
    return "fftwf" if single else "fftw"


def marker_constant(node: ast.expr) -> Any:
    """A literal marker argument (the lowering only ever writes constants there)."""
    return cast(ast.Constant, node).value


def marker_operands(node: ast.Call) -> tuple[str, str]:
    """``(out, src)``: the marker's first two arguments are always bare array names."""
    out, src = (cast(ast.Name, arg).id for arg in node.args[:2])
    return out, src


def fft_1d(node: ast.Call) -> tuple[FftTransform, ast.expr]:
    """``marker(out, src, n, inverse, norm_kind)`` -> the transform and the length expression."""
    out, src = marker_operands(node)
    inverse, norm_kind = bool(marker_constant(node.args[3])), marker_constant(node.args[4])
    return FftTransform(out, src, inverse, norm_kind), node.args[2]


def fft_nd(node: ast.Call) -> FftNd:
    """``marker(out, src, inverse, norm_kind, n_axes, leading, *extents)``."""
    out, src = marker_operands(node)
    args = [marker_constant(arg) for arg in node.args[2:6]]
    transform = FftTransform(out, src, bool(args[0]), args[1])
    return FftNd(transform, int(args[2]), bool(args[3]), list(node.args[6:]))
