"""Helpers shared by several backend emitters (C, Fortran, DaCe, Numba).

One module per concern:

* :mod:`.tokens` -- identifier scanning of shape-token strings.
* :mod:`.numpy_names` -- recognising ``np.<attr>`` references.
* :mod:`.fftw` -- decoding the FFT library markers the lowering leaves for the native backends.
* :mod:`.pinned` -- the element dtype of each manifest-pinned config knob.
* :mod:`.request` -- the emit request and the shared naming and precision steps.
"""
