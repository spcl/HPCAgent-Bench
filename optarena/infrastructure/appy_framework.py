# Copyright 2021 ETH Zurich and the OptArena authors.
# SPDX-License-Identifier: GPL-3.0-or-later
import time

from optarena.infrastructure import Framework
from optarena.infrastructure.framework import TimingResult, Timer
from typing import Any, Callable, Dict


class APPyFramework(Framework):
    """ A class for reading and processing framework information. """

    def __init__(self, fname: str):
        """ Reads framework information.
        :param fname: The framework name.
        """

        super().__init__(fname)

    def version(self) -> str:
        """ Return the framework version. """
        return 0.1

    def copy_func(self) -> Callable:
        import torch
        torch.set_default_device('cuda')

        def inner(arr):
            copy = torch.from_numpy(arr).to('cuda')
            return copy

        return inner

    def imports(self) -> Dict[str, Any]:
        import torch
        return {'torch': torch}

    def post_call(self, result: Any) -> Any:
        """Sync the CUDA stream so timing captures the async kernel (replaces
        the ``; torch.cuda.synchronize()`` appended to an exec string)."""
        import torch
        torch.cuda.synchronize()
        return result

    # ----- Native timing via torch CUDA events (device-only kernel time) ---

    def create_timer(self, program) -> Timer:
        """Allocate a start/stop torch CUDA event pair for device-side timing."""
        import torch
        timer = Timer(program)
        timer.state = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        return timer

    def start_timer(self, timer: Timer) -> None:
        timer.t0 = time.perf_counter()
        timer.state[0].record()

    def stop_timer(self, timer: Timer) -> TimingResult:
        """Record + sync the stop event; native = device-measured seconds,
        python = host wall-clock (both bracketing the same call)."""
        import torch
        start_ev, stop_ev = timer.state
        stop_ev.record()
        torch.cuda.synchronize()
        python_t = (time.perf_counter() - timer.t0) * 1.0e3  # s -> ms
        native_t = start_ev.elapsed_time(stop_ev)  # already ms
        return TimingResult(python=python_t, native=native_t)
