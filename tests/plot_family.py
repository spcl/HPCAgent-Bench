# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Shared helper for tests that assert on a rendered plot file."""
import pathlib


def one_plot(directory: pathlib.Path, output_name: str) -> pathlib.Path:
    """The single file that ``plot --output <output_name>`` wrote.

    Plots are PARTITIONED per machine and named ``<stem>.<cpu>[-<gpu>]<ext>``, so ``--output``
    names a family rather than a file -- two nodes' rows may never share a figure. A sweep records
    one machine, so exactly one member exists; asserting that is a stronger check than rebuilding
    the machine label here, which would just duplicate ``plotting.machine_label``.
    """
    stem = pathlib.Path(output_name)
    found = sorted(directory.glob(f"{stem.stem}.*{stem.suffix}"))
    assert len(found) == 1, f"expected exactly one {output_name} family member in {directory}, got {found}"
    return found[0]
