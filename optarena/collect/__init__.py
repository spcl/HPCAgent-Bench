# Copyright 2021 ETH Zurich and the OptArena authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Batch drivers over the kernel registry.

These are the collection/survey sweeps that used to live as standalone ``scripts/``
entrypoints and are now importable package modules dispatched by the ``optarena`` CLI:

* :mod:`optarena.collect.sweep` -- framework-baseline sweeps that populate the
  ``optarena.db`` results table (``run-benchmark`` sequential, ``run-framework``
  forked-per-kernel, ``run-sparse`` over sparse storage/distribution variants).
* :mod:`optarena.collect.quickstart` -- a tiny fixed-kernel demo sweep.
* :mod:`optarena.collect.pluto_survey` -- the Pluto polyhedral-backend affine survey.

The heavy per-framework imports live inside these modules (not at CLI ``--help`` time);
:mod:`optarena.cli` defers importing them until a subcommand actually runs.
"""
