# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Steps several figure modules share, split out of their drawing functions: one module per topic
(``series``: a series' colour and shape from the registry; ``layout``: canvas and chrome; ``axes``:
axis scales and ticks; ``legend``: keys). A helper used by one figure module stays in that module."""
