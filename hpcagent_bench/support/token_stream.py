# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Input streams made of variable-length tokens, for the finite-state kernels' initializers.

A Huffman bit stream, a UTF-8 byte stream and a CSV character stream are each a sequence of tokens
(a code word, an encoded code point, a field) drawn from a small table. :func:`concatenated` lays
the drawn tokens end to end with whole-array operations only, so it runs unchanged on numpy and cupy
and gives the same elements on both.
"""

from types import ModuleType
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = ["concatenated"]


def concatenated(
    tokens: NDArray[Any], lengths: NDArray[Any], ids: NDArray[Any], n: int, xp: ModuleType = np
) -> NDArray[Any]:
    """The first ``n`` elements of ``tokens[ids[0], :lengths[ids[0]]]``, then ``tokens[ids[1], ...]``, and so on.

    :param tokens: one token per row, padded to the longest.
    :param lengths: the length of each row's token, at least 1.
    :param ids: the drawn token sequence; it must cover ``n`` elements.
    :param n: the stream length.
    :param xp: the array module of ``tokens``, ``lengths`` and ``ids``.
    """
    lens = lengths[ids]
    ends = xp.cumsum(lens)
    if int(ends[-1]) < n:
        raise ValueError(f"concatenated: {len(ids)} tokens cover {int(ends[-1])} elements, {n} requested")
    kept = int(xp.searchsorted(ends, n)) + 1
    ids, lens, ends = ids[:kept], lens[:kept], ends[:kept]
    owner = xp.repeat(xp.arange(kept, dtype=np.int64), lens)[:n]
    offset = xp.arange(n, dtype=np.int64) - (ends - lens)[owner]
    return tokens[ids[owner], offset].astype(np.int64)
