# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Turn the repeat-study markers :mod:`repeat_extract` read into values: each blue (Qwen3.8-27B) or orange
(OSS-120B) marker of the figure's two panels to (model, kernel, temperature, measure, value), by the figure's own
axis calibration, and a per-group summary (n, min, median, max, sd of the log).

    python -m helpers.scripts.paper_analysis.repeat_decode <raw.json> <points.csv>"""

import csv
import json
import math
import pathlib
import statistics
import sys
from collections.abc import Sequence

BLUE = (0.122, 0.467, 0.706)
ORANGE = (1.0, 0.498, 0.055)
#: The figure's marker size, and the x centre of each (kernel, temperature) group of the blue model.
MARKER_WIDTH = 1.6
CENTRES = {
    ("QE Aug Charge", 0): 61.9,
    ("KMP Search", 0): 92.1,
    ("Affine Scan", 0): 122.3,
    ("QE Aug Charge", 1): 152.5,
    ("KMP Search", 1): 182.7,
    ("Affine Scan", 1): 212.9,
}
#: The orange model's x offset from the blue one's centre.
ORANGE_OFFSET = 10.27

type Point = tuple[str, str, int, str, float]
#: One marker as repeat_extract writes it: x, y, fill RGB, width.
type Marker = tuple[float, float, Sequence[float], float]


def decode(marks: Sequence[Marker]) -> list[Point]:
    """The points of ``marks``: the speedup panel above y = 90 (log2, 3 decades per 12.285 pt), the token panel below
    y = 150 (log2 of tokens / 1e5, 7.67 pt per doubling)."""
    rows: list[Point] = []
    for x, y, colour, width in marks:
        rgb = tuple(colour)
        if width != MARKER_WIDTH or rgb not in (BLUE, ORANGE):
            continue
        model, offset = ("Qwen3.8-27B", 0.0) if rgb == BLUE else ("OSS-120B", ORANGE_OFFSET)
        kernel, temperature = min(CENTRES, key=lambda group: abs(x - (CENTRES[group] + offset)))
        if y < 90:
            rows.append((model, kernel, temperature, "speedup", 2 ** ((51.1743 - y) * 3 / 12.2850)))
        elif y < 150:
            rows.append((model, kernel, temperature, "tokens", 1e5 * 2 ** ((131.7902 - y) / 7.67057)))
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    raw, out = (pathlib.Path(arg) for arg in (sys.argv[1:] if argv is None else argv))
    rows = decode(json.loads(raw.read_text(encoding="utf-8"))["marks"])
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "kernel", "temperature", "measure", "value"])
        writer.writerows(rows)
    groups: dict[tuple[str, str, int, str], list[float]] = {}
    for model, kernel, temperature, measure, value in rows:
        groups.setdefault((model, kernel, temperature, measure), []).append(value)
    print("model,kernel,T,measure,n,min,median,max,sd_log")
    for key, values in sorted(groups.items()):
        logs = [math.log(value) for value in values]
        spread = statistics.stdev(logs) if len(logs) > 1 else 0.0
        print(
            *key,
            len(values),
            f"{min(values):.3g}",
            f"{statistics.median(values):.3g}",
            f"{max(values):.3g}",
            f"{spread:.3f}",
            sep=",",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
