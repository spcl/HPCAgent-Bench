# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The figure commands' command line: their parser (:class:`Parser`) and a closed option as a flag
(:func:`add_argument`, parsed to its :class:`~hpcagent_agent.coded.Coded` member)."""

import argparse
import collections
import itertools
import sys
from collections.abc import Sequence
from typing import Any

from hpcagent_agent.coded import Coded

__all__ = ["Parser", "add_argument"]


class Parser(argparse.ArgumentParser):
    """A figure or table command's parser: named ``prog`` in its usage (``hpcagent-bench plot <figure>``), flags
    spelled in full, and an option holding one value refused when given twice (argparse alone keeps the last and
    says nothing)."""

    def __init__(self, prog: str, description: str | None) -> None:
        self.single: set[str] = set()
        super().__init__(
            prog=prog,
            description=description,
            formatter_class=argparse.RawDescriptionHelpFormatter,
            allow_abbrev=False,
        )

    def add_argument(self, *args: Any, **kwargs: Any) -> argparse.Action:  # noqa: ANN401 -- argparse's own signature
        action = super().add_argument(*args, **kwargs)
        if kwargs.get("action") != "append":
            self.single.update(action.option_strings)
        return action

    def parse_known_args(  # type: ignore[override]  # the stub overloads on the namespace's type
        self,
        args: Sequence[str] | None = None,
        namespace: Any = None,  # noqa: ANN401 -- argparse's own signature
    ) -> tuple[Any, list[str]]:
        given = sys.argv[1:] if args is None else list(args)
        flags = (arg.partition("=")[0] for arg in itertools.takewhile(lambda arg: arg != "--", given))
        for flag, count in collections.Counter(flag for flag in flags if flag in self.single).items():
            if count > 1:
                self.error(f"{flag} given {count} times; it takes one value")
        return super().parse_known_args(given, namespace)


def add_argument[Choice: Coded](
    parser: argparse.ArgumentParser, flag: str, kind: type[Choice], default: Choice, help_text: str
) -> None:
    """Add ``flag`` to ``parser``, read as a member of ``kind``; the parsed value is the member itself."""

    def read(text: str) -> Choice:
        try:
            return kind.parse(text)
        except ValueError as unknown:
            raise argparse.ArgumentTypeError(str(unknown)) from None

    parser.add_argument(
        flag, type=read, choices=tuple(kind), default=default, metavar="{" + ",".join(kind.texts()) + "}",
        help=help_text,
    )  # fmt: skip
