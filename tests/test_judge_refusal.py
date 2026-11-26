# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A judge refusal surfaces through ``JudgeClient`` as an ``HTTPError`` that names the judge's reason,
keeps its body readable and holds no open response, so a caller that only inspects the code leaks
nothing."""

import email.message
import io
import json
import urllib.error

import pytest

from hpcagent_bench.harness.tools import error_with_body

BODY = json.dumps({"error": "min_percent must be between 0 and 100, got -1.0"}).encode()


def raw_refusal(code: int = 400, body: bytes = BODY) -> urllib.error.HTTPError:
    """What ``urllib.request.urlopen`` raises for a judge that answered ``code`` with ``body``."""
    return urllib.error.HTTPError(
        "http://judge/profile", code, "Bad Request", email.message.Message(), io.BytesIO(body)
    )


def test_the_refusal_carries_the_judges_reason_reads_back_whole_and_leaves_nothing_open() -> None:
    """A bare 'HTTP Error 400: Bad Request' in a traceback says nothing an agent can act on. Callers read the
    body to decode the cause, so a second reader must not get an empty string. An open response is collected
    with a ResourceWarning, and warnings are errors in this suite."""
    original = raw_refusal()
    refusal = error_with_body(original)
    assert "min_percent must be between 0 and 100" in str(refusal), str(refusal)
    assert refusal.read() == BODY
    assert json.loads(refusal.read())["error"].startswith("min_percent")
    assert original.closed
    assert refusal.closed
    # It keeps the status and stays an HTTP error.
    assert isinstance(refusal, urllib.error.HTTPError)
    assert error_with_body(raw_refusal(503)).code == 503


@pytest.mark.parametrize(("amt", "expected"), [(None, BODY), (-1, BODY), (0, b""), (5, BODY[:5])])
def test_a_bounded_read_serves_a_prefix_of_the_body(amt: int | None, expected: bytes) -> None:
    assert error_with_body(raw_refusal()).read(amt) == expected
