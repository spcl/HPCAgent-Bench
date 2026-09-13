# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A judge refusal surfaces through ``JudgeClient`` as an ``HTTPError`` that names the judge's reason,
keeps its body readable and holds no open response, so a caller that only inspects the code leaks
nothing."""

import email.message
import io
import json
import urllib.error

import pytest

from hpcagent_bench.harness.tools import JudgeRefusal, error_with_body

BODY = json.dumps({"error": "min_percent must be between 0 and 100, got -1.0"}).encode()


def raw_refusal(code: int = 400, body: bytes = BODY) -> urllib.error.HTTPError:
    """What ``urllib.request.urlopen`` raises for a judge that answered ``code`` with ``body``."""
    return urllib.error.HTTPError(
        "http://judge/profile", code, "Bad Request", email.message.Message(), io.BytesIO(body)
    )


def test_the_refusal_message_carries_the_judges_reason() -> None:
    """A bare 'HTTP Error 400: Bad Request' in a traceback says nothing an agent can act on."""
    refusal = error_with_body(raw_refusal())
    assert "min_percent must be between 0 and 100" in str(refusal), str(refusal)


def test_the_refusal_keeps_the_status_and_stays_an_http_error() -> None:
    refusal = error_with_body(raw_refusal(503))
    assert isinstance(refusal, urllib.error.HTTPError)
    assert refusal.code == 503


def test_the_body_reads_back_whole_on_every_read() -> None:
    """Callers read the body to decode the cause; a second reader must not get an empty string."""
    refusal = error_with_body(raw_refusal())
    assert refusal.read() == BODY
    assert json.loads(refusal.read())["error"].startswith("min_percent")


@pytest.mark.parametrize("amt, expected", [(None, BODY), (-1, BODY), (0, b""), (5, BODY[:5])])
def test_a_bounded_read_serves_a_prefix_of_the_body(amt: int | None, expected: bytes) -> None:
    assert error_with_body(raw_refusal()).read(amt) == expected


def test_neither_the_original_response_nor_the_refusal_is_left_open() -> None:
    """An open response is collected with a ResourceWarning, and warnings are errors in this suite."""
    original = raw_refusal()
    refusal = error_with_body(original)
    assert original.closed and refusal.closed


def test_an_empty_refusal_body_is_kept_empty_not_broken() -> None:
    refusal = error_with_body(raw_refusal(404, b""))
    assert (refusal.code, refusal.read()) == (404, b"")


def test_a_refusal_built_directly_is_closed_and_readable() -> None:
    refusal = JudgeRefusal("http://judge/oracle", 421, "Misdirected", email.message.Message(), b"rank")
    assert refusal.closed and refusal.read() == b"rank"
