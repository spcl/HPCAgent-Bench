# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``model_of`` resolves a setup name to the model that ran it."""

import pytest

from hpcagent_bench import study_tags


@pytest.mark.parametrize(
    ("setup", "expected"),
    [
        ("llr40v10-oss120b-c-skills", "oss120b"),
        ("llr40v9-qwen38-fortran", "qwen38"),
        ("llr40-kimi27sglang-c-cpf-src", "kimi27sglang"),
        ("glm53llr20-glm53-c", "glm53"),
        ("llr4-qwen30b-c", "other"),
        ("", "other"),
        # Dash-bounded tokens: the language ``c`` must not match inside ``kimi27sglang``, nor a
        # model inside a longer word.
        ("llr40-kimi27sglang-c", "kimi27sglang"),
        ("prefix-notglm53here-c", "other"),
        ("llr40-oss120bx-c", "other"),
        ("xoss120b-c", "other"),
        ("something-entirely-new", "other"),  # an unrecognised setup gets the explicit "other"
    ],
)
def test_a_setup_resolves_to_the_model_that_ran_it(setup: str, expected: str) -> None:
    assert study_tags.model_of(setup) == expected


def test_registry_order_decides_when_a_setup_names_two_models() -> None:
    """A malformed setup carrying two model tokens must resolve the same way in every process, so the
    winner is registry order rather than whichever token came first in the string."""
    order = study_tags.order("models")
    first, second = order[0], order[1]
    assert study_tags.model_of(f"x-{second}-{first}-c") == first
    assert study_tags.model_of(f"x-{first}-{second}-c") == first
