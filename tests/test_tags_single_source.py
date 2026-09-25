# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Every tag has one source: experiments/tags.yaml never redefines or aliases away a label the
manifests carry in ``experiment_tags``."""

from hpcagent_bench import tags
from hpcagent_bench.spec import KERNELS, BenchSpec


def manifest_labels() -> set[str]:
    return {label.lower() for key in KERNELS for label in BenchSpec.load(key).experiment_tags}


def test_no_tags_yaml_name_is_a_manifest_label() -> None:
    tags.registry.cache_clear()
    registry = tags.registry()
    labels = manifest_labels()
    assert labels, "no manifest carries an experiment_tags value -- the scan regressed"
    clashes = sorted((set(registry.tags) | set(registry.aliases)) & labels)
    assert not clashes, f"experiments/tags.yaml redefines manifest experiment_tags labels: {clashes}"
