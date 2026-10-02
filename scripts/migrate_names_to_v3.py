#!/usr/bin/env python3
# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Move a results database from schema v2 to v3, in place, in one transaction.

v3 renames the stored names to the vocabulary of docs/concepts.md and folds the old setup and study spellings
the code no longer reads:

* table ``runs`` becomes ``episodes`` (index ``runs_key`` becomes ``episodes_key``), ``grades.run_id`` becomes
  ``episode_id`` (index ``grades_run`` becomes ``grades_episode``);
* column ``benchmark`` becomes ``kernel`` in ``episodes`` (was ``runs``), ``grades`` and
  ``reference_scaling_points``;
* the view ``grades_flat`` is recreated over the new names;
* every setup name is rewritten to the setup it is (the fold the retired ``setup_renames.yaml``,
  ``setup_aliases`` and the llrblind -> llrblind-cmp prefix applied at read time): ``setups.setup``,
  ``episodes.setup`` and the setup prefix of ``episodes.label``. Two spellings of one setup merge into one
  ``setups`` row, which needs equal identity columns; a merge that would lose a distinction, or two episodes that
  would collide on ``(job, label, rep)``, refuses the file;
* the CPF archive's ``cpf-llr-focus40-*`` setups become ``llr40-*``, and every ``<setup>-clean`` setup folds into
  ``<setup>`` (the clean suffix is retired); kernels graded under both a clean and a plain spelling are listed;
* ``setups.language`` values an older submitter corrupted with a baked-in packet token or ``-clean`` are repaired
  to the bare language (the packet moves to ``setups.packet`` when that is empty);
* ``setups.study`` is rewritten through the retired study aliases (``mlscale`` -> ``mlscale20`` included);
* ``grades.timing_reduction`` ``mw4x5-final-v2`` becomes ``mw4x5``;
* ``PRAGMA user_version`` goes 2 -> 3.

Before the commit the migrated file is checked: integrity_check, foreign_key_check, the table and column lists
against hpcagent_bench/harness/schema.sql, the row counts, a SHA-256 over every table whose content must not
change (rows in rowid order, columns mapped), and over ``episodes`` without ``label`` and ``setup``. Any difference
rolls the file back untouched. A v3 file is reported and left alone, so a second run changes nothing.

    migrate_names_to_v3.py PATH [PATH ...]        migrate files; a directory is walked for *.db, *.sqlite
    migrate_names_to_v3.py --dry-run PATH ...     do everything, print every change, then roll back
    migrate_names_to_v3.py --verify ORIGINAL NEW  compare a migrated copy against the original it came from

Exit status: 0 when every file migrated, was already migrated or verified equal; 2 when any file was refused or
differed.
"""

import argparse
import dataclasses
import hashlib
import pathlib
import re
import sqlite3
import sys
from collections.abc import Iterator, Sequence

SCHEMA_OLD = 2
SCHEMA_NEW = 3
SCHEMA_SQL = pathlib.Path(__file__).resolve().parents[1] / "hpcagent_bench" / "harness" / "schema.sql"
OLD_TABLES = frozenset({"setups", "runs", "sources", "grades"})
NEW_TABLES = frozenset({"setups", "episodes", "sources", "grades"})
TABLE_RENAMES: dict[str, str] = {"runs": "episodes"}
#: (table before the rename, old column) -> new column.
COLUMN_RENAMES: dict[tuple[str, str], str] = {
    ("runs", "benchmark"): "kernel",
    ("grades", "run_id"): "episode_id",
    ("grades", "benchmark"): "kernel",
    ("reference_scaling_points", "benchmark"): "kernel",
}
INDEX_RENAMES = (("grades_run", "grades_episode"), ("runs_key", "episodes_key"))
#: Tables whose rows must be byte-for-byte the same after the rename (columns mapped).
UNCHANGED_TABLES = (
    "sources",
    "grades",
    "grade_sources",
    "grade_cells",
    "scaling_grades",
    "scaling_points",
    "disqualifications",
    "reference_scaling_points",
)
GRADES_FLAT = """CREATE VIEW grades_flat AS
SELECT a.study, a.model, a.language, a.device, a.packet, a.harness, r.setup, r.job, r.label, r.rep, g.*
FROM grades AS g
JOIN episodes AS r ON r.id = g.episode_id
JOIN setups AS a ON a.setup = r.setup"""
SUFFIXES = (".db", ".sqlite")

#: The setup name each recorded spelling was read as (envs/setup_renames.yaml, retired with this migration).
SETUP_RENAMES: dict[str, str] = {
    "git-scicomp-kimi27sglang-kernel": "gitscicomp10-kimi27sglang-c",
    "git-scicomp-kimi27sglang-kernel-clean": "gitscicomp10-kimi27sglang-c",
    "git-scicomp-kimi27sglang-repo": "gitscicomp10-kimi27sglang-c-repo",
    "git-scicomp-kimi27sglang-repo-clean": "gitscicomp10-kimi27sglang-c-repo",
    "git-scicomp-oss120b-kernel": "gitscicomp10-oss120b-c",
    "git-scicomp-oss120b-repo": "gitscicomp10-oss120b-c-repo",
    "git-scicomp-qwen38-kernel": "gitscicomp10-qwen38-c",
    "git-scicomp-qwen38-repo": "gitscicomp10-qwen38-c-repo",
    "gpu-llr-focus40-kimi27sglang-c-openmp": "llr40-kimi27sglang-c-openmp",
    "gpu-llr-focus40-kimi27sglang-c-openmp-clean": "llr40-kimi27sglang-c-openmp",
    "gpu-llr-focus40-kimi27sglang-c-openmp-device-clean": "llr40-kimi27sglang-c-openmp-device",
    "gpu-llr-focus40-kimi27sglang-c-openmp-device-skills-clean": "llr40-kimi27sglang-c-openmp-device-skills",
    "gpu-llr-focus40-kimi27sglang-c-openmp-skills-clean": "llr40-kimi27sglang-c-openmp-skills",
    "gpu-llr-focus40-kimi27sglang-hip": "llr40-kimi27sglang-hip",
    "gpu-llr-focus40-kimi27sglang-hip-clean": "llr40-kimi27sglang-hip",
    "gpu-llr-focus40-kimi27sglang-hip-perf-playbook-amd-clean": "llr40-kimi27sglang-hip-perf-playbook-amd",
    "gpu-llr-focus40-kimi27sglang-hip-skills": "llr40-kimi27sglang-hip-skills",
    "gpu-llr-focus40-kimi27sglang-hip-skills-clean": "llr40-kimi27sglang-hip-skills",
    "gpu-llr-focus40-kimi27sglang-triton": "llr40-kimi27sglang-triton",
    "gpu-llr-focus40-kimi27sglang-triton-clean": "llr40-kimi27sglang-triton",
    "gpu-llr-focus40-kimi27sglang-triton-device-clean": "llr40-kimi27sglang-triton-device",
    "gpu-llr-focus40-kimi27sglang-triton-device-skills-clean": "llr40-kimi27sglang-triton-device-skills",
    "gpu-llr-focus40-kimi27sglang-triton-skills": "llr40-kimi27sglang-triton-skills",
    "gpu-llr-focus40-kimi27sglang-triton-skills-clean": "llr40-kimi27sglang-triton-skills",
    "gpu-llr-focus40-oss120b-c-openmp": "llr40-oss120b-c-openmp",
    "gpu-llr-focus40-oss120b-c-openmp-clean": "llr40-oss120b-c-openmp",
    "gpu-llr-focus40-oss120b-c-openmp-device-clean": "llr40-oss120b-c-openmp-device",
    "gpu-llr-focus40-oss120b-c-openmp-device-skills-clean": "llr40-oss120b-c-openmp-device-skills",
    "gpu-llr-focus40-oss120b-c-openmp-skills": "llr40-oss120b-c-openmp-skills",
    "gpu-llr-focus40-oss120b-c-openmp-skills-clean": "llr40-oss120b-c-openmp-skills",
    "gpu-llr-focus40-oss120b-hip": "llr40-oss120b-hip",
    "gpu-llr-focus40-oss120b-hip-caveman-clean": "llr40-oss120b-hip-caveman",
    "gpu-llr-focus40-oss120b-hip-clean": "llr40-oss120b-hip",
    "gpu-llr-focus40-oss120b-hip-perf-playbook-amd": "llr40-oss120b-hip-perf-playbook-amd",
    "gpu-llr-focus40-oss120b-hip-perf-playbook-amd-clean": "llr40-oss120b-hip-perf-playbook-amd",
    "gpu-llr-focus40-oss120b-hip-skills": "llr40-oss120b-hip-skills",
    "gpu-llr-focus40-oss120b-hip-skills-clean": "llr40-oss120b-hip-skills",
    "gpu-llr-focus40-oss120b-triton": "llr40-oss120b-triton",
    "gpu-llr-focus40-oss120b-triton-clean": "llr40-oss120b-triton",
    "gpu-llr-focus40-oss120b-triton-device-clean": "llr40-oss120b-triton-device",
    "gpu-llr-focus40-oss120b-triton-device-skills-clean": "llr40-oss120b-triton-device-skills",
    "gpu-llr-focus40-oss120b-triton-skills": "llr40-oss120b-triton-skills",
    "gpu-llr-focus40-oss120b-triton-skills-clean": "llr40-oss120b-triton-skills",
    "gpu-llr-focus40-ppcg-hip-clean": "llr40-ppcg-hip",
    "gpu-llr-focus40-qwen38-c-openmp": "llr40-qwen38-c-openmp",
    "gpu-llr-focus40-qwen38-c-openmp-clean": "llr40-qwen38-c-openmp",
    "gpu-llr-focus40-qwen38-c-openmp-device-clean": "llr40-qwen38-c-openmp-device",
    "gpu-llr-focus40-qwen38-c-openmp-device-skills-clean": "llr40-qwen38-c-openmp-device-skills",
    "gpu-llr-focus40-qwen38-c-openmp-skills": "llr40-qwen38-c-openmp-skills",
    "gpu-llr-focus40-qwen38-c-openmp-skills-clean": "llr40-qwen38-c-openmp-skills",
    "gpu-llr-focus40-qwen38-hip": "llr40-qwen38-hip",
    "gpu-llr-focus40-qwen38-hip-caveman-clean": "llr40-qwen38-hip-caveman",
    "gpu-llr-focus40-qwen38-hip-clean": "llr40-qwen38-hip",
    "gpu-llr-focus40-qwen38-hip-perf-playbook-amd": "llr40-qwen38-hip-perf-playbook-amd",
    "gpu-llr-focus40-qwen38-hip-perf-playbook-amd-clean": "llr40-qwen38-hip-perf-playbook-amd",
    "gpu-llr-focus40-qwen38-hip-skills": "llr40-qwen38-hip-skills",
    "gpu-llr-focus40-qwen38-hip-skills-clean": "llr40-qwen38-hip-skills",
    "gpu-llr-focus40-qwen38-hip-skills-smoke": "llr40-qwen38-hip-skills-smoke",
    "gpu-llr-focus40-qwen38-triton": "llr40-qwen38-triton",
    "gpu-llr-focus40-qwen38-triton-clean": "llr40-qwen38-triton",
    "gpu-llr-focus40-qwen38-triton-device-clean": "llr40-qwen38-triton-device",
    "gpu-llr-focus40-qwen38-triton-device-skills-clean": "llr40-qwen38-triton-device-skills",
    "gpu-llr-focus40-qwen38-triton-skills": "llr40-qwen38-triton-skills",
    "gpu-llr-focus40-qwen38-triton-skills-clean": "llr40-qwen38-triton-skills",
    "gpuv2-llr40-oss120b-hip": "llr40-oss120b-hip",
    "gpuv2-llr40-oss120b-hip-skills": "llr40-oss120b-hip-skills",
    "gpuv2-llr40-oss120b-omp": "llr40-oss120b-c-openmp",
    "gpuv2-llr40-oss120b-omp-skills": "llr40-oss120b-c-openmp-skills",
    "gpuv2-llr40-qwen38-hip": "llr40-qwen38-hip",
    "gpuv2-llr40-qwen38-hip-skills": "llr40-qwen38-hip-skills",
    "gpuv2-llr40-qwen38-omp": "llr40-qwen38-c-openmp",
    "gpuv2-llr40-qwen38-omp-skills": "llr40-qwen38-c-openmp-skills",
    "gpuv4-llr40-oss120b-pytriton": "llr40-oss120b-triton",
    "gpuv4-llr40-oss120b-pytriton-skills": "llr40-oss120b-triton-skills",
    "gpuv4-llr40-qwen38-pytriton": "llr40-qwen38-triton",
    "gpuv4-llr40-qwen38-pytriton-skills": "llr40-qwen38-triton-skills",
    "harness-focus20-qwen38-claude": "harness20-qwen38-c",
    "harness-focus20-qwen38-claude-clean": "harness20-qwen38-c",
    "harness-focus20-qwen38-miniswe-clean": "harness20-qwen38-c-miniswe",
    "harness-focus20-qwen38-openhands-clean": "harness20-qwen38-c-openhands",
    "harness-focus20-smoke-mi200-qwen38-claude": "harness20-qwen38-c-smoke",
    "harness-focus20-smoke-mi200-qwen38-miniswe": "harness20-qwen38-c-miniswe-smoke",
    "harness20-caveman-qwen38-c-clean": "harness20-caveman-c-qwen38-c",
    "harness20-oss120b-claude-autokernel-clean": "harness20-oss120b-c-autokernel",
    "harness20-oss120b-claude-clean": "harness20-oss120b-c",
    "harness20-oss120b-miniswe-clean": "harness20-oss120b-c-miniswe",
    "harness20-oss120b-openhands-clean": "harness20-oss120b-c-openhands",
    "harness20-qwen38-claude-autokernel-clean": "harness20-qwen38-c-autokernel",
    "harness20-qwen38-claude-clean": "harness20-qwen38-c",
    "harness20-qwen38-miniswe-clean": "harness20-qwen38-c-miniswe",
    "harness20-qwen38-openhands-clean": "harness20-qwen38-c-openhands",
    "llr-focus40-glm53-c-skills": "llr40-glm53-c-skills",
    "llr-focus40-mi200-smoke-qwen38-claude": "llr40-qwen38-c-smoke",
    "llr-focus40-oss120b-c": "llr40-oss120b-c",
    "llr-focus40-oss120b-c-caveman-clean": "llr40-oss120b-c-caveman",
    "llr-focus40-oss120b-c-clean": "llr40-oss120b-c",
    "llr-focus40-oss120b-c-perf-playbook-cpu": "llr40-oss120b-c-perf-playbook-cpu",
    "llr-focus40-oss120b-c-perf-playbook-cpu-clean": "llr40-oss120b-c-perf-playbook-cpu",
    "llr-focus40-oss120b-c-skills": "llr40-oss120b-c-skills",
    "llr-focus40-oss120b-c-skills-clean": "llr40-oss120b-c-skills",
    "llr-focus40-oss120b-fortran": "llr40-oss120b-fortran",
    "llr-focus40-oss120b-fortran-clean": "llr40-oss120b-fortran",
    "llr-focus40-oss120b-fortran-skills": "llr40-oss120b-fortran-skills",
    "llr-focus40-oss120b-fortran-skills-clean": "llr40-oss120b-fortran-skills",
    "llr-focus40-pluto-c-clean": "llr40-pluto-c",
    "llr-focus40-qwen38-c": "llr40-qwen38-c",
    "llr-focus40-qwen38-c-caveman-clean": "llr40-qwen38-c-caveman",
    "llr-focus40-qwen38-c-clean": "llr40-qwen38-c",
    "llr-focus40-qwen38-c-perf-playbook-cpu": "llr40-qwen38-c-perf-playbook-cpu",
    "llr-focus40-qwen38-c-perf-playbook-cpu-clean": "llr40-qwen38-c-perf-playbook-cpu",
    "llr-focus40-qwen38-c-skills": "llr40-qwen38-c-skills",
    "llr-focus40-qwen38-c-skills-clean": "llr40-qwen38-c-skills",
    "llr-focus40-qwen38-fortran": "llr40-qwen38-fortran",
    "llr-focus40-qwen38-fortran-clean": "llr40-qwen38-fortran",
    "llr-focus40-qwen38-fortran-skills": "llr40-qwen38-fortran-skills",
    "llr-focus40-qwen38-fortran-skills-clean": "llr40-qwen38-fortran-skills",
    "llr40v10-kimi27sglang-c": "llr40-kimi27sglang-c",
    "llr40v10-kimi27sglang-c-skills": "llr40-kimi27sglang-c-skills",
    "llr40v10-kimi27sglang-fortran": "llr40-kimi27sglang-fortran",
    "llr40v10-kimi27sglang-fortran-skills": "llr40-kimi27sglang-fortran-skills",
    "llr40v10-oss120b-c": "llr40-oss120b-c",
    "llr40v10-oss120b-c-skills": "llr40-oss120b-c-skills",
    "llr40v10-oss120b-fortran": "llr40-oss120b-fortran",
    "llr40v10-oss120b-fortran-skills": "llr40-oss120b-fortran-skills",
    "llr40v10-qwen38-c": "llr40-qwen38-c",
    "llr40v10-qwen38-c-skills": "llr40-qwen38-c-skills",
    "llr40v10-qwen38-fortran": "llr40-qwen38-fortran",
    "llr40v10-qwen38-fortran-skills": "llr40-qwen38-fortran-skills",
    "llr40v11-kimi27sglang-c": "llr40-kimi27sglang-c",
    "llr40v11-kimi27sglang-c-skills": "llr40-kimi27sglang-c-skills",
    "llr40v11-kimi27sglang-fortran": "llr40-kimi27sglang-fortran",
    "llr40v11-kimi27sglang-fortran-skills": "llr40-kimi27sglang-fortran-skills",
    "llr40v11-oss120b-c": "llr40-oss120b-c",
    "llr40v11-oss120b-c-skills": "llr40-oss120b-c-skills",
    "llr40v11-oss120b-fortran": "llr40-oss120b-fortran",
    "llr40v11-oss120b-fortran-skills": "llr40-oss120b-fortran-skills",
    "llr40v11-qwen38-c": "llr40-qwen38-c",
    "llr40v11-qwen38-c-skills": "llr40-qwen38-c-skills",
    "llr40v11-qwen38-fortran": "llr40-qwen38-fortran",
    "llr40v11-qwen38-fortran-skills": "llr40-qwen38-fortran-skills",
    "llr40v9-kimi27sglang-c": "llr40-kimi27sglang-c",
    "llr40v9-kimi27sglang-c-skills": "llr40-kimi27sglang-c-skills",
    "llr40v9-kimi27sglang-cpp": "llr40-kimi27sglang-cpp",
    "llr40v9-kimi27sglang-fortran": "llr40-kimi27sglang-fortran",
    "llr40v9-kimi27sglang-fortran-skills": "llr40-kimi27sglang-fortran-skills",
    "llr40v9-oss120b-c": "llr40-oss120b-c",
    "llr40v9-oss120b-c-skills": "llr40-oss120b-c-skills",
    "llr40v9-oss120b-cpp": "llr40-oss120b-cpp",
    "llr40v9-oss120b-fortran": "llr40-oss120b-fortran",
    "llr40v9-oss120b-fortran-skills": "llr40-oss120b-fortran-skills",
    "llr40v9-qwen38-c": "llr40-qwen38-c",
    "llr40v9-qwen38-c-skills": "llr40-qwen38-c-skills",
    "llr40v9-qwen38-cpp": "llr40-qwen38-cpp",
    "llr40v9-qwen38-fortran": "llr40-qwen38-fortran",
    "llr40v9-qwen38-fortran-skills": "llr40-qwen38-fortran-skills",
    "llrblind-cmp-kimi27sglang-c-clean": "llr40-kimi27sglang-c-blind",
    "llrblind-cmp-kimi27sglang-c-skills-clean": "llr40-kimi27sglang-c-skills-blind",
    "llrblind-cmp-kimi27sglang-fortran": "llr40-kimi27sglang-fortran-blind",
    "llrblind-cmp-kimi27sglang-fortran-skills": "llr40-kimi27sglang-fortran-skills-blind",
    "llrblind-cmp-oss120b-c": "llr40-oss120b-c-blind",
    "llrblind-cmp-oss120b-c-skills": "llr40-oss120b-c-skills-blind",
    "llrblind-cmp-oss120b-fortran": "llr40-oss120b-fortran-blind",
    "llrblind-cmp-oss120b-fortran-skills": "llr40-oss120b-fortran-skills-blind",
    "llrblind-cmp-oss120b-hip": "llr40-oss120b-hip-blind",
    "llrblind-cmp-oss120b-hip-clean": "llr40-oss120b-hip-blind",
    "llrblind-cmp-oss120b-hip-skills": "llr40-oss120b-hip-skills-blind",
    "llrblind-cmp-oss120b-hip-skills-clean": "llr40-oss120b-hip-skills-blind",
    "llrblind-cmp-qwen38-c": "llr40-qwen38-c-blind",
    "llrblind-cmp-qwen38-c-skills": "llr40-qwen38-c-skills-blind",
    "llrblind-cmp-qwen38-c-skills-clean": "llr40-qwen38-c-skills-blind",
    "llrblind-cmp-qwen38-fortran": "llr40-qwen38-fortran-blind",
    "llrblind-cmp-qwen38-fortran-skills": "llr40-qwen38-fortran-skills-blind",
    "llrblind-cmp-qwen38-hip": "llr40-qwen38-hip-blind",
    "llrblind-cmp-qwen38-hip-clean": "llr40-qwen38-hip-blind",
    "llrblind-cmp-qwen38-hip-skills": "llr40-qwen38-hip-skills-blind",
    "llrblind-cmp-qwen38-hip-skills-clean": "llr40-qwen38-hip-skills-blind",
    "llrblind-kimi27sglang-c": "llr40-kimi27sglang-c-blind",
    "llrblind-kimi27sglang-c-skills": "llr40-kimi27sglang-c-skills-blind",
    "llrblind-kimi27sglang-fortran-skills": "llr40-kimi27sglang-fortran-skills-blind",
    "llrblind-oss120b-c": "llr40-oss120b-c-blind",
    "llrblind-oss120b-c-skills": "llr40-oss120b-c-skills-blind",
    "llrblind-oss120b-fortran": "llr40-oss120b-fortran-blind",
    "llrblind-oss120b-fortran-skills": "llr40-oss120b-fortran-skills-blind",
    "llrblind-qwen38-c": "llr40-qwen38-c-blind",
    "llrblind-qwen38-c-skills": "llr40-qwen38-c-skills-blind",
    "llrblind-qwen38-fortran": "llr40-qwen38-fortran-blind",
    "llrblind-qwen38-fortran-skills": "llr40-qwen38-fortran-skills-blind",
    "mlscale-kimi27sglang-hip": "mlscale20-kimi27sglang-hip",
    "mlscale-kimi27sglang-hip-gemmhint": "mlscale20-kimi27sglang-hip-gemmhint",
    "mlscale-oss120b-hip": "mlscale20-oss120b-hip",
    "mlscale-oss120b-hip-dist-rccl-amd": "mlscale20-oss120b-hip-dist-rccl-amd",
    "mlscale-oss120b-hip-gemmhint": "mlscale20-oss120b-hip-gemmhint",
    "mlscale-part2-kimi27sglang-hip-gemmhint": "mlscale20-kimi27sglang-hip-gemmhint",
    "mlscale-part2-oss120b-hip-gemmhint": "mlscale20-oss120b-hip-gemmhint",
    "mlscale-part2-qwen38-hip-gemmhint": "mlscale20-qwen38-hip-gemmhint",
    "mlscale-qwen38-hip": "mlscale20-qwen38-hip",
    "mlscale-qwen38-hip-dist-rccl-amd": "mlscale20-qwen38-hip-dist-rccl-amd",
    "mlscale-qwen38-hip-gemmhint": "mlscale20-qwen38-hip-gemmhint",
    "mlscale-smoke-oss120b-hip": "mlscale20-oss120b-hip-smoke",
    "scicomp-dc-cpp-oss120b-plain": "scicomp40-oss120b-cpp",
    "scicomp-dc-fortran-oss120b-plain": "scicomp40-oss120b-fortran",
    "scicomp-dc-fortran-oss120b-plain-clean": "scicomp40-oss120b-fortran",
    "scicomp-dc-fortran-qwen38-plain": "scicomp40-qwen38-fortran",
    "scicomp-dc-fortran-qwen38-plain-clean": "scicomp40-qwen38-fortran",
    "scicomp-dc-gpu-oss120b-hip-plain": "scicomp40-oss120b-hip",
    "scicomp-dc-gpu-oss120b-hip-plain-clean": "scicomp40-oss120b-hip",
    "scicomp-dc-gpu-oss120b-triton-plain": "scicomp40-oss120b-triton",
    "scicomp-dc-gpu-oss120b-triton-plain-clean": "scicomp40-oss120b-triton",
    "scicomp-dc-gpu-qwen38-hip-plain": "scicomp40-qwen38-hip",
    "scicomp-dc-gpu-qwen38-hip-plain-clean": "scicomp40-qwen38-hip",
    "scicomp-dc-gpu-qwen38-triton-plain": "scicomp40-qwen38-triton",
    "scicomp-dc-gpu-qwen38-triton-plain-clean": "scicomp40-qwen38-triton",
    "scicomp-dc-oss120b-plain": "scicomp40-oss120b-c",
    "scicomp-dc-oss120b-plain-clean": "scicomp40-oss120b-c",
    "scicomp-dc-qwen38-plain": "scicomp40-qwen38-c",
    "scicomp-dc-qwen38-plain-clean": "scicomp40-qwen38-c",
    "scicomp-perf-playbook-fortran-oss120b-perf-playbook-cpu-clean": "scicomp40-oss120b-fortran-perf-playbook-cpu",
    "scicomp-perf-playbook-fortran-qwen38-perf-playbook-cpu-clean": "scicomp40-qwen38-fortran-perf-playbook-cpu",
    "scicomp-perf-playbook-gpu-oss120b-hip-perf-playbook-amd-clean": "scicomp40-oss120b-hip-perf-playbook-amd",
    "scicomp-perf-playbook-gpu-qwen38-hip-perf-playbook-amd-clean": "scicomp40-qwen38-hip-perf-playbook-amd",
    "scicomp-perf-playbook-kimi27sglang-perf-playbook-cpu-clean": "scicomp40-kimi27sglang-c-perf-playbook-cpu",
    "scicomp-perf-playbook-kimi27sglang-plain": "scicomp40-kimi27sglang-c",
    "scicomp-perf-playbook-kimi27sglang-plain-clean": "scicomp40-kimi27sglang-c",
    "scicomp-perf-playbook-oss120b-perf-playbook-cpu": "scicomp40-oss120b-c-perf-playbook-cpu",
    "scicomp-perf-playbook-oss120b-perf-playbook-cpu-clean": "scicomp40-oss120b-c-perf-playbook-cpu",
    "scicomp-perf-playbook-oss120b-plain": "scicomp40-oss120b-c",
    "scicomp-perf-playbook-oss120b-plain-clean": "scicomp40-oss120b-c",
    "scicomp-perf-playbook-qwen38-perf-playbook-cpu": "scicomp40-qwen38-c-perf-playbook-cpu",
    "scicomp-perf-playbook-qwen38-perf-playbook-cpu-clean": "scicomp40-qwen38-c-perf-playbook-cpu",
    "scicomp-perf-playbook-qwen38-plain": "scicomp40-qwen38-c",
    "scicomp-perf-playbook-qwen38-plain-clean": "scicomp40-qwen38-c",
    "v11w2-kimi27sglang-c": "llr40-kimi27sglang-c",
    "v11w2-kimi27sglang-c-skills": "llr40-kimi27sglang-c-skills",
    "v11w2-kimi27sglang-fortran": "llr40-kimi27sglang-fortran",
    "v11w2-kimi27sglang-fortran-skills": "llr40-kimi27sglang-fortran-skills",
    "v11w2-oss120b-c": "llr40-oss120b-c",
    "v11w2-oss120b-c-skills": "llr40-oss120b-c-skills",
    "v11w2-oss120b-fortran": "llr40-oss120b-fortran",
    "v11w2-oss120b-fortran-skills": "llr40-oss120b-fortran-skills",
    "v11w2-qwen38-c": "llr40-qwen38-c",
    "v11w2-qwen38-c-skills": "llr40-qwen38-c-skills",
    "v11w2-qwen38-fortran": "llr40-qwen38-fortran",
    "v11w2-qwen38-fortran-skills": "llr40-qwen38-fortran-skills",
}
#: (pattern, replacement) re.sub pairs applied to a setup name before SETUP_RENAMES (studies.yaml ``setup_aliases``).
SETUP_ALIASES: tuple[tuple[str, str], ...] = (
    (r"^cpf-(llr-focus40-(?!.*(?:-cpf$|-cpf-|cpfsrc)).*)$", r"\1"),
    (r"^scicomp-dc-([^-]+)-plain(-clean)?$", r"scicomp-perf-playbook-\1-plain\2"),
)
#: A pre-cmp llrblind setup is the llrblind-cmp setup of the same model, language and packet.
RENAMED_PREFIXES: tuple[tuple[str, str], ...] = (("llrblind-", "llrblind-cmp-"),)
#: Registered languages and packets (the vocabulary at the time of the migration), for repairing a corrupted language.
LANGUAGES = ("c", "cpp", "fortran", "python", "cuda", "hip", "triton", "omp")
PACKETS = frozenset(
    {
        "cpfsrc", "cpf", "lang-skills", "divide-and-conquer", "profiling", "repo", "no-score-tool", "rocprof", "nsys",
        "opt-reports", "autokernel", "lang", "all-in", "perf-playbook-cpu", "perf-playbook-amd", "perf-playbook-nvidia",
        "all-in-cpu", "all-in-amd", "all-in-nvidia", "kernel", "caveman", "cpfsrc-v2", "distributed-amd", "dist-rccl-amd",
    }
)  # fmt: skip
PACKET_ALIASES = {"skills": "lang-skills", "no-score": "no-score-tool", "openmp-offload": ""}
#: Final-grade stamps retired with this migration -> the stamp that replaces them.
STAMP_RENAMES = {"mw4x5-final-v2": "mw4x5"}
CLEAN_SUFFIX = "-clean"
#: The study each recorded study spelling was read as (studies.yaml ``aliases.studies``, retired).
STUDY_ALIASES: dict[str, str] = {
    "llr-focus40": "llr40",
    "scicomp-focus40": "scicomp40",
    "scicomp35": "scicomp40",
    "git-scicomp": "gitscicomp10",
    "harness-focus20": "harness20",
    "mixed": "harness20",
    "mlscale": "mlscale20",
    "mlscale10": "mlscale20",
    "mlscale-part2": "mlscale20",
    "cpf-llr-focus40": "llr40",
    "gpu-llr-focus40": "llr40",
    "gpusmoke5": "llr40",
    "llrblind": "llr-focus40-blind",
    "v11": "llr-focus40-v11",
    "llr40v11": "llr-focus40-v11",
    "v11w2": "llr-focus40-v11",
    "llr40v10": "llr-focus40-v10",
    "llr40v9": "llr-focus40-v9",
    "gpuv2": "llr-focus40-v11",
    "gpuv4": "llr-focus40-v11",
    "glm53llr20": "llr-focus20",
    "llr8": "llr-focus8",
    "llr9": "llr-focus9",
}


class Refused(Exception):
    """The file is not one this script recognises, or migrating it would change its content."""


@dataclasses.dataclass(frozen=True, slots=True)
class Snapshot:
    """What the verification compares: the shape of every table, row counts and row hashes."""

    columns: dict[str, tuple[str, ...]]
    counts: dict[str, int]
    digests: dict[str, str]


def quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def renamed_setup(setup: str) -> str:
    """``setup`` as the retired readers folded it: aliases, then the recorded name, then the llrblind prefix."""

    def aliased(name: str) -> str:
        for pattern, target in SETUP_ALIASES:
            name = re.sub(pattern, target, name)
        return SETUP_RENAMES.get(name, name)

    known = aliased(setup)
    if known == setup:
        for old, new in RENAMED_PREFIXES:
            if setup.startswith(old) and not setup.startswith(new):
                setup = new + setup.removeprefix(old)
                break
        known = aliased(setup)
    return re.sub(r"^cpf-llr-focus40-", "llr40-", known).removesuffix(CLEAN_SUFFIX)


def repaired_language(language: str, packet: str) -> tuple[str, str]:
    """``(language, packet)`` with a baked-in packet token and ``-clean`` unwound (the retired ``split_record_language``)."""
    text = language.removesuffix(CLEAN_SUFFIX)
    for token in LANGUAGES:
        if text == token:
            return token, packet
        if text.startswith(f"{token}-"):
            tail = PACKET_ALIASES.get(text[len(token) + 1 :], text[len(token) + 1 :])
            return token, packet or (tail if tail in PACKETS else "")
    return text, packet


def table_columns(conn: sqlite3.Connection) -> dict[str, tuple[str, ...]]:
    """Every table and view with its columns, in declaration order."""
    names = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type IN ('table', 'view') ORDER BY name")]
    return {n: tuple(r[1] for r in conn.execute(f"PRAGMA table_info({quote(n)})")) for n in names}


def digest(conn: sqlite3.Connection, table: str, columns: Sequence[str]) -> str:
    """SHA-256 over the rows of ``table`` in rowid order, ``columns`` selected by position of the given names."""
    sha = hashlib.sha256()
    select = ", ".join(quote(c) for c in columns)
    for row in conn.execute(f"SELECT {select} FROM {quote(table)} ORDER BY rowid"):
        sha.update(repr(row).encode("utf-8", "surrogatepass"))
        sha.update(b"\n")
    return sha.hexdigest()


def snapshot(conn: sqlite3.Connection, hashed: dict[str, Sequence[str]]) -> Snapshot:
    """Shape and counts of every real table; ``hashed`` names the tables hashed and the columns each hash reads."""
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    columns = {name: cols for name, cols in table_columns(conn).items() if name in tables}
    counts = {name: int(conn.execute(f"SELECT COUNT(*) FROM {quote(name)}").fetchone()[0]) for name in sorted(tables)}
    digests = {name: digest(conn, name, cols) for name, cols in hashed.items() if name in tables}
    return Snapshot(columns, counts, digests)


def describe(conn: sqlite3.Connection) -> list[str]:
    """One line per table and view: name, columns, row count."""
    lines = []
    kinds = dict(conn.execute("SELECT name, type FROM sqlite_master WHERE type IN ('table', 'view') ORDER BY name"))
    for name, columns in table_columns(conn).items():
        rows = int(conn.execute(f"SELECT COUNT(*) FROM {quote(name)}").fetchone()[0]) if kinds[name] == "table" else -1
        count = f"{rows} rows" if rows >= 0 else "view"
        lines.append(f"    {name} ({count}): {', '.join(columns)}")
    return lines


def classify(conn: sqlite3.Connection) -> str:
    """``results-v2`` or ``results-v3``; raises Refused otherwise."""
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    version = int(conn.execute("PRAGMA user_version").fetchone()[0])
    if OLD_TABLES <= tables and "episodes" not in tables:
        if version == SCHEMA_OLD:
            return "results-v2"
        raise Refused(f"v2 tables with user_version {version} (expected {SCHEMA_OLD})")
    if NEW_TABLES <= tables and "runs" not in tables:
        if version == SCHEMA_NEW:
            return "results-v3"
        raise Refused(f"v3 tables with user_version {version} (expected {SCHEMA_NEW})")
    raise Refused("not a results database at schema v2 (tables setups, runs, sources, grades)")


def old_hash_columns(columns: dict[str, tuple[str, ...]]) -> dict[str, Sequence[str]]:
    """The tables and columns hashed before the migration: the unchanged tables whole, ``runs`` without the rewritten ones."""
    hashed: dict[str, Sequence[str]] = {t: columns[t] for t in UNCHANGED_TABLES if t in columns}
    hashed["runs"] = tuple(c for c in columns["runs"] if c not in ("label", "setup"))
    hashed["grades"] = tuple(c for c in columns["grades"] if c != "timing_reduction")
    return hashed


def new_hash_columns(columns: dict[str, tuple[str, ...]]) -> dict[str, Sequence[str]]:
    hashed: dict[str, Sequence[str]] = {t: columns[t] for t in UNCHANGED_TABLES if t in columns}
    hashed["episodes"] = tuple(c for c in columns["episodes"] if c not in ("label", "setup"))
    hashed["grades"] = tuple(c for c in columns["grades"] if c != "timing_reduction")
    return hashed


def identity(row: tuple[object, ...]) -> tuple[object, ...]:
    """A ``setups`` row as the migration will write it: study alias folded, language and packet repaired."""
    study, model, language, device, packet, harness = row
    language, packet = repaired_language(str(language or ""), str(packet or ""))
    return (STUDY_ALIASES.get(str(study or ""), study), model, language, device, packet, harness)


def plan_setups(conn: sqlite3.Connection) -> dict[str, str]:
    """old setup -> new setup for every ``setups`` row; refuses a merge of rows that differ in identity."""
    rows = {
        r[0]: tuple(r[1:])
        for r in conn.execute("SELECT setup, study, model, language, device, packet, harness FROM setups")
    }
    mapping = {old: renamed_setup(old) for old in rows}
    merged: dict[str, list[str]] = {}
    for old, new in mapping.items():
        merged.setdefault(new, []).append(old)
    for new, olds in merged.items():
        if len(olds) < 2:
            continue
        identities = {identity(rows[o]) for o in olds}
        if len(identities) > 1:
            raise Refused(
                f"setups {sorted(olds)} fold into {new} but differ in study/model/language/device/packet/harness"
            )
    return mapping


def rewrite_values(conn: sqlite3.Connection, mapping: dict[str, str], log: list[str]) -> None:
    """Rewrite setup names, episode labels and study names; merge setups that fold together."""
    survivors: dict[str, str] = {}
    for old, new in mapping.items():
        survivors.setdefault(new, old)
    for old, new in sorted(mapping.items()):
        if old == new:
            continue
        count = int(conn.execute("SELECT COUNT(*) FROM episodes WHERE setup = ?", (old,)).fetchone()[0])
        log.append(f"setup {old} -> {new} ({count} episodes{'' if survivors[new] == old else ', merged'})")
        try:
            conn.execute(
                "UPDATE episodes SET label = ? || substr(label, length(?) + 1) WHERE setup = ? AND substr(label, 1, length(?) + 1) = ? || '.'",
                (new, old, old, old, old),
            )
            conn.execute("UPDATE episodes SET setup = ? WHERE setup = ?", (new, old))
        except sqlite3.IntegrityError as exc:
            raise Refused(f"renaming {old} to {new} collides on (job, label, rep): {exc}") from exc
        if survivors[new] == old:
            conn.execute("UPDATE setups SET setup = ? WHERE setup = ?", (new, old))
        else:
            conn.execute("DELETE FROM setups WHERE setup = ?", (old,))
    for old, new in sorted(STUDY_ALIASES.items()):
        count = int(conn.execute("SELECT COUNT(*) FROM setups WHERE study = ?", (old,)).fetchone()[0])
        if count:
            conn.execute("UPDATE setups SET study = ? WHERE study = ?", (new, old))
            log.append(f"study {old} -> {new} ({count} setups)")
    for setup, language, packet in conn.execute("SELECT setup, language, packet FROM setups").fetchall():
        fixed_language, fixed_packet = repaired_language(language, packet)
        if (fixed_language, fixed_packet) != (language, packet):
            conn.execute(
                "UPDATE setups SET language = ?, packet = ? WHERE setup = ?", (fixed_language, fixed_packet, setup)
            )
            log.append(
                f"language {language!r} -> {fixed_language!r}, packet {packet!r} -> {fixed_packet!r} (setup {setup})"
            )


def clean_collisions(conn: sqlite3.Connection, mapping: dict[str, str], log: list[str]) -> None:
    """List, per merged setup, the kernels graded under both a ``-clean`` and a plain spelling (latest run wins)."""
    merged: dict[str, list[str]] = {}
    for old, new in mapping.items():
        merged.setdefault(new, []).append(old)
    for new, olds in sorted(merged.items()):
        clean = [o for o in olds if o.endswith(CLEAN_SUFFIX)]
        plain = [o for o in olds if not o.endswith(CLEAN_SUFFIX)]
        if not clean or not plain:
            continue

        def kernels(names: list[str]) -> set[str]:
            marks = ",".join("?" * len(names))
            sql = f"SELECT DISTINCT g.benchmark FROM grades g JOIN runs r ON r.id = g.run_id WHERE r.setup IN ({marks})"
            return {row[0] for row in conn.execute(sql, names)}

        both = sorted(kernels(clean) & kernels(plain))
        if both:
            log.append(
                f"collision {new}: {len(both)} kernels graded under a clean and a plain spelling: {', '.join(both[:8])}"
            )


def rewrite_stamps(conn: sqlite3.Connection, log: list[str]) -> None:
    for old, new in STAMP_RENAMES.items():
        count = int(conn.execute("SELECT COUNT(*) FROM grades WHERE timing_reduction = ?", (old,)).fetchone()[0])
        if count:
            conn.execute("UPDATE grades SET timing_reduction = ? WHERE timing_reduction = ?", (new, old))
            log.append(f"grades.timing_reduction {old} -> {new} ({count} grades)")


def migrate_results(conn: sqlite3.Connection, log: list[str]) -> dict[str, str]:
    """The v2 -> v3 renames and rewrites, inside the caller's transaction; returns the setup mapping."""
    mapping = plan_setups(conn)
    clean_collisions(conn, mapping, log)
    conn.execute("DROP VIEW IF EXISTS grades_flat")
    for old, new in INDEX_RENAMES:
        conn.execute(f"DROP INDEX IF EXISTS {old}")
        log.append(f"drop index {old} (recreated as {new})")
    conn.execute("ALTER TABLE runs RENAME TO episodes")
    log.append("table runs -> episodes")
    for (table, old), new in COLUMN_RENAMES.items():
        conn.execute(f"ALTER TABLE {TABLE_RENAMES.get(table, table)} RENAME COLUMN {old} TO {new}")
        log.append(f"column {TABLE_RENAMES.get(table, table)}.{old} -> {new}")
    conn.execute("CREATE INDEX grades_episode ON grades (episode_id, kernel)")
    conn.execute("CREATE UNIQUE INDEX episodes_key ON episodes (coalesce(job, -1), label, rep)")
    rewrite_values(conn, mapping, log)
    rewrite_stamps(conn, log)
    conn.execute(GRADES_FLAT)
    conn.execute(f"PRAGMA user_version = {SCHEMA_NEW}")
    log.append(f"user_version {SCHEMA_OLD} -> {SCHEMA_NEW}")
    return mapping


def expected_shape() -> dict[str, tuple[str, ...]]:
    """Every table, view and index of schema.sql with its columns (indexes: their names, no columns)."""
    fresh = sqlite3.connect(":memory:")
    try:
        fresh.executescript(SCHEMA_SQL.read_text(encoding="utf-8"))
        shape = table_columns(fresh)
        shape.update(
            {r[0]: () for r in fresh.execute("SELECT name FROM sqlite_master WHERE type = 'index' AND sql IS NOT NULL")}
        )
        return shape
    finally:
        fresh.close()


def check_shape(conn: sqlite3.Connection) -> None:
    want = expected_shape()
    got = table_columns(conn)
    got.update(
        {r[0]: () for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'index' AND sql IS NOT NULL")}
    )
    if want != got:
        bad = sorted(k for k in set(want) | set(got) if want.get(k) != got.get(k))
        raise Refused(f"the migrated file differs from schema.sql in: {bad}")


def check_content(before: Snapshot, after: Snapshot, mapping: dict[str, str]) -> None:
    """Row counts and checksums of everything the migration must not change."""
    counts = {TABLE_RENAMES.get(t, t): n for t, n in before.counts.items()}
    counts["setups"] = len(set(mapping.values()))
    if counts != after.counts:
        bad = sorted(k for k in set(counts) | set(after.counts) if counts.get(k) != after.counts.get(k))
        raise Refused(f"row counts differ after the migration in: {bad}")
    digests = {TABLE_RENAMES.get(t, t): d for t, d in before.digests.items()}
    if digests != after.digests:
        bad = sorted(k for k in set(digests) | set(after.digests) if digests.get(k) != after.digests.get(k))
        raise Refused(f"row checksums differ after the migration in: {bad}")


def checks(conn: sqlite3.Connection) -> None:
    problems = [r[0] for r in conn.execute("PRAGMA integrity_check")]
    if problems != ["ok"]:
        raise Refused(f"integrity_check: {problems[:3]}")
    broken = conn.execute("PRAGMA foreign_key_check").fetchall()
    if broken:
        raise Refused(f"foreign_key_check reports {len(broken)} violations")


def migrate_database(path: pathlib.Path, dry_run: bool) -> str:
    conn = sqlite3.connect(path, isolation_level=None)
    try:
        if classify(conn) == "results-v3":
            return "already migrated"
        print("  before:")
        print("\n".join(describe(conn)))
        conn.execute("BEGIN IMMEDIATE")
        try:
            columns = table_columns(conn)
            before = snapshot(conn, old_hash_columns(columns))
            log: list[str] = []
            mapping = migrate_results(conn, log)
            after = snapshot(conn, new_hash_columns(table_columns(conn)))
            check_content(before, after, mapping)
            check_shape(conn)
            checks(conn)
            flat = int(conn.execute("SELECT COUNT(*) FROM grades_flat").fetchone()[0])
            if flat != after.counts["grades"]:
                raise Refused(f"grades_flat holds {flat} rows for {after.counts['grades']} grades")
            print("  changes:")
            print("\n".join(f"    {line}" for line in log))
            print("  after:")
            print("\n".join(describe(conn)))
            conn.execute("ROLLBACK" if dry_run else "COMMIT")
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
        return "would migrate (rolled back)" if dry_run else "migrated"
    except sqlite3.DatabaseError as exc:
        raise Refused(f"not a readable SQLite database: {exc}") from exc
    finally:
        conn.close()


def walk(paths: Sequence[pathlib.Path]) -> Iterator[pathlib.Path]:
    for path in paths:
        if path.is_dir():
            yield from sorted(p for p in path.rglob("*") if p.is_file() and p.suffix in SUFFIXES)
        else:
            yield path


def read_only(path: pathlib.Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def verify(original: pathlib.Path, migrated: pathlib.Path) -> None:
    """Raise Refused unless ``migrated`` is ``original`` under the renames and folds, table by table."""
    with read_only(original) as old, read_only(migrated) as new:
        if classify(old) != "results-v2":
            raise Refused("original is not a v2 results database")
        if classify(new) != "results-v3":
            raise Refused("migrated is not a v3 results database")
        before = snapshot(old, old_hash_columns(table_columns(old)))
        after = snapshot(new, new_hash_columns(table_columns(new)))
        mapping = plan_setups(old)
        check_content(before, after, mapping)
        check_shape(new)
        checks(new)
        want = sorted(
            (mapping[setup], STUDY_ALIASES.get(study or "", study))
            for setup, study in old.execute("SELECT setup, study FROM setups")
        )
        got = sorted(set(new.execute("SELECT setup, study FROM setups")))
        if sorted(set(want)) != got:
            raise Refused("the setups table differs from the original under the setup and study folds")
        total = sum(before.counts.values())
        print(f"  equal: {len(before.counts)} tables, {total} rows, {len(before.digests)} checksums")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", type=pathlib.Path, help="files or directories")
    parser.add_argument("--dry-run", action="store_true", help="check and report every change, commit nothing")
    parser.add_argument("--verify", action="store_true", help="PATHS is ORIGINAL NEW: compare, change nothing")
    args = parser.parse_args(argv)
    if args.verify:
        if len(args.paths) != 2:
            parser.error("--verify takes exactly ORIGINAL and NEW")
        print(f"{args.paths[1]} against {args.paths[0]}")
        try:
            verify(*args.paths)
        except (Refused, sqlite3.DatabaseError) as exc:
            print(f"  DIFFERS: {exc}")
            return 2
        return 0
    failed = 0
    for path in walk(args.paths):
        print(f"{path}")
        try:
            print(f"  -> {migrate_database(path, args.dry_run)}")
        except Refused as exc:
            failed += 1
            print(f"  -> REFUSED: {exc}")
    return 2 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
