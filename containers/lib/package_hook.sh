#!/bin/sh
# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
#
# An image's hook for a package of this repo: an EDITABLE install made by a second `uv sync` from an empty
# skeleton and emptied again, so the environment holds the editable finder and the dist-info but no package
# code. A step mounts the checkout's copy over the skeleton -- the judge EDF mounts the checkout at
# /opt/hpcagent-bench (hpcagent_bench), an agent step binds agent/ at /opt/hpcagent-bench-agent
# (hpcagent_agent, which /opt/hpcagent-bench/agent links to) -- which is all it takes to import the package,
# with no install step and no PYTHONPATH; without the mount the import fails.
#
#   package_hook.sh <workspace> <package dir> <environment> [uv sync flags]
#
# <workspace> holds the COPY'd pyproject.toml and uv.lock (and agent/pyproject.toml, the workspace member);
# <package dir> is the stub package directory the editable build needs; <environment> is the venv or
# interpreter prefix that receives the install. The dependencies are already there from the image's first
# `uv sync`, so this one only adds the project: the Dockerfile passes `--package hpcagent-agent` for the
# agent hook and `--no-install-package hpcagent-agent` for the judge hook, and a harness venv adds its
# `--group harness-<name>`, which installs its dependencies along with the hook.
set -eu
ulimit -c 0
workspace="$1"
package_dir="$2"
environment="$3"
shift 3
member="$(dirname "${package_dir}")"
mkdir -p "${package_dir}"
: > "${package_dir}/__init__.py"
(cd "${workspace}" && UV_PROJECT_ENVIRONMENT="${environment}" uv sync --frozen --inexact --no-cache "$@")
rm -rf "${package_dir:?}" "${member}"/*.egg-info "${member}/README.md" "${member}/LICENSE" "${member}/NOTICE"
