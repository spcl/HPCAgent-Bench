#!/bin/sh
# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
#
# An image's hook for a package of this repo: an EDITABLE install at <root>, made from an empty skeleton and
# emptied again, so the image holds the .pth hook and the dist-info but no package code. A step mounts the
# checkout's copy at <root> -- the judge EDF mounts the checkout at /opt/hpcagent-bench (hpcagent_bench), an
# agent step binds agent/ at /opt/hpcagent-bench-agent (hpcagent_agent) -- which is all it takes to import
# the package, with no install step and no PYTHONPATH; without the mount the import fails.
#
#   package_hook.sh <root> <package> <python> [uv pip install flags]
#
# The Dockerfile has COPY'd <root>/pyproject.toml (and whatever files it names, such as README.md) first.
set -eu
ulimit -c 0
root="$1"
package="$2"
python="$3"
shift 3
mkdir -p "${root}/${package}"
: > "${root}/${package}/__init__.py"
uv pip install --python "${python}" "$@" --no-deps --no-cache -e "${root}"
rm -rf "${root:?}/${package}" "${root}/${package}.egg-info" "${root}/README.md" "${root}/LICENSE" "${root}/NOTICE"
