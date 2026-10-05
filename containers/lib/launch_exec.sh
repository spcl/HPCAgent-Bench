#!/bin/sh
# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
#
# /opt/launch/bin/<command> (python3, pytest, litellm, ...): run <command> from this node's launch venv.
#
# The CSCS Container Engine applies an EDF's [env] after the image's ENTRYPOINT, so the PATH launch_venv.sh exports
# there is replaced by the EDF's; the launch venv is reached through PATH instead: the EDFs put /opt/launch/bin first, every name there is a
# link to this script, and this script drops /opt/launch/bin from PATH (so the command it execs is never itself)
# and hands the command to launch_venv.sh, which execs it with the venv's bin/ first on PATH.

# A core dump lands in the crashing process's CWD (the checkout) and Slurm propagates the
# SUBMITTER's core limit, so the floor has to be set here.
ulimit -c 0
name="${0##*/}"
PATH="$(printf '%s' "${PATH}" | sed -e 's|/opt/launch/bin:||g' -e 's|:/opt/launch/bin$||')"
export PATH
exec /opt/launch/launch_venv.sh "${name}" "$@"
