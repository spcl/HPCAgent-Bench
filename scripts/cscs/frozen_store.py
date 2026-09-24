#!/usr/bin/env python3
# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The content-addressed file store behind the jobs' frozen trees, and the sweep of ended jobs' trees.

Every cluster job copies the checkout at start (scripts/cscs/code_snapshot.sh) into
``hpcagent-bench-runs/.frozen/<kind>-<jobid>``: ~16k files, on a scratch whose quota is inodes. The
copies of one day differ in a handful of files, so ``link`` replaces each file of a fresh copy with a
hard link to ONE entry per distinct file in the store (``.frozen-store/<sha256[:2]>/<name>``), and 20
concurrent trees cost the files once plus their ~2k directories each.

An entry is named by what defines the file a job reads: ``<sha256 of the bytes>.<mode>.<mtime_ns>``.
The mtime is part of the name because the build freshness checks compare it (a ``.so`` is reused while
newer than its sources, a Pluto/PPCG output while newer than its scop), so a linked tree answers those
checks exactly as the plain copy did.

Why a shared inode stays immutable:

* the live checkout never shares one: the store is fed from the job's own copy, never from the live
  tree, and ``git checkout`` replaces a file by rename anyway;
* a job that rewrites a file of its tree replaces it (temp file + rename): the translator's
  ``write_generated`` and ``write_atomic_text``, ``framework_cache.write_atomic`` and ``save_sdfg``,
  the Pluto/PPCG publish, Python's ``__pycache__`` and numba's cache; gcc/as/ld/gfortran/hipcc unlink
  an existing output before writing it. New files (a ``__pycache__``, a build output) are new inodes;
* ``sweep --verify`` re-hashes every entry, so a writer that ever writes in place is caught.

``sweep`` is the coordinator's cleanup, never run by a job: it removes the frozen trees (and
half-built ``.partial`` copies) of jobs ``sacct`` reports ENDED -- a job it cannot place is kept --
then the store entries no tree links any more. Dry run unless ``--delete``.

Standard library only and Python 3.6: ``link`` runs on the batch host, whose python3 is the site's.
"""

import argparse
import hashlib
import os
import re
import subprocess
import sys

#: Slurm states after which the job's steps are gone for good; anything else (or no record) keeps the tree.
ENDED = ("BOOT_FAIL", "CANCELLED", "COMPLETED", "DEADLINE", "FAILED", "NODE_FAIL", "OUT_OF_MEMORY", "PREEMPTED")
ENDED += ("TIMEOUT",)
#: ``<kind>-<jobid>`` as run_cluster.sh, regrade.sbatch and mlscale-grade.sbatch name their trees.
TREE_NAME = re.compile(r"^(?:job|regrade|mlscale-grade)-([0-9]+)(?:\.partial)?$")
CHUNK = 1 << 20


def digest(path):
    """The sha256 hex digest of the file at ``path``."""
    sha = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            sha.update(block)
    return sha.hexdigest()


def entry_path(store, path, status):
    """The store entry for the file at ``path`` whose ``lstat`` is ``status``."""
    name = f"{digest(path)}.{status.st_mode & 0o7777:o}.{status.st_mtime_ns}"
    return os.path.join(store, name[:2], name)


def link_file(store, path):
    """Make ``path`` a hard link to its store entry; True when it now shares the entry's inode.

    The first tree holding a file donates its inode as the entry. Any later one swaps its own copy
    for the entry by rename, so the path always holds the same bytes. An entry swept between the
    two steps (``sweep``), or one at the file system's link limit (EMLINK), leaves the copy as it is."""
    entry = entry_path(store, path, os.lstat(path))
    os.makedirs(os.path.dirname(entry), exist_ok=True)
    try:
        os.link(path, entry)
        return True
    except FileExistsError:
        pass
    swap = f"{path}.frozen-link"
    try:
        os.link(entry, swap)
        os.rename(swap, path)
    except OSError:
        if os.path.lexists(swap):
            os.unlink(swap)
        return False
    return True


def link_tree(tree, store):
    """Link every regular file under ``tree`` into ``store``; returns (files, linked)."""
    files = linked = 0
    for root, _dirs, names in os.walk(tree):
        for name in names:
            path = os.path.join(root, name)
            if os.path.islink(path) or not os.path.isfile(path):
                continue
            files += 1
            linked += link_file(store, path)
    return files, linked


def job_states(ids):
    """``{jobid: [state, ...]}`` from ONE sacct call (a requeued job has a row per start)."""
    out = subprocess.run(
        ["sacct", "-X", "-n", "-P", "-o", "JobIDRaw,State", "-j", ",".join(sorted(ids))],
        stdout=subprocess.PIPE,
        check=True,
    ).stdout.decode()
    states = {}
    for line in out.splitlines():
        jobid, _, state = line.partition("|")
        states.setdefault(jobid.split("+")[0].split("_")[0], []).append(state.split(" ")[0])
    return states


def ended_trees(frozen):
    """(ended, kept): the tree paths under ``frozen`` whose job sacct reports ended, and the rest."""
    trees = {}
    for name in sorted(os.listdir(frozen)):
        match = TREE_NAME.match(name)
        if match and os.path.isdir(os.path.join(frozen, name)):
            trees[os.path.join(frozen, name)] = match.group(1)
    states = job_states(set(trees.values())) if trees else {}
    ended, kept = [], []
    for tree, jobid in sorted(trees.items()):
        rows = states.get(jobid, [])
        (ended if rows and all(state in ENDED for state in rows) else kept).append(tree)
    return ended, kept


def count_inodes(path):
    """Files plus directories under ``path`` (itself included)."""
    return 1 + sum(len(dirs) + len(names) for _root, dirs, names in os.walk(path))


def remove_tree(path):
    """``rm -rf``: a tree holds only files and directories."""
    subprocess.run(["rm", "-rf", "--", path], check=True)


def unlinked_entries(store):
    """The store entries no frozen tree links any more."""
    found = []
    for root, _dirs, names in os.walk(store):
        found.extend(os.path.join(root, n) for n in names if os.lstat(os.path.join(root, n)).st_nlink == 1)
    return found


def corrupt_entries(store):
    """The store entries whose bytes no longer hash to their name (a writer wrote through a link)."""
    return [
        os.path.join(root, name)
        for root, _dirs, names in os.walk(store)
        for name in names
        if digest(os.path.join(root, name)) != name.split(".")[0]
    ]


def sweep(frozen, store, delete, verify):
    """The coordinator's cleanup; returns the exit status."""
    ended, kept = ended_trees(frozen) if os.path.isdir(frozen) else ([], [])
    freed = 0
    for tree in ended:
        inodes = count_inodes(tree)
        print("{} {} ({} inodes)".format("remove" if delete else "would remove", tree, inodes))
        if delete:
            remove_tree(tree)
            freed += inodes
    for tree in kept:
        print(f"keep {tree} (job not ended per sacct)")
    unlinked = unlinked_entries(store) if os.path.isdir(store) else []
    print("{} {} store entries no tree links".format("remove" if delete else "would remove", len(unlinked)))
    if delete:
        for entry in unlinked:
            os.unlink(entry)
        freed += len(unlinked)
        print(f"freed {freed} inodes")
    if verify and os.path.isdir(store):
        corrupt = corrupt_entries(store)
        for entry in corrupt:
            print(f"CORRUPT {entry}")
        if corrupt:
            return 1
    return 0


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command")
    link = sub.add_parser("link", help="link a fresh frozen tree into the store (code_snapshot.sh)")
    link.add_argument("tree")
    link.add_argument("store")
    clean = sub.add_parser("sweep", help="remove ended jobs' frozen trees and unlinked store entries")
    clean.add_argument("frozen", help="the .frozen directory")
    clean.add_argument("store", help="the .frozen-store directory")
    clean.add_argument("--delete", action="store_true", help="act; the default only reports")
    clean.add_argument("--verify", action="store_true", help="re-hash every store entry; exit 1 on a mismatch")
    args = parser.parse_args(argv)
    if args.command == "link":
        files, linked = link_tree(args.tree, args.store)
        print(f"frozen_store: linked {linked}/{files} files of {args.tree} into {args.store}")
        return 0
    if args.command == "sweep":
        return sweep(args.frozen, args.store, args.delete, args.verify)
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
