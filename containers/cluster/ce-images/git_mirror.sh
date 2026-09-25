#!/bin/sh
# Build-time git plumbing shared by every Dockerfile under ce-images/. COPY it to /tmp, run it, rm it.
#
#   git_mirror.sh setup   write /usr/local/bin/gitretry and point GitHub URLs at /git-mirrors
#   git_mirror.sh drop    remove the mirror rewrite again before the image ships
#
# Every network git call goes through the gitretry wrapper, and where build.sh mounted a mirror
# (build_common.sh:ce_mirror_args) none of them reaches GitHub at all. The `insteadOf` rewrite covers
# callers the wrapper cannot: spack clones its package repo in-process and vLLM's CMake
# FetchContent clones triton itself, and neither retries on GitHub's rate limiting.
# `protocol.file.allow` goes with it: git refuses local-path submodule clones since the
# CVE-2022-39253 fix, and the rewrite turns every submodule URL into exactly that, so
# `submodule update` would abort with `transport 'file' not allowed`. An unauthenticated clone
# from a busy egress IP gets a 401 (`could not read Username for https://github.com`), which looks
# like a credentials failure and is a transient one: GIT_TERMINAL_PROMPT=0 makes it fail instead of
# blocking on a prompt no build has a terminal for, and the backoff is what fixes it.
#
# `drop` matters because the rewrite points at a path that exists only while the mirror is
# mounted; left in /root/.gitconfig, every git clone run inside the shipped container would
# silently rewrite to a directory that is not there.
set -eux
ulimit -c 0

case "${1:-}" in
setup)
    printf '%s\n' \
      '#!/bin/sh' \
      'set -eu' \
      'export GIT_TERMINAL_PROMPT=0' \
      'n=1' \
      'while :; do' \
      '  git "$@" && exit 0' \
      '  [ "${n}" -ge 10 ] && exit 1' \
      '  d=$((n * n * 10)); if [ "${d}" -gt 300 ]; then d=300; fi' \
      '  sleep "${d}"' \
      '  n=$((n + 1))' \
      'done' \
      > /usr/local/bin/gitretry
    chmod 0755 /usr/local/bin/gitretry
    if [ -d /git-mirrors ]; then
        for m in /git-mirrors/*/*.git; do
            r="${m#/git-mirrors/}"; r="${r%.git}"
            git config --global "url./git-mirrors/${r}.git.insteadOf" "https://github.com/${r}.git"
            git config --global --add "url./git-mirrors/${r}.git.insteadOf" "https://github.com/${r}"
        done
        git config --global --add safe.directory '*'
        git config --global protocol.file.allow always
    fi
    ;;
drop)
    git config --global --name-only --get-regexp '^url\..*\.insteadof$' 2>/dev/null \
      | sort -u | while read -r k; do git config --global --unset-all "${k}" || true; done
    git config --global --unset-all protocol.file.allow || true
    ;;
*)
    echo "usage: git_mirror.sh setup|drop" >&2
    exit 2
    ;;
esac
