#!/bin/bash
#SBATCH --job-name=probe-raw-fftw
#SBATCH -A a-g34
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=00:20:00
#SBATCH --nice=0
#SBATCH --output=%x-%j.out
# Isolate: does RAW libfftw3 (no numpyto translator in the loop) hang in the judge image too?
# plan vs execute timed separately. A hang gets SIGABRT'd (core dump on) instead of a plain
# `timeout` kill, so a gdb backtrace on the core pinpoints where it is stuck.
set -uo pipefail
REPO="${REPO:?set REPO}"
srun --environment=hpcagent-bench-judge-mi300-latest bash -c '
  set -x
  ulimit -c unlimited
  cd /tmp
  gcc --version | head -1
  gcc -O2 -std=c11 -I/opt/spack-install/linux-zen4/fftw-3.3.11-eiyz4wxr6nxiqinbwtz5vtkxvytjrdqc/include \
      '"${REPO}"'/scripts/cscs/_minimal_fftw_probe.c -o /tmp/fftw_probe \
      -L/opt/spack-install/linux-zen4/fftw-3.3.11-eiyz4wxr6nxiqinbwtz5vtkxvytjrdqc/lib -lfftw3 -lm \
      -Wl,-rpath,/opt/spack-install/linux-zen4/fftw-3.3.11-eiyz4wxr6nxiqinbwtz5vtkxvytjrdqc/lib
  echo "core_pattern: $(cat /proc/sys/kernel/core_pattern 2>&1)"
  which gdb || echo "no gdb in image"
  for n in 16 1024 8388608 86794130; do
    echo "=== N=$n ==="
    rm -f /tmp/core /tmp/core.*
    timeout --signal=ABRT 90 /tmp/fftw_probe "$n"
    rc=$?
    echo "rc=$rc"
    if [ "$rc" = "134" ]; then
      core=$(ls -t /tmp/core /tmp/core.* 2>/dev/null | head -1)
      if [ -n "$core" ] && which gdb >/dev/null 2>&1; then
        gdb -batch -ex bt /tmp/fftw_probe "$core" 2>&1 | tail -30
      else
        echo "no core captured (core=$core)"
      fi
    fi
  done
'
