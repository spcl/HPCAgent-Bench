# Runtime — install, container backends, parallelism

## Install (sudoless)

```bash
pip install -e .                                  # optarena + the numpyto_* translators
pip install -r requirements/cpu.txt               # numeric deps for the hardware target
pip install -r requirements/harbor.txt            # harbor + udocker (only to run the benchmark)
optarena-install-apptainer                         # Apptainer, unprivileged, into ~/.local (optional)
```

Everything is pip-installable except Apptainer itself (a Go binary). `udocker` is
pip-installable; `optarena-install-apptainer` runs Apptainer's official unprivileged
installer for the rootless case. There is no pip package that bundles the Apptainer
binary (`spython` only wraps the CLI).

## Container backends (`runtime.backend`)

| backend | runs via | online? | sudo / daemon | install |
|---------|----------|---------|----------------|---------|
| `apptainer` | local, no Harbor | offline (default) | rootless | `optarena-install-apptainer` |
| `udocker` | local, no Harbor | offline | rootless, pure user space | `pip install udocker` |
| `singularity` | Harbor (`--env singularity`) | online | rootless (user namespaces) | Apptainer |
| `docker` | Harbor (`--env docker`) | online | needs the Docker daemon / group | system Docker |

**Default to the local, sudoless backends (`apptainer`/`udocker`) for offline runs**
(HPC compute nodes, air-gapped), and **Harbor (`docker`/`singularity`) when online**
for orchestration (image pulls, fleet of trials, cloud agents).

**Can Harbor run offline?** Yes — Harbor's network needs are (1) pulling images and
(2) the agent calling a cloud LLM. Point it at **local** prebuilt images (`.sif` /
a locally-loaded docker image, no registry) and a **local** agent, and `harbor run`
works with no internet (its `singularity` provider runs a local `.sif` directly). So:
offline → local `apptainer`/`udocker` by default, *or* Harbor with local images + a
local agent; online → Harbor with remote images/agents.

Harbor provides `docker` and `singularity`; pick per run with `harbor run --env <type>`
or set `[environment].type` in `adapters/optarena/optarena.yaml`. `apptainer` and
`udocker` run an image directly without Harbor (`optarena.containers.local_run_command`).
Build the two images once (Apptainer/podman, both rootless), then run with any backend.

## HPC notes

The norm on clusters is **build off-cluster, run on-cluster**:

- **Building** unprivileged needs `uidmap` (`newuidmap`/`newgidmap`) + per-user
  `/etc/subuid` ranges. These are admin-configured and often missing on HPC, so build
  the `.sif` where you control the box (workstation, CI, podman/Docker) and copy it
  over. Our real-build test (`test_packaging.py::test_apptainer_builds_and_imports`)
  is opt-in and skips when this tooling is absent — matching HPC reality.
- **Running** needs none of that. Apptainer/Singularity is usually preinstalled
  (`module load apptainer`) in setuid/userns mode, so `apptainer run image.sif` needs
  no `uidmap` and no sudo. Where it is absent, `udocker` (pip, pure user space) runs
  the image with no admin support at all. Assume Apptainer modules, GPU drivers, MPI,
  a shared FS, and Slurm — not root or package installs.

## Parallelism — many agents, one timer

Each kernel is an independent task, so the **agent/solve/correctness** phase scales out
freely (e.g. 80 agents at once: raise `n_concurrent_trials`). Only **performance
timing** needs all of the CPU, so it must run one-at-a-time, or contention corrupts the
measurement.

Set `measurement.timing_lock` to a shared path and the grader `flock`s it around the
timing section: agents solve in parallel while exactly one performance measurement runs
at a time. The clean separation — parallel correctness, serial timing — needs the timing
core to split verify (parallel) from measure (serial); see
[HANDOFF_measurement_rigor.md](HANDOFF_measurement_rigor.md). Until then the lock
serializes the whole grade, which still lets agents run in parallel and keeps every
measurement contention-free.
