# Design — Container Launch + 3-Tier Job Submission

Status: approved design, implementation-ready. Design-only; no code lands from this doc.
Scope: (A) one container-launch factory replacing the four scattered assemblers, and
(B) the 3-tier CSCS submission topology + the judge-server concurrency guarantee.

Grounded against the current tree (lines cited inline): `optarena/containers.py`,
`scripts/run_agent_in_container.sh`, `optarena/agent_bench/judge_scheduler.py`,
`optarena/agent_bench/service.py`, `adapters/optarena/run_adapter.py`,
`scripts/cscs/run_campaign.sbatch`, `optarena/config.yaml`, `docs/RUNTIME.md`.

---

## 0. Settled decisions

- **CE on Alps, apptainer everywhere else, `--native` for local dev.** One rule, no
  per-tier ambiguity. Apptainer is not the sanctioned runtime on Alps (CE / enroot +
  Pyxis via `srun --environment=<edf>` is); apptainer stays the one runtime for CI
  (already gated), generic HPC, and local image builds.
- **One Dockerfile (OCI) is the single build recipe** — everybody is docker-compatible, so
  the OCI image is the universal source: apptainer builds a SIF from it, CE imports/references
  the same OCI, docker/podman run it directly. CE has no build recipe of its own (it only
  *runs* an OCI image). The separate apptainer `.def` files are retired in favor of building the
  SIF from the OCI image. Each role image is `FROM` the **CSCS public GPU base image** (aarch64
  GH200, CUDA/GPU stack preinstalled).
- **CE's "config" is the EDF (TOML), a RUN spec not a build recipe** — image ref (registry OCI
  or a `.sqsh` on scratch) + mounts + `[annotations]` hooks (CXI, aws-ofi-nccl) + `[env]`.
  `srun --environment=<edf>` instantiates the container from it. Shipped as
  `scripts/cscs/env.toml.example`.
- **3 role EDFs on Alps** (inference / agent / judge) — the sbatch passes a per-role
  `--environment=<role.toml>`. Inference EDF → Lorenzo's known-good vLLM image (do NOT rebuild
  the vLLM stack on apptainer); agent + judge EDFs → the OptArena image. All three `FROM` the
  CSCS public GPU base.
- **Judge is a request-server**: each request carries the kernel body; concurrent
  requests must not collide. Invariant from the harness: **1 agent = 1 benchmark at a
  time**, so two concurrent requests are always *different* kernels (never the same one).

### Topology

| Tier | Alps | Elsewhere | Local |
|---|---|---|---|
| inference (vLLM) | **CE** — Lorenzo's vLLM EDF | apptainer, or hit a remote vLLM | — |
| agent ("think") | **CE** | **apptainer** | `--native` |
| judge ("measure", server) | **CE** | **apptainer** | `--native` |

Both CE and apptainer are first-class instances of the one launch factory (§1). The
existing `scripts/cscs/run_campaign.sbatch` already drives the CE path correctly and is
left hand-written (§4, out of scope).

---

## A. Container-launch factory

### 1. Problem

Every launch is `outer-prefix + [image-ref] + inner-argv` (inner almost always
`python -m optarena.cli agent <args>`); only the flag SPELLING and the image-ref FORM
vary. Yet argv is assembled four disconnected times: (a) three hand-built bash functions
`run_apptainer`/`run_docker`/`run_podman` (`run_agent_in_container.sh:53-89`) with
per-hw GPU arrays (`:35-43`); (b) `harbor run … --env singularity` name hardcoded
(`run_adapter.py:116`); (c) `DEFAULT_JUDGE_LAUNCHER` srun template
(`judge_scheduler.py:80`); (d) the Alps CE srun launcher with the image glued into
`--environment` (`run_campaign.sbatch:65,115`). Plus a **dead selector**:
`config.yaml runtime.backend` (values `apptainer|udocker|singularity|docker`) has zero
Python readers and conflicts with the shell's separate `$OPTARENA_CONTAINER_RUNTIME`
(values `apptainer|podman|docker`) — two keys, two vocabularies, one concept.

### 2. The factory — home + single source of truth

Home: `optarena/containers.py` (base layer; already holds the live
`optarena-install-apptainer` installer, and its docstring already promises this rebuild).
May import `optarena.config`/`optarena.paths`; **must not** import `optarena.agent_bench`
(layering). `docs/RUNTIME.md:63` already names `optarena.containers.local_run_command` as
the local-run entry — adopt that name (zero doc churn).

Single source of truth = one language-neutral flat file `optarena/container_backends.txt`
(`<backend>.<field>=<value>` lines, `.properties` shape — deliberately not JSON/YAML),
read by **both** consumers so there is no hand-kept mirror:

- Python: `dict(line.split("=", 1))` (stdlib).
- Bash: `while IFS='=' read -r k v` into `declare -A` (no jq, no python).

This is the headline: the **python-less HPC login host physically reads the same source
Python does.** Values are verbatim from the audit. Two non-fabricated choices:
`udocker.gpu.*` is EMPTY (udocker enables NVIDIA at setup time via `udocker setup
--nvidia`, not a run flag); `podman`/`udocker` `harbor_env` are EMPTY (Harbor exposes only
`docker`+`singularity`, so empty = "not a Harbor backend"). Runtime probe commands
(`test -f`, `docker image inspect`, `podman image exists`) are NOT in the file — they are
per-CLI behavior, never `eval`-ed from data; probing stays in bash.

### 3. Python surface (all public, yapf-120)

```python
BACKENDS_PATH = pathlib.Path(__file__).parent / "container_backends.txt"

@dataclass(frozen=True)
class WrapperSpelling:
    name: str
    verb: tuple[str, ...]
    bind_flag: str; workdir_flag: str; env_flag: str
    gpu: Mapping[str, tuple[str, ...]]   # {"nvidia": (...), "amd": (...)}; cpu -> ()
    image_form: str                      # "sif" | "tag"
    image_default: str                   # "optarena-{hw}.sif" | "optarena:{hw}"
    harbor_env: str                      # "singularity" | "docker" | "" (empty = not Harbor)

def load_backends(path=BACKENDS_PATH) -> dict[str, WrapperSpelling]
SPELLINGS = load_backends(); PASSTHROUGH_ENV: tuple[str, ...]
def resolve_backend(explicit=None) -> str
def default_image(backend, hardware="cpu") -> str
def collect_env(hardware, extra=None) -> list[tuple[str, str]]
def local_run_command(inner, *, backend=None, hardware="cpu", image=None,
                      env=None, repo_root=None) -> list[str]      # THE factory
def harbor_env_for(backend=None) -> str                          # raises for empty harbor_env
def ce_launcher(environment=None, gpus="1", ntasks="1", overlap=False,
                node="{node}") -> tuple[str, ...]
def require_ce_environment(prefix, image=None) -> None
```

YAGNI: no `LaunchSpec`/`Launch` intermediate — `local_run_command` takes kwargs, returns
flat argv. Fold order (one pass, identical to what the three bash functions already emit):
`[backend] + verb + gpu[hw] + (env_flag "K=V")* + bind_flag "REPO:REPO" + workdir_flag REPO
+ default_image() + inner`. `collect_env` order is pinned (image env, then fixed seed list,
then sorted dynamic `OPTARENA_*`) so bash and Python folds are **byte-identical** — the
golden parity test leans on this.

### 4. Backend mapping

| Backend | Shape | Handling |
|---|---|---|
| apptainer / docker / podman | exec-wrapper | `SPELLINGS` row → generic `local_run_command` |
| udocker | exec-wrapper | `SPELLINGS` row (**new**, closes the config-named-but-unimplemented gap); GPU empty by design |
| **ce** | image-is-a-flag | `ce_launcher()`, NOT a fold row |
| **harbor** | orchestrator | outside the factory; `harbor_env_for()` returns a provider NAME only |

**CE — glued `--environment`.** CE's image *is* a flag; mounts/env come from the EDF, not
the CLI, so it never enters the fold. `ce_launcher()` returns only the outer srun prefix,
`{node}` un-substituted so `judge_scheduler.srun_wrap` keeps sole ownership of per-slot
substitution (`judge_scheduler.py:177`). With no args it returns exactly today's
`DEFAULT_JUDGE_LAUNCHER` → `judge_scheduler` sets
`DEFAULT_JUDGE_LAUNCHER = containers.ce_launcher()`. The Alps richer default is
`ce_launcher(environment="${ENV_TOML}", overlap=True)`. The audit shows two flag spellings
in the live sbatch — space form `--environment ${ENV_TOML}` (`:65`) and glued
`--environment=${ENV_TOML}` (`:84,115`) — so `require_ce_environment` (the fail-loud graft)
accepts BOTH and raises only when a CE launch carries neither the token nor an explicit
image. It never touches the plain-srun `DEFAULT_JUDGE_LAUNCHER` (legitimately no
`--environment`).

**Harbor — orchestrator.** Stays outside the wrapper factory; we hand it a backend NAME.
`harbor_env_for()` maps `apptainer→"singularity"`, `docker→"docker"`; `podman`/`udocker`/`ce`
have empty `harbor_env` → **raise**. `run_adapter.py` swaps its hardcoded `("--env","singularity")`
for `harbor_env_for()`, catching the raise to point `ce` at the script lane — default resolves to
`singularity`, byte-identical, but the string lives in one place.

**Harbor cannot run CE (verified in the installed package).** Harbor's `EnvironmentType` enum
lists only `docker` + `singularity` as LOCAL container backends; the rest
(daytona/e2b/modal/ec2/gke/openshift/…) are cloud sandboxes — **no enroot / Pyxis / CSCS
Container-Engine / slurm**. CE is unreachable through Harbor.

**Two submission lanes, one approach.** The SAME universal OCI image and the SAME 3-tier
think/judge topology run under two launchers: **(1) script + CE** on Alps
(`run_campaign.sbatch` → `srun --environment=<edf>`), and **(2) harbor + apptainer** everywhere
else (`run_adapter.py` → `harbor run --env singularity`). Kept until each is verified end-to-end;
they differ only in the outer launcher, not in the image or the grading.

### 5. Selector unification

`config.yaml runtime.backend` becomes THE canonical selector (now with readers), vocabulary
widened to `{apptainer, docker, podman, udocker, ce}` — adds `podman`+`ce`, **drops
`singularity`** as a selectable value. Harbor-ness is decoupled from the runtime name:
`singularity` exists only as `apptainer.harbor_env`. Which path you are on (Harbor vs local
exec) is decided by **which entry point you call** (`run_adapter.py` vs
`run_agent_in_container.sh`), not by the backend string.

`resolve_backend(explicit)` precedence: (1) `explicit` arg → (2) `$OPTARENA_RUNTIME_BACKEND`
→ (3) `$OPTARENA_CONTAINER_RUNTIME` (legacy bash var, honored as a deprecated alias for one
release so one knob drives both paths) → (4) `config.get("runtime.backend")` → (5)
`"apptainer"`. Default is NOT flipped to `auto` (stays `apptainer`; unnecessary behavior
change rejected).

### 6. Host-Python decision

**A host-side Python interpreter is NOT guaranteed before the container is entered.**
`run_agent_in_container.sh` exists to run the harness inside the image on a bare HPC login
node that may have only bash + a container runtime (the agent image deliberately excludes
the optarena package). So the factory is kept OFF the host runtime path. The bridge is the
**data file, not a Python call**: bash reads `container_backends.txt` and folds argv in pure
bash. Robust in both directions — if "host always has Python" later proves true nothing
changes; if false nothing breaks. Recorded assumption, not a bet.

---

## B. Job submission — 3-tier topology + judge-server concurrency

### 7. What already exists

- `run_campaign.sbatch` — 3 node pools in one alloc: agent (think) / vLLM (CE) / judge
  (measure). Single-node TP=4 default, real Ray multi-node path, fail-fast readiness.
- `judge_scheduler` — work-stealing `_work_pool`: **one worker per DeviceSlot, GPU pinned**
  via thread-local `cp.cuda.Device(index)`; `srun_wrap` dispatches a remote slot.
- `pipeline` — `TwoStageScheduler` pipelines think→grade; `grade_request_to_json/from_json`
  already puts `{task, submission(kernel body), params}` on the wire (the "request carries
  the kernel body" server contract).
- `service.py` — the standalone judge HTTP server (`do_POST /score` → `_submission_from_body`
  → `score()`), i.e. the request-server the design calls for.
- `env.toml.example` — a single-EDF template; already notes the per-role-EDF split option.

### 7b. Images + 3 role EDFs (Alps)

- **Build (universal OCI — one recipe for CE AND apptainer).** One Dockerfile per role `FROM`
  the CSCS public GPU base is the ONLY image recipe. Build it once, consume it four ways:
  ```
  podman build -f containers/<role>.Dockerfile -t optarena:<role> .   # OCI image (--platform linux/arm64 for Alps)
  apptainer build optarena-<role>.sif docker-daemon://optarena:<role> # SIF from the SAME OCI (no .def)
  enroot import -o optarena-<role>.sqsh dockerd://optarena:<role>      # CE squashfs (or EDF points at a registry ref)
  docker/podman run optarena:<role> ...                               # run OCI directly
  ```
  The apptainer `.def` files (`containers/*.def`) are **retired** — apptainer builds from the OCI
  image, so there is no second build recipe to keep in sync. **CI impact:** the `e2e-container`
  job's `apptainer build --fakeroot optarena-cpu.sif containers/cpu.def` step
  (`.github/workflows/tests.yml:317`) changes to build the SIF from the OCI image
  (`podman build … && apptainer build … docker-daemon://…`), keeping the same gated smoke test.
- **Run:** `run_campaign.sbatch` currently threads a single `ENV_TOML` into every `srun`. Add
  three role variables — `VLLM_ENV_TOML` / `AGENT_ENV_TOML` / `JUDGE_ENV_TOML`, each defaulting
  to `ENV_TOML` — and pass the matching one to each tier's `srun --environment`. Inference →
  vLLM EDF (Lorenzo's image); agent + judge → OptArena-image EDFs. Backward-compatible: unset
  role vars = today's single-EDF behavior. To avoid three near-identical files, ship ONE
  annotated `env.toml.example` whose comments show the per-role deltas (image ref + mounts +
  hooks) — agent + judge use the SAME OptArena image and differ only in run config; only the
  inference role points at a different (vLLM) image.

### 8. The gap — judge server does not bound concurrent timing

`service.py:282` uses `ThreadingHTTPServer`: each POST runs in its own thread with **no
device-slot arbitration and no GPU pin** (`do_POST` calls `score()` directly, `:233`).

- ✅ **Build isolation is already safe** — every grade builds in its own throwaway dir
  (`sandbox.py` `TemporaryDirectory("agentbench_{kernel}_")`; `optimizers.py`
  `mkdtemp("opt_{kernel}_")`). Concurrent *different-kernel* grades never share a build dir,
  and the 1-agent-1-benchmark rule removes the only same-kernel collision
  (`.dacecache/<sdfg>` is keyed per kernel).
- ❌ **Timing is NOT safe** — K simultaneous requests → K threads time on the same CPU/GPU
  at once. Contended timing corrupts the speedup ratio = wrong leaderboard number. The
  1-agent-1-benchmark rule does **not** fix this (different kernels still contend for
  cores/GPU).

### 9. The fix — a device-slot pool that sequentializes kernels per device

The judge holds a slot pool built from `JudgeConfig` — **one slot per GPU and one (or more)
per CPU** (`gpus_per_node` + `cpu_slots_per_node`, both already in config →
`DeviceSlot(kind="gpu"|"cpu", index)`). A grade acquires the slot matching the kernel's
device (GPU kernel → a GPU slot, CPU kernel → a CPU slot), **pins** it
(`cp.cuda.Device(index)` / `CUDA_VISIBLE_DEVICES` / CPU affinity), runs, releases in a
`finally`. While a slot is held the device is **exclusive → kernels sequentialize on that
device** — that is what makes the timing trustworthy. Reuses `judge_scheduler`'s `DeviceSlot`
+ pinning (one slot model, not a second).

**Slot key is `(device_kind, device_index, request_type)`** so timing-bearing types stay
device-exclusive while non-timing types don't block them:

- timed types (`/score`, `/baseline`) → hold the device slot **exclusively** (one timed
  kernel per device at a time),
- non-timed types (correctness-only / `/oracle`) → a separate lane keyed by type, so a
  correctness check never occupies a timing slot and a flood of one type can't starve another.

`do_POST` picks the lane from the route + the task's residency (CPU vs GPU), acquires, pins,
runs, releases. Requests beyond a lane's slot count **queue** (block → backpressure; optional
503 + `Retry-After` past a max depth) rather than oversubscribe. Identical under CE (Alps) or
apptainer (elsewhere) — orthogonal to the runtime.

---

## 10. Migration sequencing

Each step independently landable + green; test named. Push straight to main per repo policy.

1. **Factory + single-source file.** Add `container_backends.txt` + `setup.py`
   `package_data`/`MANIFEST.in`. Grow `containers.py` (§3). Tests
   (`tests/test_container_launch.py`, no real container/GPU/LLM): file parses; per-backend
   argv assertions × {cpu,nvidia,amd}; `default_image`; `resolve_backend` precedence incl.
   the `$OPTARENA_CONTAINER_RUNTIME` alias; `harbor_env_for` map+raise; `require_ce_environment`
   both flag forms.
2. **Host shell reads the file.** Rewrite `run_agent_in_container.sh` (read → `declare -A` →
   one table-driven fold with `--print`; delete the 3 functions + GPU arrays; keep
   selection/probe/fallthrough). Test = golden parity: bash `--print` argv == `local_run_command`
   argv byte-identical over {apptainer,docker,podman,udocker}×{cpu,nvidia,amd}.
3. **CE de-dup.** `judge_scheduler`: `DEFAULT_JUDGE_LAUNCHER = containers.ce_launcher()` (~2
   lines). Regression-lock the tuple + a structure/prefix parity vs `run_campaign.sbatch:65`.
4. **Harbor name live.** `run_adapter.py:116` → `harbor_env_for(resolve_backend())`. Tests:
   default→`--env singularity`; `$OPTARENA_RUNTIME_BACKEND=docker`→`--env docker`; `podman`→raise.
5. **Selector + docs reconcile** (config/doc only). Widen documented values, drop `singularity`,
   note the deprecated alias; `docs/RUNTIME.md` backend table split into wrapper axis vs
   `harbor_env`.
6. **Judge-server slot-bounding** (§9). `service.py`: per-`(device,type)` slot pool + pin around
   the timed section. Test: N>slots concurrent `/score` requests never run more than the lane's
   slot count of timed sections at once (instrument a counter); results unchanged vs serial.
7. **3 role EDFs in the sbatch** (§7b). Add `VLLM_ENV_TOML`/`AGENT_ENV_TOML`/`JUDGE_ENV_TOML`
   (each default `ENV_TOML`); ship the three `env.<role>.toml.example` files. Backward-compatible.
   Config/script only. (Follow-on, not gating the factory: retire the apptainer `.def` files by
   building the SIF from the shared OCI image.)

---

## 11. Risks / out of scope

**Risks:** package-data file must ship in the wheel (Step 1 test asserts installed-path
resolves); flat-file token invariant (whitespace-separated tokens, no intra-token spaces —
documented in the file header); bash-4 `declare -A` (already assumed); CE parity is structural
only (live `${ENV_TOML}` placeholder + space form); one knob now drives both paths (documented
precedence; `harbor_env_for` raises rather than emit an invalid provider).

**Out of scope (deliberate):** `run_campaign.sbatch` stays hand-written — NOT sourced from the
flat file, CE/vLLM srun lines untouched (don't touch the live cluster path). No new
`scripts/lib/*.sh`. Config default not flipped to `auto`. Bash never gains a Python dependency
(host-Python question resolved: NO). Harbor stays an orchestrator; the factory only supplies its
provider name. Lorenzo's vLLM inference EDF is owned separately (a distinct image, referenced by
the sbatch `--environment`), not built from OptArena's def files.

---

## 12. Duplication budget

Every place the same fact could live twice, and what keeps it to one:

| Concern | Copies | Kept minimal by |
|---|---|---|
| Backend flag spellings (DATA) | **1** | one flat `container_backends.txt`; both python + bash READ it (no mirror) |
| Argv fold LOGIC | 2 (python + bash) | **irreducible** — host may lack python (§6). Kept to a fixed-order concat; the golden parity test (Step 2) locks them byte-identical so drift is caught, not lived with |
| Image recipe | **1** | one Dockerfile per role; apptainer SIF built FROM the OCI (`.def` retired); CE imports the same OCI |
| Device-slot model | **1** | judge server reuses `judge_scheduler.DeviceSlot` + pinning; no second scheduler |
| Grade body | **1** | `grade_once` already the single body for in-proc + remote legs (shipped) |
| Harbor provider name (`"singularity"`) | **1** | lives only in `apptainer.harbor_env`; `run_adapter` reads it, no hardcode |
| Runtime selector | **1** | canonical `runtime.backend`; `$OPTARENA_CONTAINER_RUNTIME` is a deprecated *alias*, not a parallel key |
| CE launcher default | 2 (watch) | `ce_launcher()` (python) vs the sbatch `JUDGE_LAUNCHER` string — sbatch is hand-written by design (no python dep); the Step 3 structural parity test locks them. The one residual to watch |
| Role EDFs | 1 template | one annotated `env.toml.example` with per-role deltas; agent + judge share the OptArena image |

The only genuine code duplication is the two-language fold (host-no-python forces it) and the
CE-launcher default (bash-can't-call-python forces it); both are pinned by a parity test rather
than trusted. Everything else collapses to one source.
