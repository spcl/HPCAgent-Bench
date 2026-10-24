# Adding a packet

A packet is a registered bundle of skill pages, MCP tools and env switches, plus an optional method
(its own loop text and tools). It is a `PacketDef` entry of `vocabulary.PACKETS`, added in
`hpcagent_bench/skill_packets.py`, in a page's frontmatter or by a plugin; `hpcagent_bench/packets.py` resolves it. A
single skill is its own packet and needs no entry; a `;`-separated list (`rocprof;nsys`) is an ad-hoc packet. Run commands from the repo root
with the package installed (`uv sync`) and `. hpcagent_bench/cluster/env.sh` (`PYTHONHASHSEED=0`).
The mechanism every registered kind shares is in [registry.md](registry.md).

| Kind | Files |
|---|---|
| skill bundle | one class in `skill_packets.py` |
| env or tool switch | the class (or the page's `packet:` block) with `env` and `tools`; a listed tool is a `@tool(core=False)`, served only where `HPCAGENT_BENCH_PACKET_TOOLS` names it |
| method | a plugin ([plugins.md](plugins.md)): `methods/<name>/method.md`, its tools under `tools/`, and the packet in `plugin.toml` naming both |

## Examples (from `skill_packets.py`)

```python
PACKETS.add(  # skill bundle, CPU languages only
    "perf-playbook-cpu",
    PacketDef(
        name="Performance Toolkit (CPU)",
        skills=("divide-and-conquer", "profiling", "opt-reports"),
        device=PacketDevice.CPU,
    ),
)
PACKETS.add(  # composition
    "all-in-cpu",
    PacketDef(name="All-in (CPU)", packets=("cpf-cpu-src", "perf-playbook-cpu", "lang")),
)
```

A packet that is one page plus a name, tools or env switches declares itself in that page's frontmatter instead
(the CPF and tracer pages), so a new skill with its packet is one file:

```yaml
packet:
  order: 1            # a plugin's page leaves it out and takes the next free slot
  name: Canonical Parallel Form Tool, CPU
  aliases: [cpf-tool]
  tools: [canonical_parallel_form]
  env: {HPCAGENT_BENCH_SERVICE_CANONICAL_PARALLEL_FORM_DIR: "${HPCAGENT_BENCH_CPF_CACHE}"}
```

## What a packet holds

`PACKETS.add(key, PacketDef(...), aliases=(...))`. The fields are:

| Field | Meaning |
|---|---|
| `name` | display name (required) |
| `skills` | page directories to stage; `lang` expands to the language pages for the setup, `*` to every shipped page except packet-tool manuals |
| `packets` | registered keys to compose, resolved recursively; each must exist |
| `env` | `(KEY, value)` switches; `${VAR}` is filled from the caller's environment |
| `method` | a method a plugin ships (its `method.md` is the prompt's `{{HINTS}}`), at most one per resolved packet |
| `tools` | MCP tools served only in this packet's setups; its `skills` pages become their manual |
| `device` | `PacketDevice.CPU`, `.AMD` or `.NVIDIA`; resolving for a language that device does not run is refused |
| `frozen` | reason a recorded key takes no new submissions; it still resolves for old records |
| `marker` | the shape the packet wears instead of the shape pool's next free one |
| `short` | a figure's short spelling |

Anything else is refused at import, and so is a wrong type, a missing `name`, a taken key or alias, a composed key that is not registered and a `device` that is not one of the three. The key `""`
is the no-packet control: it takes `slotted=False`.

A packet that stages a file rather than a page announces it in the task text through `packet_note`
in `hpcagent_bench/cluster/make_problems.py`.

The canonical parallel form (CPF) packets are four keys: `cpf-cpu-src` and `cpf-gpu-src` stage the form as the
agent's starting source (`CPF_DROPIN_DIR`), `cpf-tool-cpu` and `cpf-tool-gpu` serve it through the
`canonical_parallel_form` tool. `cpf-src` and `cpf-tool` are the device-neutral spellings a `PACKETS` knob takes:
aliases of the CPU keys, which `packets.by_device` renames to the GPU keys for a device language
(`packets.DEVICE_ONLY_LANGUAGES`). The setup records the device key it resolved to.

Each has a `-kinds` counterpart (`cpf-cpu-src-kinds`, `cpf-gpu-src-kinds`, `cpf-tool-cpu-kinds`,
`cpf-tool-gpu-kinds`; `cpf-src-kinds` and `cpf-tool-kinds` on a `PACKETS` knob) that serves the same code with
each loop's hint cut to one `// par`, `// seq` or `// unknown` and no other comment, and whose page defines those
three words in place of the hint table. The plain keys serve the whole hints (`proofs`: class, reason, witness, the
header contract and library-node provenance). The detail is `HPCAGENT_BENCH_CPF_HINTS` (`cpf.hints`,
`cpf_cache.HintDetail`), set by the `-kinds` packets and read by the judge per setup, the tool, the staging scripts
and the prepare job (`--cpf-hints`). It is part of every form's cache key, so one cache holds both details and never
serves one for the other; the judge refuses a request naming a detail its
setup does not serve.

## Rules

- The registration position is the slot. It sets the packet's hue and, unless it names a `marker`, its shape,
  and the packet with the lowest slot is a composition's lead. Add a new packet last; `tests/test_vocabulary.py`
  pins the order, so append yours there.
- A recorded key is immutable: changing its skills, env or method needs a new key. A rename with the
  same meaning is an alias (`aliases=("old-name",)`).
- `packets.canonical` names a registered composite only when the staged pages and switches match it
  exactly.
- A running job sources its env once; a registry edit reaches only newly submitted setups.

## Validate

```bash
HPCAGENT_BENCH_CPF_CACHE=$SCRATCH/cpf-cache python -m hpcagent_bench.packets --packet cpf-tool --language c
python hpcagent_bench/cluster/make_problems.py --select scaled_add --language c --packet perf-playbook-cpu > problems.jsonl
python -m pytest --maxfail=10 tests/test_packets.py tests/test_packet_env.py \
  tests/test_make_problems_packet.py tests/test_packet_wiring.py \
  tests/test_skill_isolation_matrix.py
```

`python -m hpcagent_bench.packets` prints the resolved env and a final `HPCAGENT_BENCH_RECORD_PACKET=<canonical key>`
line, which `record_identity` writes into the setup's `.env` and the DB stores as `setups.packet`.

## External packets (pages kept outside the repository)

A study can bring its own skill pages without committing them. `skills.external`
(`HPCAGENT_BENCH_SKILLS_EXTERNAL`, a list of folders) names folders that each hold a `skills.yaml`:

```yaml
packets:
  preconditioner-v4-single:
    name: Solver Preconditioners v4, One Page
    pages: [single/solver-preconditioners.md]
  preconditioner-v4-split:
    name: Solver Preconditioners v4, Five Pages
    pages: [split/solver-preconditioners-verdict.md, split/solver-preconditioners-frozen.md]
```

- A page is a markdown file inside the folder with the shipped pages' frontmatter (`name`, `description`,
  `when`). Its `name` is its identity: the task lists it as `/skills/<name>.md` and it is staged under that name.
- An external packet stages pages and nothing else. `tools`, `env` and `method` are refused: a tool runs beside
  the judge, and an external one could read what grading hides. `packets` may name shipped packets or pages
  to stage with it.
- No external page or packet reuses a shipped name or another folder's.
- External packets join the vocabulary after the shipped ones, in key order, so each gets a slot of its own.
- A setup names one in `record.packet` (`PACKETS`) like a shipped packet, with `HPCAGENT_BENCH_SKILLS_EXTERNAL`
  in its env. `make_problems.py` records each page's file in the problems line, so staging needs no external
  folder; plotting a study's rows needs `skills.external` set again so the packet's display name resolves.
- The registry is read once per process: a running job does not see an edit to the folder.

`hpcagent_bench/external_skills.py` reads the folders; `tests/test_external_skills.py` pins the rules.
