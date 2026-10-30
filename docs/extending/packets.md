# Adding a packet

A packet is a registered bundle of skill pages, MCP tools, env switches and prompt overlays. There is one way to
declare it: a `[packets.<key>]` table. The shipped packets are the tables of `hpcagent_bench/packets.toml`; a plugin
declares its own in its `plugin.toml` with the same table ([plugins.md](plugins.md)). `hpcagent_bench/skill_packets.py`
reads them (`PacketTable`) into `PacketDef` entries of `vocabulary.PACKETS`, and `hpcagent_bench/packets.py` resolves a
spec. A single skill page is its own packet and needs no table; a `;`-separated list (`rocprof;nsys`) is an ad-hoc
packet. Run commands from the repo root with the package installed (`uv sync`) and `. hpcagent_bench/cluster/env.sh`
(`PYTHONHASHSEED=0`). The mechanism every registered kind shares is in [registry.md](registry.md).

## Examples (from `packets.toml`)

```toml
[packets.cpf-tool-cpu]       # one page plus a tool and a switch: the page named like the key is staged
name = "Canonical Parallel Form Tool, CPU"
aliases = ["cpf-tool"]
tools = ["canonical_parallel_form"]
env = { HPCAGENT_BENCH_SERVICE_CANONICAL_PARALLEL_FORM_DIR = "${HPCAGENT_BENCH_CPF_CACHE}" }

[packets.perf-playbook-cpu]  # skill bundle, CPU languages only
name = "Performance Toolkit (CPU)"
skills = ["divide-and-conquer", "profiling", "opt-reports"]
device = "cpu"

[packets.all-in-cpu]         # composition
name = "All-in (CPU)"
packets = ["cpf-cpu-src", "perf-playbook-cpu", "lang"]

[packets.repo]               # env switches and a prompt overlay
name = "Git Reformulation"
env = { REPO_LAYOUT = "1", REPO_LAYOUT_LANGUAGE = "c" }
overlays = ["repo"]
```

## What a table holds

| Field | Meaning |
|---|---|
| `name` | display name (required) |
| `aliases` | other spellings of the key (a rename keeps its meaning) |
| `skills` | page directories to stage; `lang` expands to the language pages for the setup, `*` to every shipped page except packet-tool manuals. Default: the page named like the key, when its source (the package, or the plugin) ships one |
| `packets` | registered keys to compose, resolved recursively; each must exist and none may reach itself |
| `env` | `KEY = value` switches; `${VAR}` is filled from the caller's environment |
| `overlays` | prompt overlays its setups apply, in order ([prompts.md](../prompts.md)): a built-in name (`agent/prompt_overlays/<name>.yaml`) or `<plugin>:<path>` |
| `tools` | MCP tools served only in this packet's setups; its `skills` pages become their manual |
| `agents` | subagents it offers (a plugin's `agents/<name>.md`) |
| `device` | `cpu`, `amd` or `nvidia`; resolving for a language that device does not run is refused |
| `frozen` | reason a recorded key takes no new submissions; it still resolves for old records |
| `marker` | the shape the packet wears instead of the shape pool's next free one |
| `short` | a figure's short spelling |
| `extends` | a table of the same file this one starts from: tables merge key by key, lists replace, `aliases` are not inherited |
| `agent` | a plugin's phases the agent runs ([plugins.md](plugins.md#phases)); only a plugin declares phases |

Anything else is refused, and so is a wrong type, a missing `name`, a taken key or alias, a composed key that is not
registered and a `device` that is not one of the three. The key `""` is the no-packet control, registered by
`skill_packets.py` before the tables.

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

- The file order is the slot: the shipped tables first, then each plugin's in plugin and key order. The slot sets
  the packet's hue and, unless it names a `marker`, its shape, and the packet with the lowest slot is a
  composition's lead. Add a new shipped packet last; `tests/test_vocabulary.py` pins the order, so append yours
  there.
- A recorded key is immutable: changing its skills, env, tools or overlays needs a new key. A rename with the
  same meaning is an alias.
- `packets.canonical` names a registered composite only when the staged pages and switches match it
  exactly.
- A study's own pages and packets live in a user plugin (`HPCAGENT_BENCH_USER_PLUGIN_DIRS`), never in the
  repository.
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
