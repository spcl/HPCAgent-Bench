# The registry: one typed entry per thing the repository names

Everything the repository names and orders is an entry of a `Kind` (`hpcagent_bench/registry.py`), added in the
module that owns the kind. An entry is a typed value: a frozen dataclass that validates itself when it is built
(`ModelEntry`, `PacketDef`, `Column`, `anticheat.Gate`), or a display name (`str`) for kinds that are only a name.

```python
MODELS.add("qwen38", ModelEntry("Qwen3.8-27B", "Qwen/Qwen3.8-27B-FP8"))
HARNESSES.add("claude", "Claude Code")
```

A kind whose entries are code takes a decorator instead: `Kind.register(key)` adds the decorated object
itself, and a few kinds wrap it with the fields they need (`anticheat.post_run_gate`, the significance tests).

## What is checked, and when

- An entry checks its own fields when it is built: a `Column` refuses an adapter that is not
  `package.module:Class`, an empty precision set and a flavor without its column; a `Gate` refuses an empty
  `where`, a malformed `symbol`, `reruns` without a check and a check on a gate enforced in place.
- `Kind.add` refuses a key or an alias that is already taken, and leaves the kind as it was.
- The rules that span entries run once everything is registered (`vocabulary.check_vocabulary()`, called by
  `study_tags.registry()`): every entry but the no-packet control has a slot, a composed packet exists, and the
  columns of one framework `base` name one `Framework` adapter (autotuning flavors name an `Agent`). Exactly one
  grading protocol has the role `preview`, and `measurement.credited_protocol` names a grade protocol
  (`protocols.check_protocols()`). The framework naming scheme (`<column>[_<flavor>]`, a two-device backend's
  column `<base>_<arch>`, a flavor on its column's `arch`) is checked at import by
  `frameworks.framework.check_flavor_registry()`; see [optimizer.md](optimizer.md).

Closed categories inside an entry are enums (`vocabulary.Arch`, `DaceStage`, `Transform`,
`PacketDevice`, `anticheat.Verdict`), never strings; their spelling exists only where a value is stored or read
(`hpcagent_agent.coded.Coded`).

## The slot

An entry's slot is its registration position within its kind: the colour and marker it is drawn with, the position
a column is listed at. Nobody states one. A new entry is added last (a plugin's after every shipped one), so it never
repaints another; `slotted=False` is the one entry with no slot, the no-packet control.

What a slot decides (`hpcagent_bench/stats/palette.py` reads it; nothing else carries a colour):

| Kind | Slot decides |
|---|---|
| model | marker `slot` of the marker pool; hue `slot` of the tab20 ramp where a figure colours models |
| optimizer | marker `slot` counted from the END of the pool, so a new model never repaints it; hue `slot` |
| harness | the clearest shapes of the treatment pool, in slot order, ahead of every packet; hue `slot` |
| packet | hue `slot` (a combination wears its lead part's hue, lightened one step per extra part, and the lead is the part with the lowest slot) and the shape: the next free one of the treatment pool unless the packet names a `marker` |
| language, device | hue `slot` |
| framework | hue `slot` |

The colour ramp, the marker pools and the lightness step are the only data left in
`hpcagent_bench/envs/registry.yaml`. `tests/test_vocabulary.py` pins each kind's order and
`tests/test_palette.py` pins the resolved colours, so a change that would repaint a published figure fails naming
the entity.

## The kinds

| Kind | Module | Entry | Registered as |
|---|---|---|---|
| `vocabulary.MODELS` | `hpcagent_bench/models.py` | `ModelEntry(name, serves)`: the display name and the checkpoint the tag is expected to serve | the model tag setups record |
| `vocabulary.OPTIMIZERS` | `hpcagent_bench/models.py` | the display name | a compiler or pipeline that stands where a model stands (`dace`, `cpf`); device variants are aliases |
| `vocabulary.HARNESSES` | `hpcagent_bench/models.py` | the display name | the `harness` column value: the loop that drives the model |
| `vocabulary.LANGUAGES` | `hpcagent_bench/models.py` | the proper name (`C++`, not `Cpp`) | the language a setup asked for |
| `vocabulary.DEVICES` | `hpcagent_bench/models.py` | the display name | the `device` column value |
| `vocabulary.PACKETS` | a `[packets.<key>]` table: `hpcagent_bench/packets.toml` (read by `skill_packets.py`) or a plugin's `plugin.toml` | `PacketDef(name, skills=, packets=, env=, prompt_overlays=, tools=, agents=, device=, frozen=, marker=, short=, phases=, workflow=)` | the `packet` column value; see [packets.md](packets.md) |
| `vocabulary.FRAMEWORKS` | `hpcagent_bench/columns.py` | `Column(display, adapter, base, full_name, postfix, arch, sweep_deterministic, precisions, ...)`; the DaCe family is built by `columns.dace_column(arch, stage)`, a plugin's flavor by `columns.flavored(base, key, ...)` | a framework column: the CLI name and the `framework` column value; see [optimizer.md](optimizer.md) |
| `anticheat.ANTICHEAT` | `hpcagent_bench/anticheat.py` | `Gate(title, catches, verdict, where, symbol=, reruns=, check=)`; a post-run gate is its check function under `@post_run_gate(key, title=, catches=, verdict=, where=, ...)`, returning `anticheat.Found(effect, text)` per finding | an anti-cheat gate; its slot is the position a submission meets it, and the table of [anti_cheat.md](../anti_cheat.md) lists the same gates in the same order (`tests/test_anticheat.py`). A gate without a `check` is enforced in the module `where` names; one that `reruns` the submission is skipped once the grade is rejected |
| `grading_protocol(stamp, role, statistic, inputs=, repeat=)` (a call) | `hpcagent_bench/protocols.py` | one line per stamp, see [protocol.md](protocol.md#defining-a-grading-protocol) | a `timing_reduction` stamp; the one grade protocol `measurement.credited_protocol` names is credited |
| `@paired_test(name, version=)`, `@proportion_test`, `@paired_proportion_test`, `@correction`, `@timing_test` | `hpcagent_bench/stats/significance.py` | a FUNCTION returning a `significance.Result` (a correction: the adjusted p values); no slot | a statistical test: a reporting one the configuration picks by name (`statistics.*`), a timing test a grading protocol names (`timing_test`); see [the test registry](../measurement_statistics.md#the-test-registry) |

An alias resolves to the entity it names, takes no slot, and shares the key namespace: a spelling never gets its
own colour or its own legend entry. A registered packet's definition is immutable once a results database has
recorded it; a changed meaning goes under a new key.

## Adding one

1. Add the entry after the existing ones of its kind.
2. Append its key to the matching tuple in `tests/test_vocabulary.py`, in the same commit.
3. For a model, add its serving layer (see [inference.md](inference.md)); for a packet, its pages
   (see [packets.md](packets.md)).

A plugin adds entries the same way.
