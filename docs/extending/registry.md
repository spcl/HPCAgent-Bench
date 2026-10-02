# The registry: one decorator per kind

Everything the repository names and orders is registered by a class decorator, in the module that
owns the kind. The mechanism is `hpcagent_bench/registry.py`; each kind is a `Kind`: its fields, how
a decorated class becomes an entry, and the entries registered so far.

```python
@llm("qwen38", order=0, aliases=("qwen3.8",))
class Qwen38:
    name = "Qwen3.8-27B"
    serves = "Qwen/Qwen3.8-27B-FP8"
```

## What every decorator checks

The checks run when the decorator is applied, at import, never at first use:

- Every field without a default is required; every other attribute is optional with its default.
- An attribute the kind does not declare is refused (`lable = ...` is an error, not a setting that
  does nothing) and so is a value of the wrong type.
- A key, an alias or an `order` that is already taken is refused, and the registry is left as it was.
- `order` is a non-negative integer, or `None` for the one entry that has no slot (the no-packet control).

The cross-entry rules a single decorator cannot check run once everything is registered
(`vocabulary.check_vocabulary()`, called by `study_tags.registry()`): every kind but the packets gives
every entry a slot, the control is the only packet without one, a composed packet exists and a packet's
`device` is `cpu`, `amd` or `nvidia`.

## The slot

`order` is an explicit integer, unique within the kind, and it is the entry's slot. Import order never
matters. An existing entry's order never changes; a new entry takes the next free one
(`Kind.next_order()`). Never reuse the slot of a retired entry: keep the entry (a retired one stays
registered, and a packet marks it `frozen`) so the colours and shapes after it do not move.

What a slot decides (`hpcagent_bench/stats/palette.py` reads it; nothing else carries a colour):

| Kind | Slot decides |
|---|---|
| LLM | marker `order` of the marker pool; hue `order` of the tab20 ramp where a figure colours models |
| optimizer | marker `order` counted from the END of the pool, so a new LLM never repaints it; hue `order` |
| harness | the clearest shapes of the treatment pool, in slot order, ahead of every packet; hue `order` |
| packet | hue `order` (a combination wears its lead part's hue, lightened one step per extra part, and the lead is the part with the lowest slot) and the shape: the next free one of the treatment pool unless the class names a `marker` |
| language, device | hue `order` |

The colour ramp, the marker pools and the lightness step are the only data left in
`hpcagent_bench/envs/registry.yaml`. `tests/test_vocabulary.py` pins every slot by value and
`tests/test_palette.py` pins the resolved colours, so a change that would repaint a published figure
fails naming the entity.

## The kinds

| Decorator | Module | The class must provide | Registered as |
|---|---|---|---|
| `@llm(key, order=, aliases=)` | `hpcagent_bench/models.py` | `name` (display name), `serves` (the checkpoint the tag is expected to serve) | the model tag setups record |
| `@optimizer(key, order=, aliases=)` | `hpcagent_bench/models.py` | `name` | a compiler or pipeline that stands where an LLM stands (`dace`, `cpf`); device variants are aliases |
| `@harness(key, order=, aliases=)` | `hpcagent_bench/models.py` | `name` | the `harness` column value |
| `@language(key, order=, aliases=)` | `hpcagent_bench/models.py` | `name` (proper name: `C++`, not `Cpp`) | the language a setup asked for |
| `@device(key, order=, aliases=)` | `hpcagent_bench/models.py` | `name` | the `device` column value |
| `@packet(key, order=, aliases=)` | `hpcagent_bench/skill_packets.py` | `name`; optionally `skills`, `packets`, `tools`, `env`, `method`, `device`, `frozen`, `marker`, `short` | the `packet` column value; see [packets.md](packets.md) |

An alias resolves to the entity it names, takes no slot, and shares the key namespace: a spelling never
gets its own colour or its own legend entry. A registered packet's definition is immutable once a results
database has recorded it; a changed meaning goes under a new key.

## Adding one

1. Add the class, with `order=` the next free slot (`vocabulary.MODELS.next_order()` and so on).
2. Add its slot to the matching table in `tests/test_vocabulary.py`, in the same commit.
3. For a model, add its serving layer (see [inference.md](inference.md)); for a packet, its pages
   (see [packets.md](packets.md)).
