"""Per-emit state of the JAX emitter, shared by its modules.

``emit_jax`` resets it at the start of each kernel and ``emit_function`` at the start of each
function, so the emitted text never depends on what was emitted before.
"""

import dataclasses


@dataclasses.dataclass(slots=True)
class EmitState:
    #: Emitting the jit form: a data-dependent ternary lowers to ``jnp.where`` (a traced ternary
    #: cannot yield a concrete bool); eager mode leaves it verbatim.
    jit_mode: bool = False
    #: Temp-name counter of the ``__tup<k>`` / ``__pa<k>`` unpack splits; unique within one kernel.
    tuple_ctr: int = 0
    #: Temp-name counter of the ``__chain<k>`` chained-assign splits; unique within one kernel.
    chain_ctr: int = 0
    #: Static (concrete at trace time) names of the function being emitted.
    emit_static: set[str] = dataclasses.field(default_factory=set)
    #: Module-level constant names carried into the emitted module (weather stencils' ``BET_M``).
    module_consts: set[str] = dataclasses.field(default_factory=set)
    #: Scalar subset of ``module_consts`` mapped to its concrete Python value, for branch folding.
    module_const_values: dict[str, object] = dataclasses.field(default_factory=dict)
    #: Function-local names bound once to a constant-literal sequence (lulesh's ``faces``).
    local_consts: set[str] = dataclasses.field(default_factory=set)
    #: Names aliasing ``scipy.linalg.eigh`` in the current module.
    eigh_aliases: set[str] = dataclasses.field(default_factory=set)


STATE = EmitState()
