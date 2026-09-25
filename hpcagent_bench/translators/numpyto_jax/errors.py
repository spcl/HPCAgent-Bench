"""The one exception the JAX emitter raises."""


class EmitError(Exception):
    """A numpy construct the prototype does not (yet) lower."""
