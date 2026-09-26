"""The one exception the JAX emitter raises."""

__all__ = ["EmitError"]


class EmitError(Exception):
    """A numpy construct the prototype does not (yet) lower."""
