"""Framework base class + registry.

A new framework is one Python file under :mod:`optarena.frameworks`
carrying ``@register_framework("<name>")`` on a :class:`Framework`
subclass. Path-glob discovery imports every sibling module on first
access to :data:`FRAMEWORKS`.

This module intentionally does NOT depend on the legacy
:mod:`optarena.infrastructure.framework` module so the migration can
proceed in two steps: new frameworks register here, old frameworks
keep working via the legacy ``generate_framework`` factory, and the
two registries are merged at lookup time by the driver.
"""
import importlib
import pkgutil
from typing import Dict, Set, Type

from optarena.flags import Mode
from optarena.precision import Precision

# Timing lives in optarena.infrastructure.framework (the live execution base:
# Timer + create/start/stop/free_timer + measure, in milliseconds). This
# registry module is discovery-only (compile_args / env / supports / precision)
# and intentionally carries NO timer lifecycle of its own -- when a registry
# framework is wired to execution it inherits the single timer implementation
# from the infrastructure base, so there is exactly one source of truth.

#: Framework short name → subclass.
FRAMEWORKS: Dict[str, Type["Framework"]] = {}


def register_framework(name: str):
    """Decorator: register ``cls`` under ``name`` in :data:`FRAMEWORKS`."""

    def deco(cls: Type["Framework"]) -> Type["Framework"]:
        if name in FRAMEWORKS:
            raise ValueError(f"Framework {name!r} already registered "
                             f"by {FRAMEWORKS[name].__module__}")
        FRAMEWORKS[name] = cls
        cls._name = name
        return cls

    return deco


class Framework:
    """Base class for framework adapters.

    Subclasses set the four class attributes below and override
    :meth:`compile_args` (and optionally :meth:`env`) to translate a
    :class:`~optarena.flags.Mode` into the native config the underlying
    toolchain expects.

    :cvar full_name: Human-readable label shown in reports.
    :cvar postfix: Suffix used to locate the per-kernel implementation
        file (``<kernel>_<postfix>.py`` under the kernel's folder).
    :cvar arch: Either ``"cpu"`` or ``"gpu"``; drives the default Mode
        selection when none is provided by the driver.
    :cvar SUPPORTED_PRECISIONS: Set of precisions the framework can
        execute. The sweep driver skips any precision not in this set.
    """
    full_name: str = ""
    postfix: str = ""
    arch: str = "cpu"
    SUPPORTED_PRECISIONS: Set[Precision] = frozenset(
        {Precision.FP32, Precision.FP64})

    _name: str = ""

    @property
    def name(self) -> str:
        """Short name under which the framework is registered."""
        return self._name

    def compile_args(self, mode: Mode) -> str:
        """Return the compile-flag string for ``mode``.

        Frameworks that do not compile native code (numpy, pure-Python
        impls) may return an empty string.
        """
        return ""

    def env(self, mode: Mode) -> Dict[str, str]:
        """Return env vars to set for the run (thread counts, etc.).

        Default: forward :func:`optarena.flags.cpu_env` for the mode.
        """
        from optarena.flags import cpu_env
        return cpu_env(mode)

    def supports(self, precision: Precision) -> bool:
        """``True`` when ``precision`` is in :attr:`SUPPORTED_PRECISIONS`."""
        return precision in self.SUPPORTED_PRECISIONS

    def version(self) -> str:
        """Return the framework's package version (best-effort)."""
        try:
            from importlib.metadata import version
            return version(self._name)
        except Exception:
            return "external"


def _autoload():
    """Import every sibling under :mod:`optarena.frameworks` so each
    ``@register_framework`` decorator runs."""
    try:
        from optarena import frameworks as _pkg
    except ImportError:
        return
    for _, modname, _ in pkgutil.iter_modules(_pkg.__path__):
        importlib.import_module(f"{_pkg.__name__}.{modname}")


_autoload()
