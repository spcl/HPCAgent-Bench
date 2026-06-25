"""Language registry + single-source compilation (Workstream F).

Adding a new native language to OptArena is, by design, two local edits and
nothing under ``optarena/NumpyTo*/`` (see the header of
``optarena/envs/compilers.yaml``):

1. one compiler block in ``compilers.yaml`` (with a ``baseline_ref`` naming a
   constant in :mod:`optarena.flags`),
2. one extension in :data:`LANG_EXT` here.

A kernel then opts in by listing the language in its manifest ``languages:``.
This module owns the second edit plus the runtime helpers:

* :func:`discover_variants` -- glob the per-kernel ``cpp_backend`` directory for
  emitted ``<short>_*_auto.<ext>`` files, filtered to the kernel's declared
  ``languages``.
* :func:`compile_variant` -- read ``compilers.yaml``, resolve the
  ``baseline_ref`` to its :mod:`optarena.flags` constant via ``vars(flags)[ref]``
  (the repo's no-``getattr`` rule), compose autopar / CUDA for the mode, and
  substitute the compile-command template. It returns the argv; it does NOT run
  it (the caller owns process launching).
"""
import pathlib
import shlex
from typing import Dict, List, Optional, Sequence, Tuple

import yaml

from optarena import flags, paths
from optarena.flags import Mode
from optarena.spec import BenchSpec

#: Repo-relative location of the flat per-compiler table.
COMPILERS_YAML: pathlib.Path = paths.ROOT / "optarena" / "envs" / "compilers.yaml"

#: Language token -> source-file extension (no leading dot). The fourth edit
#: when adding a language. Mirrors the per-language rendering in
#: ``abi_contract.md`` §7.
LANG_EXT: Dict[str, str] = {
    "c": "c",
    "cpp": "cpp",
    "fortran": "f90",
    # GPU implementation targets (host-pointer C-ABI entry; agent owns device
    # transfers + launch). nvcc/hipcc already in compilers.yaml.
    "cuda": "cu",
    "hip": "hip",
}


def _load_compilers() -> Dict[str, dict]:
    """Parse ``compilers.yaml`` into ``{compiler_name: block}``."""
    return yaml.safe_load(COMPILERS_YAML.read_text())


def _backend_dir(spec: BenchSpec) -> pathlib.Path:
    """The kernel's ``cpp_backend`` directory (where emits + builds live)."""
    return paths.BENCHMARKS / spec.relative_path / "cpp_backend"


def discover_variants(spec: BenchSpec) -> List[Tuple[str, pathlib.Path]]:
    """Return ``[(lang, source_path)]`` for the kernel's emitted variants.

    Globs ``cpp_backend/<short>_*_auto.<ext>`` for every extension in
    :data:`LANG_EXT`, then keeps only languages the kernel declares in
    ``spec.languages`` (an empty declaration means "no language restriction" --
    accept all discovered ones, the back-compat default). Results are sorted by
    ``(lang, filename)`` for determinism.
    """
    backend = _backend_dir(spec)
    allowed = set(spec.languages) if spec.languages else None
    found: List[Tuple[str, pathlib.Path]] = []
    if not backend.exists():
        return found
    for lang, ext in LANG_EXT.items():
        if allowed is not None and lang not in allowed:
            continue
        for src in sorted(backend.glob(f"{spec.short_name}_*_auto.{ext}")):
            found.append((lang, src))
    found.sort(key=lambda t: (t[0], t[1].name))
    return found


def _resolve_baseline(block: dict, mode: Mode) -> str:
    """Resolve a compiler block's flag string for ``mode``.

    ``baseline_ref`` names a constant in :mod:`optarena.flags`; we look it up via
    ``vars(flags)[ref]`` (NOT ``getattr`` -- the repo rule). CUDA blocks carry
    no baseline_ref and use :func:`flags.compose_cuda`; an ``autopar_ref`` (when
    present and the mode is multi-core) is appended via
    :func:`flags.compose_autopar`.
    """
    if block.get("cuda"):
        return flags.compose_cuda()
    if block.get("hip"):
        return flags.compose_hip()
    ref = block.get("baseline_ref")
    if ref is None:
        return ""
    flag_vars = vars(flags)
    if ref not in flag_vars:
        raise KeyError(f"baseline_ref {ref!r} is not a constant in optarena.flags")
    baseline = flag_vars[ref]
    autopar_ref = block.get("autopar_ref")
    autopar = flag_vars.get(autopar_ref) if autopar_ref else None
    return flags.compose_autopar(baseline, autopar, mode)


def _compiler_for_lang(compilers: Dict[str, dict], lang: str) -> Tuple[str, dict]:
    """Pick the first compiler block whose ``lang`` matches ``lang``."""
    for cname, block in compilers.items():
        if block.get("lang") == lang:
            return cname, block
    raise KeyError(f"no compiler in compilers.yaml emits lang {lang!r}")


def compile_variant(
    spec: BenchSpec,
    lang: str,
    mode: Mode = Mode.SINGLE_CORE,
    *,
    src: Optional[pathlib.Path] = None,
    compiler: Optional[str] = None,
) -> List[str]:
    """Build the compile argv for ``(spec, lang, mode)`` -- does NOT run it.

    :param spec: the kernel descriptor.
    :param lang: language token (key of :data:`LANG_EXT`).
    :param mode: evaluation mode (drives autopar / CUDA flag composition).
    :param src: explicit source path; defaults to the first variant
        :func:`discover_variants` finds for ``lang``.
    :param compiler: explicit ``compilers.yaml`` block name; defaults to the
        first block whose ``lang`` matches.
    :returns: the substituted compile command as an argv list.
    :raises KeyError: for an unknown language / compiler / baseline_ref.
    :raises FileNotFoundError: when no source can be resolved.
    """
    if lang not in LANG_EXT:
        raise KeyError(f"unknown language {lang!r}; expected one of "
                       f"{sorted(LANG_EXT)}")

    compilers = _load_compilers()
    if compiler is not None:
        if compiler not in compilers:
            raise KeyError(f"no such compiler {compiler!r} in compilers.yaml")
        block = compilers[compiler]
    else:
        compiler, block = _compiler_for_lang(compilers, lang)

    if src is None:
        variants = [p for (vl, p) in discover_variants(spec) if vl == lang]
        if not variants:
            raise FileNotFoundError(f"{spec.short_name}: no {lang} variant under "
                                    f"{_backend_dir(spec)}")
        src = variants[0]

    baseline = _resolve_baseline(block, mode)
    obj = src.with_suffix(".o")
    lib = _backend_dir(spec) / f"lib{spec.short_name}.so"

    subst = {
        "cc": block["cc"],
        "baseline": baseline,
        "src": str(src),
        "obj": str(obj),
        "objs": str(obj),
        "lib": str(lib),
    }

    cmd: List[str] = []
    for tok in block["compile"]:
        rendered = tok.format(**subst)
        # The {baseline} token expands to a flag string; shell-split it back into
        # argv items (shlex keeps a quoted group like nvcc's ``-Xcompiler='...'``
        # together; an empty baseline contributes nothing).
        if tok == "{baseline}":
            cmd.extend(shlex.split(rendered))
        else:
            cmd.append(rendered)
    return cmd


def build_kernel_lib_commands(
    sources: List[Tuple[str, pathlib.Path]],
    out_so: pathlib.Path,
    *,
    build_dir: Optional[pathlib.Path] = None,
    mode: Mode = Mode.SINGLE_CORE,
    compiler: Optional[str] = None,
    extra_flags: str = "",
) -> List[List[str]]:
    """Compile several ``(lang, src)`` pairs and link them into ONE ``out_so``.

    This is the shared-``cpp_backend`` build path that replaces the per-kernel
    ``CMakeLists.txt`` the foundation flatten dropped: a foundation kernel's
    several precision/backend sources (``<short>_d.cpp``, ``<short>_d.c``,
    ``<short>_f.cpp``, ...) carry distinct symbol suffixes and link into a
    single ``lib<short>.so`` that :func:`optarena.benchmarks.cpp_runtime.\
wrap_kernel` dlopens. Flags resolve from :mod:`optarena.flags` via
    ``compilers.yaml`` (no literal optimization flags -- the same matrix the rest
    of the harness uses).

    :param sources: ``(lang, source_path)`` pairs; ``c`` -> the C compiler,
        ``cpp`` -> the C++ compiler (chosen per source by ``lang``).
    :param out_so: the shared library to produce.
    :param build_dir: where object files land (defaults to ``out_so``'s
        parent). Object names embed the source filename *including* its
        extension, so a ``.c``/``.cpp`` pair sharing a stem does not collide.
    :param mode: evaluation mode (drives autopar flag composition).
    :param compiler: force a specific ``compilers.yaml`` block for every source
        + the link step (e.g. ``clangpp`` for the Polly/Pluto presets, which are
        clang-only) instead of picking the first block per language.
    :param extra_flags: a flag string appended to every compile baseline and to
        the link command (the Polly/Pluto preset deltas from :mod:`optarena.flags`).
    :returns: argv lists to run in order; the last produces ``out_so``.
    :raises ValueError: when ``sources`` is empty.
    :raises KeyError: for an unknown language.
    """
    if not sources:
        raise ValueError("build_kernel_lib_commands: no sources to compile")
    compilers = _load_compilers()
    out_so = pathlib.Path(out_so)
    build_dir = pathlib.Path(build_dir) if build_dir is not None else out_so.parent

    def render(tokens: List[str], subst: Dict[str, str]) -> List[str]:
        out: List[str] = []
        for tok in tokens:
            rendered = tok.format(**subst)
            # {baseline} and {objs} each expand to a space-joined string that
            # must become several argv items; everything else is one token.
            if tok in ("{baseline}", "{objs}"):
                out.extend(shlex.split(rendered))
            else:
                out.append(rendered)
        return out

    forced = None
    if compiler is not None:
        if compiler not in compilers:
            raise KeyError(f"no such compiler {compiler!r} in compilers.yaml")
        forced = compilers[compiler]

    cmds: List[List[str]] = []
    objs: List[str] = []
    has_cpp = has_fortran = False
    for lang, src in sources:
        if lang not in LANG_EXT:
            raise KeyError(f"unknown language {lang!r}; expected one of {sorted(LANG_EXT)}")
        block = forced if forced is not None else _compiler_for_lang(compilers, lang)[1]
        src = pathlib.Path(src)
        obj = build_dir / f"{src.name}.o"
        baseline = _resolve_baseline(block, mode)
        subst = {
            "cc": block["cc"],
            "baseline": f"{baseline} {extra_flags}".strip() if extra_flags else baseline,
            "src": str(src),
            "obj": str(obj),
            "objs": str(obj),
            "lib": str(out_so),
        }
        cmds.append(render(block["compile"], subst))
        objs.append(str(obj))
        has_cpp = has_cpp or lang == "cpp"
        has_fortran = has_fortran or lang == "fortran"

    # Pick the link driver that pulls in the right runtime: a forced compiler
    # wins (Polly/Pluto link with clang); else gfortran for Fortran objects
    # (libgfortran -- e.g. system_clock in the self-timing), else the C++ driver
    # when any C++ object is present, else the C driver.
    if forced is not None:
        link_block = forced
    else:
        link_lang = "fortran" if has_fortran else ("cpp" if has_cpp else "c")
        _, link_block = _compiler_for_lang(compilers, link_lang)
    link_subst = {
        "cc": link_block["cc"],
        "baseline": "",
        "src": "",
        "obj": "",
        "objs": " ".join(objs),
        "lib": str(out_so),
    }
    link_argv = render(link_block["link"], link_subst)
    link_argv.extend(link_block.get("link_extra") or [])
    if extra_flags:  # Polly/Pluto need -fopenmp -lgomp at link too
        link_argv.extend(shlex.split(extra_flags))
    cmds.append(link_argv)
    return cmds


def build_shared_lib_commands(
        lang: str,
        src: pathlib.Path,
        out_so: pathlib.Path,
        *,
        mode: Mode = Mode.SINGLE_CORE,
        compiler: Optional[str] = None,
        extra_compile: Sequence[str] = (),
        extra_link: Sequence[str] = (),
) -> List[List[str]]:
    """Compile+link argv(s) that turn one source file into ``out_so`` -- the
    sandbox path (caller-chosen, workdir-local paths; the repo tree is untouched).

    Unlike :func:`compile_variant` (which targets the in-repo ``cpp_backend``
    and returns only the compile step), this emits the FULL chain for an
    arbitrary source/output location, still entirely matrix-driven (flags resolve
    from :mod:`optarena.flags` via ``compilers.yaml``):

    * a language whose ``compile`` template writes the ``.so`` directly returns
      a single argv;
    * the rest return ``[compile -> .o, link -> .so]`` and apply any
      ``link_extra`` (e.g. gfortran's ``-lgfortran``).

    ``extra_compile`` (e.g. ``-I`` include dirs, ``-D`` defines) are appended to
    the COMPILE argv and ``extra_link`` (e.g. ``-L``/``-lopenblas``) to the LINK
    argv -- for building against an external dependency. Every block is two-step
    (compile -> ``.o``, link -> ``.so``), so the two sets must NOT be conflated:
    a ``-I`` on the link step or a ``-l`` on the compile step is silently
    ineffective. The optimization flags still come entirely from the matrix; the
    caller restricts these to dependency tokens (see
    :func:`optarena.agent_bench.sandbox.split_build`).

    :returns: a list of argv lists to run in order; the last produces ``out_so``.
    """
    if lang not in LANG_EXT:
        raise KeyError(f"unknown language {lang!r}; expected one of {sorted(LANG_EXT)}")
    compilers = _load_compilers()
    if compiler is not None:
        if compiler not in compilers:
            raise KeyError(f"no such compiler {compiler!r} in compilers.yaml")
        block = compilers[compiler]
    else:
        compiler, block = _compiler_for_lang(compilers, lang)

    src = pathlib.Path(src)
    out_so = pathlib.Path(out_so)
    # Extension-inclusive object name (foo.c.o, not foo.o) so a .c and .cpp
    # sharing a stem in one workdir do not clobber each other's object.
    obj = src.with_name(src.name + ".o")
    baseline = _resolve_baseline(block, mode)
    subst = {
        "cc": block["cc"],
        "baseline": baseline,
        "src": str(src),
        "obj": str(obj),
        "objs": str(obj),
        "lib": str(out_so),
    }

    def render(tokens: List[str]) -> List[str]:
        out: List[str] = []
        for tok in tokens:
            rendered = tok.format(**subst)
            if tok == "{baseline}":
                out.extend(shlex.split(rendered))  # shell-split (keeps quoted -Xcompiler groups)
            else:
                out.append(rendered)
        return out

    cmds: List[List[str]] = [render(block["compile"])]
    if extra_compile:
        cmds[0].extend(extra_compile)  # first argv compiles the source (sees -I/-D)
    link = block.get("link")
    if link:
        link_argv = render(link)
        link_argv.extend(block.get("link_extra") or [])
        # An OpenMP-parallelized object (multi-core / autopar baseline carries
        # -fopenmp) emits GOMP_* references that must also be resolved at link;
        # the link template carries no {baseline}, so propagate -fopenmp here.
        if "-fopenmp" in baseline and "-fopenmp" not in link_argv:
            link_argv.append("-fopenmp")
        cmds.append(link_argv)
    if extra_link:
        cmds[-1].extend(extra_link)  # final argv produces the .so (sees -L/-l)
    return cmds
