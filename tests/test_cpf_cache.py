# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The CPF cache serves exactly the form rendered for the exact question asked, or refuses by name.

Every consumer of a canonical parallel form -- the judge's route, the drop-in staging, the launch
gates -- reads through a view into the content-addressed cache. The failure modes these tests pin
are the ones a flat directory had: a key that drifts so a hit is never found or a stale one is, a
prefix match that hands one kernel another's source, a modified or half-written file served as ok,
and a miss that nobody can trace back to what was missing.
"""

import json
import os
import pathlib
import subprocess
import sys

import pytest

from hpcagent_bench import cpf_cache

OPTIONS = {
    "kernel": "k",
    "language": "c",
    "precision": "fp64",
    "target": "cpu",
    "mode": "form",
    "bridge": "b",
    "dace_env": {},
}

#: ``cache_key("sdfg", "dace", OPTIONS)``. A change to how keys are derived orphans every entry
#: every campaign has rendered, so it has to be a deliberate edit of this literal.
PINNED_KEY = "cced2bb37a9411bfa93129e591208cf79c5a369cf44e834a1556e29860a1279d"


def publish(cache: pathlib.Path, key: str, name: str, code: str = "void f(void) {}\n") -> None:
    stem = name.rsplit(".", 1)[0]
    cpf_cache.publish(cache, key, {"kernel": stem}, (name, code), (f"{stem}_binding.json", "{}\n"))


def view_with(tmp_path: pathlib.Path, kernel: str, dialect: str = "c", target: str = "cpu") -> pathlib.Path:
    """A view whose ``kernel`` entry points at a published read form and drop-in."""
    cache, view = tmp_path / "cache", tmp_path / "view"
    cpf_cache.open_view(view, cache, target, "dace")
    ext = cpf_cache.LANGUAGE_EXT[dialect]
    modes = {}
    for mode in cpf_cache.MODES:
        key = cpf_cache.cache_key("sdfg", "dace", {**OPTIONS, "kernel": kernel, "language": dialect, "mode": mode})
        publish(cache, key, f"{kernel}_fp64_cpf.{ext}", f"// {kernel} {mode}\n")
        modes[mode] = {"key": key, "verdict": "ok", "cached": False}
    cpf_cache.record(view, kernel, dialect, "fp64", modes)
    return view


def test_the_key_is_pinned_and_blind_to_dict_order() -> None:
    """The same question hashes to the same key however its options were assembled."""
    assert cpf_cache.cache_key("sdfg", "dace", OPTIONS) == PINNED_KEY
    shuffled = dict(reversed(list(OPTIONS.items())))
    assert cpf_cache.cache_key("sdfg", "dace", shuffled) == PINNED_KEY


def test_the_key_is_the_same_in_every_interpreter() -> None:
    """Prerender ranks and the judge are different processes; str hashing must not reach the key."""
    code = f"from hpcagent_bench import cpf_cache; print(cpf_cache.cache_key('sdfg', 'dace', {OPTIONS!r}))"
    for seed in ("1", "12345"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        done = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
        assert done.stdout.strip() == PINNED_KEY


@pytest.mark.parametrize(
    ("sdfg", "dace", "change"),
    [
        ("other", "dace", {}),
        ("sdfg", "other", {}),
        ("sdfg", "dace", {"language": "c++"}),
        ("sdfg", "dace", {"precision": "fp32"}),
        ("sdfg", "dace", {"target": "gpu"}),
        ("sdfg", "dace", {"mode": "dropin"}),
        ("sdfg", "dace", {"bridge": "edited"}),
        ("sdfg", "dace", {"dace_env": {"DACE_compiler_cpu_openmp_sections": "0"}}),
        ("sdfg", "dace", {"abi_order": ["a", "workspace", "workspace_size"]}),
    ],
)
def test_every_input_moves_the_key(sdfg: str, dace: str, change: dict[str, object]) -> None:
    """An input the key ignores is an input whose change serves a stale form as a hit."""
    assert cpf_cache.cache_key(sdfg, dace, {**OPTIONS, **change}) != PINNED_KEY


def test_the_source_digest_follows_content_not_location(tmp_path: pathlib.Path) -> None:
    """A snapshot and a checkout of the same dace hit the same entries; an edit misses them."""
    for tree in ("one", "two"):
        package = tmp_path / tree / "dace"
        (package / "codegen").mkdir(parents=True)
        (package / "codegen" / "cpf.py").write_text("RENDER = 1\n")
        (package / "config_schema.yml").write_text("a: 1\n")
        (package / "external" / "cub").mkdir(parents=True)
        (package / "__pycache__").mkdir()
    (tmp_path / "two" / "dace" / "external" / "cub" / "vendored.py").write_text("x = 1\n")
    (tmp_path / "two" / "dace" / "__pycache__" / "cpf.py").write_text("stale\n")
    one = cpf_cache.source_digest(tmp_path / "one" / "dace")
    assert cpf_cache.source_digest(tmp_path / "two" / "dace") == one
    (tmp_path / "two" / "dace" / "codegen" / "cpf.py").write_text("RENDER = 2\n")
    assert cpf_cache.source_digest(tmp_path / "two" / "dace") != one


def test_a_published_key_is_a_hit_and_publishing_it_again_writes_nothing(tmp_path: pathlib.Path) -> None:
    cache = tmp_path / "cache"
    assert not cpf_cache.is_hit(cache, PINNED_KEY)
    assert cpf_cache.publish(
        cache, PINNED_KEY, {"kernel": "k"}, ("k_fp64_cpf.c", "one\n"), ("k_fp64_cpf_binding.json", "{}")
    )
    entry = cpf_cache.entry_path(cache, PINNED_KEY)
    before = {path.name: path.stat().st_mtime_ns for path in entry.iterdir()}
    assert cpf_cache.is_hit(cache, PINNED_KEY)
    assert not cpf_cache.publish(
        cache, PINNED_KEY, {"kernel": "k"}, ("k_fp64_cpf.c", "two\n"), ("k_fp64_cpf_binding.json", "{}")
    )
    assert {path.name: path.stat().st_mtime_ns for path in entry.iterdir()} == before
    assert (entry / "k_fp64_cpf.c").read_text() == "one\n"
    assert not [path for path in entry.parent.iterdir() if path.name.startswith(".")], "staging left behind"


def test_the_manifest_carries_the_key_and_no_timestamp(tmp_path: pathlib.Path) -> None:
    """Two renders of the same inputs must write byte-identical manifests."""
    first, second = tmp_path / "a", tmp_path / "b"
    for cache in (first, second):
        cpf_cache.publish(
            cache, PINNED_KEY, {"dace_commit": "abc"}, ("k_fp64_cpf.c", "x\n"), ("k_fp64_cpf_binding.json", "{}")
        )
    manifests = [(cpf_cache.entry_path(c, PINNED_KEY) / cpf_cache.MANIFEST_NAME).read_bytes() for c in (first, second)]
    assert manifests[0] == manifests[1]
    manifest = json.loads(manifests[0])
    assert manifest["key"] == PINNED_KEY and manifest["dace_commit"] == "abc"


def test_a_modified_artefact_is_a_miss_naming_the_key(tmp_path: pathlib.Path) -> None:
    view = view_with(tmp_path, "k")
    source, _ = cpf_cache.resolve(view, "k", "c", "fp64", "form")
    source.write_text("// edited by hand\n")
    with pytest.raises(cpf_cache.CacheMiss, match=source.parent.name):
        cpf_cache.resolve(view, "k", "c", "fp64", "form")


def test_a_pointer_to_a_missing_entry_names_the_key(tmp_path: pathlib.Path) -> None:
    view = view_with(tmp_path, "k")
    source, _ = cpf_cache.resolve(view, "k", "c", "fp64", "dropin")
    key = source.parent.name
    for path in source.parent.iterdir():
        path.unlink()
    source.parent.rmdir()
    with pytest.raises(cpf_cache.CacheMiss, match=key):
        cpf_cache.resolve(view, "k", "c", "fp64", "dropin")


def test_a_failed_render_is_a_miss_naming_its_key_and_error(tmp_path: pathlib.Path) -> None:
    cache, view = tmp_path / "cache", tmp_path / "view"
    cpf_cache.open_view(view, cache, "cpu", "dace")
    failed = {"key": PINNED_KEY, "verdict": "timeout", "error": "render exceeded 14400s"}
    cpf_cache.record(view, "cloudsc", "c", "fp64", {"form": failed, "dropin": failed})
    with pytest.raises(cpf_cache.CacheMiss) as caught:
        cpf_cache.resolve(view, "cloudsc", "c", "fp64", "form")
    assert PINNED_KEY in str(caught.value) and "14400s" in str(caught.value)


def test_a_kernel_nothing_was_rendered_for_names_the_entry(tmp_path: pathlib.Path) -> None:
    view = view_with(tmp_path, "k")
    with pytest.raises(cpf_cache.CacheMiss, match="other_fp64_cpf.c.json"):
        cpf_cache.resolve(view, "other", "c", "fp64", "form")


def test_a_flat_directory_of_forms_is_not_a_view(tmp_path: pathlib.Path) -> None:
    """A directory of loose files carries no key, so nothing in it can be traced or trusted."""
    (tmp_path / "cloudsc_fp64_cpf.c").write_text("// loose\n")
    with pytest.raises(cpf_cache.CacheMiss, match="not a CPF cache view"):
        cpf_cache.resolve(tmp_path, "cloudsc", "c", "fp64", "form")


def test_lookup_is_by_exact_name(tmp_path: pathlib.Path) -> None:
    """cloudsc_init sits beside cloudsc; a request for cloudsc must never be answered with it."""
    view = view_with(tmp_path, "cloudsc_init")
    with pytest.raises(cpf_cache.CacheMiss):
        cpf_cache.resolve(view, "cloudsc", "c", "fp64", "form")
    source, _ = cpf_cache.resolve(view, "cloudsc_init", "c", "fp64", "form")
    assert source.read_text() == "// cloudsc_init form\n"


def test_a_registry_key_resolves_like_its_short_name(tmp_path: pathlib.Path) -> None:
    view = view_with(tmp_path, "cloudsc")
    source, _ = cpf_cache.resolve(view, "scientific_computing/weather/cloudsc/cloudsc", "c", "fp64", "form")
    assert source.read_text() == "// cloudsc form\n"


def test_a_gpu_view_serves_the_device_form_for_a_host_dialect(tmp_path: pathlib.Path) -> None:
    """The tool asks a hip arm's judge for c++; the device form is the only one a gpu view holds."""
    view = view_with(tmp_path, "k", dialect="hip", target="gpu")
    source, _ = cpf_cache.resolve(view, "k", "c++", "fp64", "form")
    assert source.suffix == ".hip"


def test_a_view_refuses_a_second_renderer(tmp_path: pathlib.Path) -> None:
    """One campaign's view must not hold forms from two dace trees or two targets."""
    cpf_cache.open_view(tmp_path / "view", tmp_path / "cache", "cpu", "dace-one")
    cpf_cache.open_view(tmp_path / "view", tmp_path / "cache", "cpu", "dace-one")
    with pytest.raises(ValueError, match="new view"):
        cpf_cache.open_view(tmp_path / "view", tmp_path / "cache", "cpu", "dace-two")
    with pytest.raises(ValueError, match="new view"):
        cpf_cache.open_view(tmp_path / "view", tmp_path / "cache", "gpu", "dace-one")


def test_check_lists_every_miss_and_fails(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    view = view_with(tmp_path, "k")
    common = ["--view", str(view), "--language", "c", "--mode", "dropin"]
    assert cpf_cache.main(["check", "--kernels", "k", *common]) == 0
    assert capsys.readouterr().out == ""
    assert cpf_cache.main(["check", "--kernels", "k,absent", *common]) == 1
    assert "absent_fp64_cpf.c.json" in capsys.readouterr().out


def test_check_prints_one_line_per_missing_kernel(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    view = view_with(tmp_path, "k")
    assert (
        cpf_cache.main(["check", "--view", str(view), "--language", "c", "--mode", "form", "--kernels", "k,a,b"]) == 1
    )
    lines = capsys.readouterr().out.splitlines()
    assert [line.split(":", 1)[0] for line in lines] == ["a", "b"]


def test_stage_copies_the_dropin_under_the_task_basename(tmp_path: pathlib.Path) -> None:
    view = view_with(tmp_path, "k")
    dest = tmp_path / "tasks" / "k"
    assert cpf_cache.main(["stage", "--view", str(view), "--kernel", "k", "--language", "c", "--dest", str(dest)]) == 0
    assert [p.name for p in dest.iterdir()] == ["k.c"]
    assert (dest / "k.c").read_text() == "// k dropin\n"


def test_stage_refuses_a_kernel_the_view_cannot_serve(tmp_path: pathlib.Path) -> None:
    view = view_with(tmp_path, "k")
    dest = tmp_path / "tasks" / "absent"
    assert (
        cpf_cache.main(["stage", "--view", str(view), "--kernel", "absent", "--language", "cpp", "--dest", str(dest)])
        == 1
    )
    assert not dest.exists()
