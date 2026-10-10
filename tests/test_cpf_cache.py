# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
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
import re
import subprocess
import sys
import threading

import pytest

from hpcagent_bench import cpf_cache

OPTIONS = {
    "kernel": "k",
    "language": "c",
    "precision": "fp64",
    "target": "cpu",
    "bridge": "b",
    "dace_env": {},
}

#: ``cache_key("sdfg", "dace", OPTIONS)``. A change to how keys are derived orphans every entry
#: every experiment has rendered, so it has to be a deliberate edit of this literal.
PINNED_KEY = "b46e3a8171163d366d472af313edd6e819d4227d3b8c2d882962a2d93fb6d63b"


def publish(cache: pathlib.Path, key: str, name: str, code: str = "void f(void) {}\n", device: bool = False) -> None:
    """One entry: ``name`` as the source, and with ``device`` a gpu form's ``<stem>.hip`` device unit beside it."""
    stem = name.rsplit(".", 1)[0]
    units = [("source", name, code)] + ([("device", f"{stem}.hip", f"// kernels of {code}")] if device else [])
    cpf_cache.publish(
        cache, key, {"kernel": stem, "entry": f"{stem}_entry"}, [*units, ("binding", f"{stem}_binding.json", "{}\n")]
    )


def view_with(tmp_path: pathlib.Path, kernel: str, dialect: str = "c", target: str = "cpu") -> pathlib.Path:
    """A view whose ``kernel`` entry points at a published form."""
    cache, view = tmp_path / "cache", tmp_path / "view"
    cpf_cache.open_view(view, cache, target, "dace")
    # A gpu form is a host .cpp and a device .hip.
    ext = cpf_cache.LANGUAGE_EXT["c++" if target == "gpu" else dialect]
    key = cpf_cache.cache_key("sdfg", "dace", {**OPTIONS, "kernel": kernel, "language": dialect})
    publish(cache, key, f"{kernel}_fp64_cpf.{ext}", f"// {kernel} form\n", device=target == "gpu")
    cpf_cache.record(view, kernel, dialect, "fp64", {"key": key, "verdict": "ok", "cached": False})
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
        ("sdfg", "dace", {"bridge": "edited"}),
        ("sdfg", "dace", {"dace_env": {"DACE_compiler_cpu_openmp_sections": "0"}}),
        ("sdfg", "dace", {"signature": ["double *restrict a", "const int64_t N"]}),
    ],
)
def test_every_input_moves_the_key(sdfg: str, dace: str, change: dict[str, object]) -> None:
    """An input the key ignores is an input whose change serves a stale form as a hit."""
    assert cpf_cache.cache_key(sdfg, dace, {**OPTIONS, **change}) != PINNED_KEY


def test_a_published_key_is_a_hit_and_publishing_it_again_writes_nothing(tmp_path: pathlib.Path) -> None:
    cache = tmp_path / "cache"
    assert not cpf_cache.is_hit(cache, PINNED_KEY)
    assert cpf_cache.publish(
        cache,
        PINNED_KEY,
        {"kernel": "k"},
        [("source", "k_fp64_cpf.c", "one\n"), ("binding", "k_fp64_cpf_binding.json", "{}")],
    )
    entry = cpf_cache.entry_path(cache, PINNED_KEY)
    before = {path.name: path.stat().st_mtime_ns for path in entry.iterdir()}
    assert cpf_cache.is_hit(cache, PINNED_KEY)
    assert not cpf_cache.publish(
        cache,
        PINNED_KEY,
        {"kernel": "k"},
        [("source", "k_fp64_cpf.c", "two\n"), ("binding", "k_fp64_cpf_binding.json", "{}")],
    )
    assert {path.name: path.stat().st_mtime_ns for path in entry.iterdir()} == before
    assert (entry / "k_fp64_cpf.c").read_text() == "one\n"
    assert not [path for path in entry.parent.iterdir() if path.name.startswith(".")], "staging left behind"


def test_concurrent_publishes_to_one_key_never_let_a_reader_see_a_mismatched_entry(tmp_path: pathlib.Path) -> None:
    """publish() assembles an entry in a SIBLING staging directory and renames it into place, "so a
    reader sees a whole entry or none" (its own docstring). Several rendering ranks can legitimately
    race to publish the same key (the prepare job shards by kernel, not by (kernel, language); two
    prepare jobs with overlapping tags race the same way), so a reader hammering the
    cache throughout that race must only ever see CacheMiss or a fully self-consistent, hash-verified
    entry -- never a manifest paired with another writer's bytes -- and a writer must never crash on
    a directory a sibling publish() call is mid-rewrite of."""
    cache = tmp_path / "cache"
    stop = threading.Event()
    bad: list[str] = []

    def writer(index: int) -> None:
        for attempt in range(40):
            try:
                cpf_cache.publish(
                    cache,
                    PINNED_KEY,
                    {"kernel": "k"},
                    [
                        ("source", "k_fp64_cpf.c", f"// writer {index} attempt {attempt}\n"),
                        ("binding", "k_fp64_cpf_binding.json", f'{{"writer": {index}}}'),
                    ],
                )
            except Exception as exc:  # noqa: BLE001 -- publish() must never raise out of a normal race
                bad.append(f"writer {index}: {type(exc).__name__}: {exc}")

    def reader() -> None:
        while not stop.is_set():
            try:
                if cpf_cache.is_hit(cache, PINNED_KEY):
                    cpf_cache.verified_manifest(cache, PINNED_KEY)
            except cpf_cache.CacheMiss:
                pass  # expected: the entry was mid-replacement when this read landed
            except Exception as exc:  # noqa: BLE001 -- anything else is the corruption this test guards against
                bad.append(f"reader: {type(exc).__name__}: {exc}")

    reader_thread = threading.Thread(target=reader)
    reader_thread.start()
    writers = [threading.Thread(target=writer, args=(i,)) for i in range(6)]
    for t in writers:
        t.start()
    for t in writers:
        t.join()
    stop.set()
    reader_thread.join()

    assert bad == [], bad
    # The race must settle: after every writer is done, the cache holds one complete, verifiable entry.
    assert cpf_cache.is_hit(cache, PINNED_KEY)
    cpf_cache.verified_manifest(cache, PINNED_KEY)
    entry_dir = cpf_cache.entry_path(cache, PINNED_KEY)
    assert not [p for p in entry_dir.parent.iterdir() if p.name.startswith(".")], "staging left behind"


def test_an_entry_that_fails_verification_is_replaced_by_the_next_publish(tmp_path: pathlib.Path) -> None:
    """Moving a damaged entry aside instead of deleting it in place must still replace it, or a
    corrupted render would be a permanent miss that no rerun repairs."""
    cache = tmp_path / "cache"
    cpf_cache.publish(
        cache,
        PINNED_KEY,
        {"kernel": "k"},
        [("source", "k_fp64_cpf.c", "one\n"), ("binding", "k_fp64_cpf_binding.json", "{}")],
    )
    entry = cpf_cache.entry_path(cache, PINNED_KEY)
    (entry / "k_fp64_cpf.c").write_text("damaged\n")
    assert cpf_cache.publish(
        cache,
        PINNED_KEY,
        {"kernel": "k"},
        [("source", "k_fp64_cpf.c", "two\n"), ("binding", "k_fp64_cpf_binding.json", "{}")],
    )
    assert (entry / "k_fp64_cpf.c").read_text() == "two\n"
    assert cpf_cache.is_hit(cache, PINNED_KEY)
    assert not [p for p in entry.parent.iterdir() if p.name.startswith(".")], "moved-aside entry left behind"


def test_a_valid_canonical_entry_is_kept_when_a_sibling_publishes_the_same_key(tmp_path: pathlib.Path) -> None:
    """Keys are content addresses: the entry already in place is another rank's equal result, and
    rewriting it under a concurrent reader is the race that crashed publish."""
    cache, first, second = tmp_path / "cache", tmp_path / "first.sdfgz", tmp_path / "second.sdfgz"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    key = cpf_cache.canonical_key("program", "commit", {"target": "cpu"})
    cpf_cache.publish_canonical(cache, key, {"verdict": "ok"}, first)
    cpf_cache.publish_canonical(cache, key, {"verdict": "ok"}, second)
    entry = cpf_cache.canonical_entry(cache, key)
    assert entry is not None, entry
    assert entry[1] is not None, entry
    assert entry[1].read_bytes() == b"first", entry
    parent = cpf_cache.canonical_path(cache, key).parent
    assert not [p for p in parent.iterdir() if p.name.startswith(".")], "staging left behind"


def test_the_manifest_carries_the_key_and_no_timestamp(tmp_path: pathlib.Path) -> None:
    """Two renders of the same inputs must write byte-identical manifests."""
    first, second = tmp_path / "a", tmp_path / "b"
    for cache in (first, second):
        cpf_cache.publish(
            cache,
            PINNED_KEY,
            {"dace_commit": "abc"},
            [("source", "k_fp64_cpf.c", "x\n"), ("binding", "k_fp64_cpf_binding.json", "{}")],
        )
    manifests = [(cpf_cache.entry_path(c, PINNED_KEY) / cpf_cache.MANIFEST_NAME).read_bytes() for c in (first, second)]
    assert manifests[0] == manifests[1]
    manifest = json.loads(manifests[0])
    assert manifest["key"] == PINNED_KEY
    assert manifest["dace_commit"] == "abc"


def test_a_modified_artefact_is_a_miss_naming_the_key(tmp_path: pathlib.Path) -> None:
    view = view_with(tmp_path, "k")
    source = cpf_cache.resolve(view, "k", "c", "fp64").source
    source.write_text("// edited by hand\n")
    with pytest.raises(cpf_cache.CacheMiss, match=source.parent.name):
        cpf_cache.resolve(view, "k", "c", "fp64")


def test_a_canonical_entry_serves_its_sdfg_until_the_file_is_modified(tmp_path: pathlib.Path) -> None:
    """A damaged stored SDFG must be produced again, never loaded and rendered."""
    cache, sdfg = tmp_path / "cache", tmp_path / "canonical.sdfgz"
    sdfg.write_bytes(b"sdfg")
    key = cpf_cache.canonical_key("program", "commit", {"target": "cpu"})
    cpf_cache.publish_canonical(cache, key, {"verdict": "ok"}, sdfg)
    entry = cpf_cache.canonical_entry(cache, key)
    assert entry is not None, entry
    assert entry[1] is not None, entry
    assert entry[1].read_bytes() == b"sdfg", entry
    entry[1].write_bytes(b"edited")
    assert cpf_cache.canonical_entry(cache, key) is None


def test_a_cached_canonicalize_failure_is_served_without_a_file(tmp_path: pathlib.Path) -> None:
    cache = tmp_path / "cache"
    key = cpf_cache.canonical_key("program", "commit", {"target": "cpu"})
    cpf_cache.publish_canonical(cache, key, {"verdict": "fail", "error": "ValueError: cannot lift"}, None)
    manifest = {"verdict": "fail", "error": "ValueError: cannot lift", "key": key, "layout": cpf_cache.LAYOUT}
    assert cpf_cache.canonical_entry(cache, key) == (manifest, None)


@pytest.mark.parametrize(
    ("program", "commit", "options"),
    [
        ("other", "commit", {"target": "cpu"}),
        ("program", "moved", {"target": "cpu"}),
        ("program", "commit", {"target": "gpu"}),
    ],
)
def test_every_input_moves_the_canonical_key(program: str, commit: str, options: dict[str, object]) -> None:
    """A regenerated program or a moved dace extended must canonicalize again, not hit the old SDFG."""
    assert cpf_cache.canonical_key(program, commit, options) != cpf_cache.canonical_key(
        "program", "commit", {"target": "cpu"}
    )


def test_a_pointer_to_a_missing_entry_names_the_key(tmp_path: pathlib.Path) -> None:
    view = view_with(tmp_path, "k")
    source = cpf_cache.resolve(view, "k", "c", "fp64").source
    key = source.parent.name
    for path in source.parent.iterdir():
        path.unlink()
    source.parent.rmdir()
    with pytest.raises(cpf_cache.CacheMiss, match=key):
        cpf_cache.resolve(view, "k", "c", "fp64")


def test_a_failed_render_is_a_miss_naming_its_key_and_error(tmp_path: pathlib.Path) -> None:
    cache, view = tmp_path / "cache", tmp_path / "view"
    cpf_cache.open_view(view, cache, "cpu", "dace")
    failed = {"key": PINNED_KEY, "verdict": "timeout", "error": "render exceeded 14400s"}
    cpf_cache.record(view, "cloudsc", "c", "fp64", failed)
    with pytest.raises(cpf_cache.CacheMiss) as caught:
        cpf_cache.resolve(view, "cloudsc", "c", "fp64")
    assert PINNED_KEY in str(caught.value)
    assert "14400s" in str(caught.value)


def test_a_kernel_nothing_was_rendered_for_names_the_entry(tmp_path: pathlib.Path) -> None:
    view = view_with(tmp_path, "k")
    with pytest.raises(cpf_cache.CacheMiss, match=re.escape("other_fp64_cpf.c.json")):
        cpf_cache.resolve(view, "other", "c", "fp64")


def test_a_flat_directory_of_forms_is_not_a_view(tmp_path: pathlib.Path) -> None:
    """A directory of loose files carries no key, so nothing in it can be traced or trusted."""
    (tmp_path / "cloudsc_fp64_cpf.c").write_text("// loose\n")
    with pytest.raises(cpf_cache.CacheMiss, match="not a CPF cache view"):
        cpf_cache.resolve(tmp_path, "cloudsc", "c", "fp64")


def test_lookup_is_by_exact_name(tmp_path: pathlib.Path) -> None:
    """cloudsc_init sits beside cloudsc; a request for cloudsc must never be answered with it."""
    view = view_with(tmp_path, "cloudsc_init")
    with pytest.raises(cpf_cache.CacheMiss):
        cpf_cache.resolve(view, "cloudsc", "c", "fp64")
    source = cpf_cache.resolve(view, "cloudsc_init", "c", "fp64").source
    assert source.read_text() == "// cloudsc_init form\n"


def test_a_registry_key_resolves_like_its_short_name(tmp_path: pathlib.Path) -> None:
    view = view_with(tmp_path, "cloudsc")
    source = cpf_cache.resolve(view, "scientific_computing/weather/cloudsc/cloudsc", "c", "fp64").source
    assert source.read_text() == "// cloudsc form\n"


def test_a_gpu_view_serves_the_device_form_for_a_host_dialect(tmp_path: pathlib.Path) -> None:
    """The tool asks a hip setup's judge for c++; the device form is the only one a gpu view holds."""
    view = view_with(tmp_path, "k", dialect="hip", target="gpu")
    form = cpf_cache.resolve(view, "k", "c++", "fp64")
    assert (form.source.suffix, form.device and form.device.suffix) == (".cpp", ".hip")


def test_a_gpu_form_stages_as_the_two_units_a_gpu_submission_is(tmp_path: pathlib.Path) -> None:
    """agent/gpu-build.md: the host entry as ``<kernel>.cpp``, the kernels as ``<kernel>.hip``."""
    view = view_with(tmp_path, "k", dialect="hip", target="gpu")
    dest = tmp_path / "tasks" / "k"
    stage = ["stage", "--view", str(view), "--kernel", "k", "--language", "hip", "--target", "gpu"]
    assert cpf_cache.main([*stage, "--dest", str(dest), "--name", "k_reference"]) == 0
    assert sorted(path.name for path in dest.iterdir()) == ["k_reference.cpp", "k_reference.hip"]


def test_a_view_refuses_a_second_renderer(tmp_path: pathlib.Path) -> None:
    """One experiment's view must not hold forms from two dace trees or two targets."""
    cpf_cache.open_view(tmp_path / "view", tmp_path / "cache", "cpu", "dace-one")
    cpf_cache.open_view(tmp_path / "view", tmp_path / "cache", "cpu", "dace-one")
    with pytest.raises(ValueError, match="new view"):
        cpf_cache.open_view(tmp_path / "view", tmp_path / "cache", "cpu", "dace-two")
    with pytest.raises(ValueError, match="new view"):
        cpf_cache.open_view(tmp_path / "view", tmp_path / "cache", "gpu", "dace-one")


def test_check_lists_every_miss_and_fails(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    view = view_with(tmp_path, "k")
    common = ["--view", str(view), "--language", "c", "--target", "cpu"]
    assert cpf_cache.main(["check", "--kernels", "k", *common]) == 0
    assert capsys.readouterr().out == ""
    assert cpf_cache.main(["check", "--kernels", "k,absent", *common]) == 1
    assert "absent_fp64_cpf.c.json" in capsys.readouterr().out


def test_check_prints_one_line_per_missing_kernel(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    view = view_with(tmp_path, "k")
    check = ["check", "--view", str(view), "--language", "c", "--target", "cpu"]
    assert cpf_cache.main([*check, "--kernels", "k,a,b"]) == 1
    lines = capsys.readouterr().out.splitlines()
    assert [line.split(":", 1)[0] for line in lines] == ["a", "b"]


def test_stage_copies_the_form_under_the_task_basename(tmp_path: pathlib.Path) -> None:
    view = view_with(tmp_path, "k")
    dest = tmp_path / "tasks" / "k"
    stage = ["stage", "--view", str(view), "--kernel", "k", "--language", "c", "--target", "cpu"]
    assert cpf_cache.main([*stage, "--dest", str(dest)]) == 0
    assert [p.name for p in dest.iterdir()] == ["k.c"]
    assert (dest / "k.c").read_text() == "// k form\n"


def test_stage_refuses_a_kernel_the_view_cannot_serve(tmp_path: pathlib.Path) -> None:
    view = view_with(tmp_path, "k")
    dest = tmp_path / "tasks" / "absent"
    stage = ["stage", "--view", str(view), "--kernel", "absent", "--language", "cpp", "--target", "cpu"]
    assert cpf_cache.main([*stage, "--dest", str(dest)]) == 1
    assert not dest.exists()


@pytest.mark.parametrize(
    ("dialect", "held", "asked", "language"), [("hip", "gpu", "cpu", "c++"), ("c", "cpu", "gpu", "c")]
)
def test_a_view_of_the_other_target_fails_the_check_whole(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str], dialect: str, held: str, asked: str, language: str
) -> None:
    """A gpu view serves hip for any dialect, so every kernel still resolves; only the target tells a setup
    it would run on the other device's forms."""
    view = view_with(tmp_path, "k", dialect=dialect, target=held)
    check = ["check", "--view", str(view), "--language", language, "--target", asked]
    assert cpf_cache.main([*check, "--kernels", "k"]) == 1
    expected = f"view {view} holds {held} forms, not the {asked} forms this setup runs on"
    assert capsys.readouterr().out.splitlines() == [expected]


def test_stage_refuses_a_view_of_the_other_target(tmp_path: pathlib.Path) -> None:
    """A gpu view would stage a .hip file into a C task folder."""
    view = view_with(tmp_path, "k", dialect="hip", target="gpu")
    dest = tmp_path / "tasks" / "k"
    stage = ["stage", "--view", str(view), "--kernel", "k", "--language", "c", "--target", "cpu"]
    assert cpf_cache.main([*stage, "--dest", str(dest)]) == 1
    assert not dest.exists()


def test_check_verified_refuses_a_form_the_judge_never_graded(tmp_path: pathlib.Path) -> None:
    """A rendered form is not a checked one: ``--verified`` names it until a grade is filed."""
    view = view_with(tmp_path, "gemm")
    assert cpf_cache.missing(view, ["gemm"], "c", "fp64", "cpu") == []
    (line,) = cpf_cache.missing(view, ["gemm"], "c", "fp64", "cpu", verified=True)
    assert "never graded" in line


def test_check_verified_refuses_an_unverified_verdict_and_passes_an_ok_one(tmp_path: pathlib.Path) -> None:
    view = view_with(tmp_path, "gemm")
    cpf_cache.record_verification(view, "gemm", "c", "fp64", {"verdict": "unverified", "reason": "segfault"})
    (line,) = cpf_cache.missing(view, ["gemm"], "c", "fp64", "cpu", verified=True)
    assert "unverified: segfault" in line
    cpf_cache.record_verification(view, "gemm", "c", "fp64", {"verdict": "ok"})
    assert cpf_cache.missing(view, ["gemm"], "c", "fp64", "cpu", verified=True) == []


def test_a_verdict_on_other_bytes_does_not_verify_the_served_form(tmp_path: pathlib.Path) -> None:
    """The verdict is tied to the form's cache key: re-pointing the view voids it."""
    view = view_with(tmp_path, "gemm")
    cpf_cache.record_verification(view, "gemm", "c", "fp64", {"verdict": "ok"})
    name = cpf_cache.pointer_name("gemm", "fp64", "c")
    record = json.loads((view / cpf_cache.VERIFIED_NAME / name).read_text())
    cpf_cache.write_json(view / cpf_cache.VERIFIED_NAME / name, {**record, "key": "0" * 64})
    (line,) = cpf_cache.missing(view, ["gemm"], "c", "fp64", "cpu", verified=True)
    assert "was graded as" in line
