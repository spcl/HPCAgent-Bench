# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The serving rules and the benchmark overlays of ``hpcagent_bench/cluster/serve_args.py``: a flag edit keeps the rest of
the string as it was, the int4 checkpoint refuses expert parallelism without the Triton MoE runner, and every
``[bench.*]`` of experiments.toml applies to its model and passes the rules."""

import contextlib
import io
import pathlib
import tempfile

import pytest

from hpcagent_bench.cluster import env_spec, serve_args

ARGS = '"--trust-remote-code --attention-backend aiter --mem-fraction-static 0.52 --enable-metrics"'
INT4 = {"MOE_CHECKPOINT": "int4-wna16"}


def test_apply_replaces_in_place_appends_and_drops_with_the_value() -> None:
    changed = serve_args.apply(
        ARGS, {"--mem-fraction-static": "0.48", "--ep-size": "4", "--enable-hicache": ""}, ["--attention-backend"]
    )
    assert changed == '"--trust-remote-code --mem-fraction-static 0.48 --enable-metrics --ep-size 4 --enable-hicache"'
    assert serve_args.apply("--a 1 --b", {}, []) == "--a 1 --b", "an unquoted value stays unquoted"
    assert serve_args.parse(ARGS)["--trust-remote-code"] == ""


def test_apply_keeps_a_shell_reference_verbatim() -> None:
    value = '"--chat-template ${HPCAGENT_BENCH_REPO}/containers/inference/t.jinja --x 1"'
    assert "${HPCAGENT_BENCH_REPO}/containers/inference/t.jinja" in serve_args.apply(value, {"--x": "2"}, [])


@pytest.mark.parametrize("bad", ["positional --a", "--a 1 --a 2"])
def test_a_string_that_is_not_a_flag_list_is_refused(bad: str) -> None:
    with pytest.raises(ValueError, match="not a --flag|twice"):
        serve_args.parse(bad)


def test_an_argument_string_that_is_not_a_flag_list_is_refused_not_raised() -> None:
    assert "not a list of --flags" in serve_args.refusal({"SGLANG_EXTRA_ARGS": '"--a 1 --a 2"'})


@pytest.mark.parametrize(
    ("args", "env", "refused"),
    [
        ("--ep-size 4", INT4, True),
        ("--ep-size 4 --moe-runner-backend aiter", INT4, True),
        ("--ep-size 4 --moe-runner-backend triton", INT4, False),
        ("--ep-size 1", INT4, False),
        ("--attention-backend aiter", INT4, False),
        ("--ep-size 4", {}, False),
    ],
)
def test_expert_parallelism_on_the_int4_checkpoint_needs_the_triton_runner(
    args: str, env: dict[str, str], refused: bool
) -> None:
    message = serve_args.refusal({"SGLANG_EXTRA_ARGS": f'"{args}"'} | env)
    assert bool(message) is refused
    assert not refused or "--moe-runner-backend triton" in message


def test_every_bench_config_of_experiments_toml_applies_and_keeps_the_rules() -> None:
    for name, table in env_spec.TABLES["bench"].items():
        base = f"experiment:{table['model']}"
        changed = serve_args.bench_env(name, base)
        assert serve_args.refusal(env_spec.render(base) | changed) == "", name
    assert "0.48" in serve_args.bench_env("kimi-fraction-048", "experiment:kimi27sglang")["SGLANG_EXTRA_ARGS"]


def test_a_bench_config_for_another_model_or_none_is_refused() -> None:
    with pytest.raises(SystemExit, match="is for kimi27sglang"):
        serve_args.bench_env("kimi-base", "experiment:qwen38")
    with pytest.raises(SystemExit, match="no .bench.nope"):
        serve_args.bench_env("nope", "experiment:kimi27sglang")


def test_the_rule_refuses_a_staged_kimi_env_with_ep_and_not_the_one_the_toml_renders(tmp_path: pathlib.Path) -> None:
    env = tmp_path / ".env.s"
    rendered = env_spec.render("experiment:kimi27sglang")
    env.write_text(env_spec.as_text(rendered))
    assert serve_args.main(["check", str(env)]) == 0
    ep4 = serve_args.apply(rendered["SGLANG_EXTRA_ARGS"], {"--ep-size": "4"}, [])
    env.write_text(env_spec.as_text(rendered | {"SGLANG_EXTRA_ARGS": ep4}))
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        assert serve_args.main(["check", str(env)]) == 2
    assert "--moe-runner-backend triton" in err.getvalue()


def test_the_cli_prints_the_overlay_and_force_rules_skips_the_rules() -> None:
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        assert serve_args.main(["bench", "kimi-ep4-triton", "--base", "experiment:kimi27sglang"]) == 0
    printed = out.getvalue()
    assert printed.startswith("SGLANG_EXTRA_ARGS=")
    assert "--ep-size 4" in printed and "--moe-runner-backend triton" in printed


if __name__ == "__main__":
    for bad in ("positional --a", "--a 1 --a 2"):
        test_a_string_that_is_not_a_flag_list_is_refused(bad)
    test_apply_replaces_in_place_appends_and_drops_with_the_value()
    test_apply_keeps_a_shell_reference_verbatim()
    test_an_argument_string_that_is_not_a_flag_list_is_refused_not_raised()
    for case in (
        ("--ep-size 4", INT4, True),
        ("--ep-size 4 --moe-runner-backend aiter", INT4, True),
        ("--ep-size 4 --moe-runner-backend triton", INT4, False),
        ("--ep-size 1", INT4, False),
        ("--attention-backend aiter", INT4, False),
        ("--ep-size 4", {}, False),
    ):
        test_expert_parallelism_on_the_int4_checkpoint_needs_the_triton_runner(*case)
    test_every_bench_config_of_experiments_toml_applies_and_keeps_the_rules()
    test_a_bench_config_for_another_model_or_none_is_refused()
    test_the_rule_refuses_a_staged_kimi_env_with_ep_and_not_the_one_the_toml_renders(pathlib.Path(tempfile.mkdtemp()))
    test_the_cli_prints_the_overlay_and_force_rules_skips_the_rules()
