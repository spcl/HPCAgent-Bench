# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Count an attempt's generated tokens with the model's OWN tokenizer, when the server counted none.

THE LAST RESORT, and it is only ever reached by an attempt with no `result` record -- a timeout or a
crash (T7-T9 of docs/DESIGN_data_collection_and_scoring.md). Those attempts are the expensive ones:
an agent killed at its wall spent its whole budget and, on these endpoints, left no output count at
all, because the per-turn assistant events report ``output_tokens: 0`` and the number arrives once,
on a record it never reached.

WHAT IS COUNTED: every assistant message's thinking text, answer text, and each tool call's input
serialized as the CLI sends it. That is what the model emitted; it is not what the server BILLED,
which also includes the role, channel and tool-call markers the transcript does not carry. Measured
against 50 transcripts that do have a result record, retokenized/server is a median

    gpt-oss-120b (vLLM)     0.961   [0.901-0.981]   n=20
    Kimi-K2.7-Code (SGLang) 0.977   [0.960-0.989]   n=10
    Qwen3.8-27B-FP8 (SGLang) 1.034  [0.973-4.730]   n=20

so this runs 2-4 percent LOW for the two that agree with their server, and the Qwen figure carries a
heavy tail (F9: that server's result record is itself short on some complete episodes, unexplained).
NO CORRECTION CONSTANT is applied -- a fudge factor fitted on 50 transcripts would make the number
look measured. A row counted this way says so in ``output_source``, so it can be excluded instead.

STDLIB-ONLY CALLERS: ``token_cost.py`` must stay importable inside the agent image, which has no
tokenizers package, so it never imports this module. The analysis passes :func:`output_counter` in.
"""

import functools
import glob
import json
import pathlib
from collections.abc import Callable, Iterable


#: Offline HuggingFace cache the tokenizers are read from. No download is ever attempted: a missing
#: tokenizer is a counter that returns None, not a network call on a compute node.
#:
#: ONE name for the weights, and it is HuggingFace's own ``HF_HOME``, whose hub cache is always
#: ``$HF_HOME/hub``. That is the library's contract rather than our convention, so a separate
#: "weights directory" variable would only be a second spelling of the same path -- free to drift
#: from the one the serving jobs actually load from.
#:
#: This was an absolute path naming one user's scratch. For anyone else the glob below simply came
#: up empty, and because an absent tokenizer is reported as "no counter" rather than an error, the
#: visible effect was token counts going None -- a misconfiguration wearing the costume of a model
#: this repo happens not to have cached.
def hf_hub_dir() -> pathlib.Path:
    """The hub cache directory, resolved in this order:

    1. ``HF_HOME`` -- what every sbatch and EDF in this repo exports.
    2. ``HPCAGENT_BENCH_CACHE`` -- the single cache root; ``HF_HOME`` defaults underneath it.
    3. ``huggingface_hub``'s own default, so a workstation with neither set still works.
    """
    import os

    if hf_home := os.environ.get("HF_HOME"):
        return pathlib.Path(hf_home) / "hub"
    if cache := os.environ.get("HPCAGENT_BENCH_CACHE"):
        return pathlib.Path(cache) / "hf" / "hub"
    try:
        from huggingface_hub.constants import HF_HUB_CACHE

        return pathlib.Path(HF_HUB_CACHE)
    except ImportError:
        return pathlib.Path.home() / ".cache" / "huggingface" / "hub"


HF_HUB = hf_hub_dir()

#: Arm model tag (``HPCAGENT_BENCH_RECORD_MODEL``) -> the repo id its ``VLLM_MODEL`` names. The env
#: is the truth when a run has it; this covers the runs whose launch env was not kept beside them.
MODEL_REPOS: dict[str, str] = {
    "qwen38": "Qwen/Qwen3.8-27B-FP8",
    "oss120b": "openai/gpt-oss-120b",
    "kimi27sglang": "moonshotai/Kimi-K2.7-Code",
    "glm53": "zai-org/GLM-5.3",
}

#: Kimi ships a tiktoken vocabulary and its own class rather than a ``tokenizer.json``; this is the
#: ``pat_str`` of ``tokenization_kimi.TikTokenTokenizer``, copied so the split matches the model's.
KIMI_PATTERN = "|".join(
    [
        r"[\p{Han}]+",
        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*"
        r"[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+"
        r"[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
        r"\p{N}{1,3}",
        r" ?[^\s\p{L}\p{N}]+[\r\n]*",
        r"\s*[\r\n]+",
        r"\s+(?!\S)",
        r"\s+",
    ]
)

#: ``message.model`` of the CLI's placeholder turn, which the model did not generate.
SYNTHETIC_MODEL = "<synthetic>"


def snapshot(repo: str) -> pathlib.Path | None:
    """The newest local snapshot directory of ``repo``, or None when the cache has no copy.

    A cache that does not exist AT ALL raises instead of returning None. The two are different
    faults wearing the same face: one model absent from a populated cache is expected and degrades
    to "no counter", whereas a hub directory that is not there means HF_HOME points somewhere
    wrong, and every model will be "missing". Silently reporting that as None is how a whole
    campaign's token counts come back empty with nothing in the log to say why.
    """
    if not HF_HUB.is_dir():
        raise FileNotFoundError(
            f"HuggingFace hub cache {HF_HUB} does not exist, so no tokenizer can be found and "
            f"every token count would silently be None. Set HF_HOME (hub is $HF_HOME/hub) or "
            f"HPCAGENT_BENCH_CACHE (hub is $HPCAGENT_BENCH_CACHE/hf/hub)."
        )
    found = sorted(glob.glob(str(HF_HUB / f"models--{repo.replace('/', '--')}" / "snapshots" / "*")))
    return pathlib.Path(found[-1]) if found else None


def repo_of(model: str) -> str:
    """``model`` as a repo id: an arm's short tag maps through :data:`MODEL_REPOS`, and anything
    already shaped like ``org/name`` (a ``VLLM_MODEL``) is taken as it stands."""
    return MODEL_REPOS.get(model, model)


@functools.lru_cache(maxsize=8, typed=True)
def counter(model: str) -> Callable[[str], int] | None:
    """A ``text -> token count`` for ``model``, or None when its tokenizer is not in the cache.

    The heavy imports live here: an analysis that never reaches the fallback never loads them, and
    the agent image never calls this at all.
    """
    root = snapshot(repo_of(model))
    if root is None:
        return None
    fast = root / "tokenizer.json"
    if fast.is_file():
        import tokenizers

        encoding = tokenizers.Tokenizer.from_file(str(fast))
        return lambda text: len(encoding.encode(text, add_special_tokens=False).ids)
    vocab = root / "tiktoken.model"
    if not vocab.is_file():
        return None
    import tiktoken
    from tiktoken.load import load_tiktoken_bpe

    model_encoding = tiktoken.Encoding(
        name=vocab.name,
        pat_str=KIMI_PATTERN,
        mergeable_ranks=load_tiktoken_bpe(str(vocab)),
        special_tokens={},
    )
    return lambda text: len(model_encoding.encode(text, disallowed_special=()))


def generated_text(events: Iterable[dict[str, object]]) -> list[str]:
    """Every piece of text the model produced across an attempt's assistant events, in order.

    One event carries one content block, so the blocks are taken as they come rather than deduped by
    message id -- that dedup belongs to USAGE, which each event repeats, and applying it here would
    keep one block per turn. The CLI's synthetic placeholder is not the model's work and is skipped.
    """
    out: list[str] = []
    for event in events:
        if event.get("type") != "assistant":
            continue
        message = event.get("message")
        if not isinstance(message, dict) or message.get("model") == SYNTHETIC_MODEL:
            continue
        blocks = message.get("content")
        if not isinstance(blocks, list):
            continue
        for block in blocks:
            if not isinstance(block, dict):
                continue
            kind = block.get("type")
            if kind == "thinking":
                out.append(str(block.get("thinking") or ""))
            elif kind == "text":
                out.append(str(block.get("text") or ""))
            elif kind == "tool_use":
                out.append(json.dumps(block.get("input")))
    return out


def output_counter(model: str) -> Callable[[list[dict[str, object]]], int | None]:
    """``token_cost``'s ``output_counter``: events -> generated tokens, None when it cannot count.

    None rather than 0 on a missing tokenizer, so an attempt the fallback could not reach keeps
    ``output_source: "none"`` instead of being recorded as having generated nothing.
    """
    count = counter(model)

    def count_events(events: list[dict[str, object]]) -> int | None:
        if count is None:
            return None
        return sum(count(chunk) for chunk in generated_text(events))

    return count_events


#: Where a campaign keeps the environment each job was launched with: ``<campaign>/.agent-launch/
#: <job id>/.env``, beside the run directories themselves. ``VLLM_MODEL`` in it is the repo id the
#: server actually served, which is the only unambiguous answer to "whose tokenizer".
LAUNCH_DIR = ".agent-launch"

#: The launch-env keys that name the model, best first: the repo id, then the arm's short tag.
MODEL_KEYS = ("VLLM_MODEL", "HPCAGENT_BENCH_RECORD_MODEL")


def launch_env(worker_dir: pathlib.Path) -> dict[str, str]:
    """The ``.env`` of the job this worker directory belongs to, or {} when it was not kept.

    ``<campaign>/<job>/agents/node-<n>/problem-<id>-worker-<w>`` puts the job id three levels up and
    the campaign four, which is where the launcher writes ``.agent-launch/<job>/.env``.
    """
    if len(worker_dir.parents) < 4:
        return {}
    run_dir = worker_dir.parents[2]
    path = run_dir.parent / LAUNCH_DIR / run_dir.name / ".env"
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}
    found: dict[str, str] = {}
    for line in text.splitlines():
        name, sep, value = line.partition("=")
        if sep and name.strip():
            found[name.strip()] = value.strip().strip('"').strip("'")
    return found


@functools.lru_cache(maxsize=64, typed=True)
def model_for_run(run_dir: pathlib.Path) -> str:
    """The model served for this run directory, empty when its launch env was not kept.

    Cached on the RUN directory: every worker of a job shares one server, and the .env is read once.
    """
    environment = launch_env(run_dir / "agents" / "node-0" / "problem-0-worker-0")
    for key in MODEL_KEYS:
        value = environment.get(key, "")
        if value:
            return value
    return ""


def counter_for(worker_dir: pathlib.Path) -> Callable[[list[dict[str, object]]], int | None] | None:
    """The retokenizing counter for whichever model served this worker's job, or None.

    None at every step that cannot be answered -- no launch env, an unknown model, no tokenizer in
    the cache -- because a missing fallback must leave ``output_source: "none"`` standing rather
    than invent a count.
    """
    if len(worker_dir.parents) < 3:
        return None
    model = model_for_run(worker_dir.parents[2])
    if not model or counter(model) is None:
        return None
    return output_counter(model)
