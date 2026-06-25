# Local coding agents (zero-cost, sudoless)

Everything here runs **fully local** (no API key, no cloud, $0) on top of
[Ollama](https://ollama.com), and installs **without sudo**. The canonical model
is **`qwen2.5-coder:7b`** (chat/edit/agent) with **`qwen2.5-coder:1.5b`** for fast
autocomplete.

## 1. Set up Ollama + the models (sudoless)

```bash
scripts/install_ollama.sh            # detects ollama; if missing, installs to ~/.local (no sudo)
                                     # starts the server, pulls qwen2.5-coder:{7b,1.5b}
# pull extra models:
scripts/install_ollama.sh qwen2.5-coder:32b deepseek-coder-v2:16b
```

Works on **Linux, WSL, and macOS**. If Ollama is already on `PATH` it is reused;
otherwise it is installed under `~/.local` (Linux/WSL) or via Homebrew / an
unpacked app bundle (macOS) — never touching system dirs.

## 2. Three ways to use it

### a) optarena benchmark agent (the in-harness auto-tuner)

The benchmark loop models an agent as an auto-tuner: it hands the model a kernel
+ the exact C-ABI signature, gets back an implementation, compiles it, and scores
correctness + speedup. The Ollama backend needs **no Python package** (it speaks
HTTP over the stdlib):

```bash
python -m optarena.cli agent --agent ollama --kernels gemm --languages c
# OPTARENA_OLLAMA_MODEL / OPTARENA_OLLAMA_HOST override the model / server.
```

### b) Continue.dev — VS Code inline assistant

Best for autocomplete + chat while you code. Install the **Continue** extension,
then point `~/.continue/config.yaml` at the local Ollama models —
`qwen2.5-coder:7b` for chat/edit/apply and the tiny `qwen2.5-coder:1.5b` for
tab-autocomplete (snappy on a CPU-only machine).

### c) Aider — terminal coding agent (autonomous, multi-file)

Best for "do this task" workflows — closest to Claude Code:

```bash
pip install -r requirements/agent-aider.txt   # aider-chat
aider --model ollama/qwen2.5-coder:7b
```

Inside Aider:

```
> /add src/myfile.py        # scope the files
> fix the off-by-one in the loop
```

It edits files, shows a diff, and asks before applying.

### What to actually run

On a **no-GPU laptop, Continue.dev is the better fit**: its `1.5b` autocomplete
feels instant, while a terminal agent does multi-step loops that each cost a few
seconds — painful at CPU inference speeds (2–5 tok/s). Use Aider when you want
autonomous multi-file edits and can tolerate the latency (or have a GPU).

## 3. How a coding agent works (and running until completion)

A coding agent is a **loop around an LLM with tools**:

```
observe (repo state, errors)  ->  think (LLM)  ->  act (edit file / run shell / run tests)
        ^---------------------------------------------------------------|
        repeat until a STOP condition: tests pass, task done, or budget exhausted
```

The LLM never edits files itself — it *emits actions* (a diff, a shell command, a
tool call), the harness executes them, feeds the result back, and loops. This is
the **ReAct / tool-use loop**. "Runs independently until completion" just means
the stop condition is automatic (a passing test suite or a self-declared "done")
rather than a human pressing enter each turn.

**The most common way to get an autonomous local coder** is to run a mature
agent in non-interactive mode rather than hand-rolling the loop:

- **Aider, scripted** — the simplest turnkey option for a local repo:
  ```bash
  aider --model ollama/qwen2.5-coder:7b --yes-always --auto-test \
        --test-cmd "pytest -q" --message "implement X and make the tests pass"
  ```
  `--yes-always` removes the confirm prompts, `--auto-test` re-runs the tests
  after each edit and feeds failures back — i.e. it loops until green.
- **OpenHands / SWE-agent** — heavier, fully-autonomous agents (sandboxed shell +
  editor + browser) when you need more than file edits.
- **Roll-your-own** — point an OpenAI-style tool-use loop at Ollama's
  OpenAI-compatible endpoint (`http://localhost:11434/v1`) and loop on a
  `run_tests` tool until it returns 0. Only worth it for full control.

For the **optarena benchmark**, the harness *is* the loop: `agent --agent ollama`
generates → compiles → scores, and the correctness/speedup gate is the stop
condition. (A propose→compile→repair retry loop that feeds the compiler error
back to the model is the natural next step here.)

## 4. Sudoless containers with Apptainer

On shared / HPC machines without Docker or root, **Apptainer** runs unprivileged
and reuses the same image definitions Docker does — build a SIF directly from a
committed definition and run it:

```bash
apptainer build optarena-cpu.sif containers/cpu.def
apptainer exec optarena-cpu.sif python3 run_benchmark.py -b gemm -f numpy -p S -v True
```
