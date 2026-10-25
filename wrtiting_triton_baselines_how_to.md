To be removed before final PR

### Phase 1: SSH & Cluster Setup

#### 1. Configure SSH
Add the following configuration to your local `~/.ssh/config` (replace `<eth-username>` with your nethz username):

```ssh
Host student-cluster
    HostName student-cluster.inf.ethz.ch
    User <eth-username>
    ForwardAgent yes
    Compression yes
    ServerAliveInterval 30
    ServerAliveCountMax 4

Host *
    ControlMaster auto
    ControlPath ~/.ssh/cm-%C
    ControlPersist 10m
```

Copy your SSH key to the cluster so you do not need to type your password repeatedly:
```bash
ssh-copy-id student-cluster
```
*(Note: If you are connecting from outside ETH, connect to the ETH VPN first).*

#### 2. Connect & Check Allocation
Log in to the cluster and verify your remaining GPU hours:
```bash
ssh student-cluster
courses
```
Our course tag is `dphpc` (you should see it listed along with your remaining GPU hours).

#### 3. Recommended IDE Setup (VS Code Remote-SSH)
- In VS Code, install the **Remote - SSH** extension (`ms-vscode-remote.remote-ssh`).
- Press `F1` (or `Ctrl+Shift+P`), choose **Remote-SSH: Connect to Host...**, and select `student-cluster`.
- Open your checkout of `HPCAgent-Bench` on the cluster.
- *Reminder*: The VS Code server runs on the login node; **do not run GPU computation directly in the VS Code terminal**.

---

### Phase 2: Repository & Environment Setup

#### 1. Clone & Branch Setup
On the login node, clone the fork, checkout the project development branch (`dphpc-baselines-project-dev`), and create your own branch to work on:
```bash
git clone git@github.com:schreadr/HPCAgent-Bench.git
cd HPCAgent-Bench
git checkout dphpc-baselines-project-dev
git checkout -b <your-branch-name>
```
*(Note: `dphpc-baselines-project-dev` is already based on HPCAgent-Bench v0.2.1).*

#### 2. Install `uv`
`uv` is not pre-installed on the cluster. Install it to your user home directory:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc  # or restart your shell / add ~/.cargo/bin or ~/.local/bin to PATH
```

#### 3. Manage Environments: x86 (5060 Ti) vs. ARM (GB10)
Because your home directory is shared across nodes, **never use the same `.venv` for both x86 and ARM (GB10)**.
- **Login nodes & 5060 Ti nodes** are **x86_64**.
- **GB10 nodes** are **ARM64 (aarch64)**. You **must** build the ARM Python environment on a GB10 node.

##### Environment A: x86 (for RTX 5060 Ti)
You can sync this on the login node or inside a 5060 Ti node:
```bash
# Set a distinct environment path for x86
export UV_PROJECT_ENVIRONMENT=.venv-x86
uv sync --extra nvgpu
```

##### Environment B: ARM64 (for GB10)
Start an interactive job on a GB10 node to build the ARM environment:
```bash
srun --gpus gb10:1 --pty -A dphpc -t 60 bash --login
```
Inside the GB10 node:
```bash
export UV_PROJECT_ENVIRONMENT=.venv-arm64
uv sync --extra nvgpu
exit  # Exit immediately when done building to avoid wasting GPU hours!
```

---

### Phase 3: Selecting Kernels to Work On

Your supervisor emphasized targeting **sparsely populated dwarfs** and avoiding toy kernels. In HPCAgent-Bench (`hpcagent_bench/benchmarks/scientific_computing/`), the most sparsely populated dwarfs currently lacking Triton baselines are:

| Berkeley Dwarf | Existing Triton Baselines | Missing Triton Baselines |
| :--- | :--- | :--- |
| **`backtrack_branch_bound`** | 0 / 2 | `knapsack`, `nqueens` |
| **`graphical_models`** | 0 / 2 | `hmm_forward`, `viterbi` |
| **`combinational_logic`** | 1 / 2 | `bitonic_sort` |
| **`graph_traversal`** | 0 / 4 | `bfs`, `dfs`, `pagerank`, `sssp` |
| **`finite_state_machine`** | 0 / 7 | `dfa`, `kmp`, `aho_corasick`, `csv_tokenize`, etc. |
| **`dynamic_programming`** | 2 / 5 | `lcs`, `needleman_wunsch`, `smith_waterman` |
| **`unstructured_grids`** | 0 / 10 | `lulesh`, `velocity_tendencies`, `zekin_scatter`, etc. |

---

### Phase 4: Day-to-Day Development Workflow (Fast Feedback Loop)

Do **not** use Docker/Apptainer or the full benchmark suite for day-to-day coding. Use interactive Slurm sessions and a lightweight test script.

#### 1. Allocate an Interactive GPU Node
- **For RTX 5060 Ti**:
  ```bash
  srun --gpus 5060ti:1 --pty -A dphpc -t 60 bash --login
  ```
- **For GB10**:
  ```bash
  srun --gpus gb10:1 --pty -A dphpc -t 60 bash --login
  ```

#### 2. Implement Your Triton Kernel
Create `<kernel>_triton.py` in the same directory as `<kernel>_numpy.py`.

Key conventions ([docs/triton_authoring.md](docs/triton_authoring.md)):
- **Function Name**: Matches the sole top-level function of `<kernel>_numpy.py` (usually `kernel`).
- **Arguments**: Dense arrays are passed as GPU `torch.Tensor`. Scalars are passed as NumPy scalars—convert them (`int(a)`, `float(a)`) before passing to `@triton.jit`.
- **Autotune**: Order configs best-first. If accumulating into an output (e.g. `atomic_add`), use `restore_value=["<output_name>"]`.
- **Synchronization**: The harness automatically synchronizes with CUDA events after the call; **do not** call `torch.cuda.synchronize()` inside your entry function.

Example structure (`<kernel>_triton.py`):
```python
import torch
import triton
import triton.language as tl
from hpcagent_bench.support.helpers.triton_utilities import powers_of_2

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": b}, num_warps=w)
        for b in powers_of_2(8) for w in [2, 4, 8]
    ],
    key=["N"],
)
@triton.jit
def _kernel(x_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    val = tl.load(x_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, val * 2.0, mask=mask)

def kernel(x: torch.Tensor, out: torch.Tensor):
    N = x.numel()
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)
    _kernel[grid](x, out, N)
    return out
```

#### 3. Fast Correctness Check (No Container / No Full Harness)
Create a quick scratch test script (e.g., `.scratch/test_quick.py`) to verify your kernel directly against the reference:

```python
import torch
import numpy as np

# 1. Import initialize and numpy reference
from hpcagent_bench.benchmarks.scientific_computing.dense_linear_algebra.atax.atax import initialize
from hpcagent_bench.benchmarks.scientific_computing.dense_linear_algebra.atax.atax_numpy import kernel as numpy_ref
from hpcagent_bench.benchmarks.scientific_computing.dense_linear_algebra.atax.atax_triton import kernel as triton_kernel

# 2. Initialize test data (preset S or custom sizes)
x_np, A_np, out_np = initialize(128, 128, datatype=np.float32)

# 3. Compute reference output
ref_out = numpy_ref(x_np, A_np, out_np.copy())

# 4. Transfer to GPU for Triton
A_torch = torch.from_numpy(A_np).cuda()
x_torch = torch.from_numpy(x_np).cuda()
out_torch = torch.zeros_like(torch.from_numpy(out_np)).cuda()

# 5. Run Triton kernel
triton_out = triton_kernel(A_torch, x_torch)  # or in-place depending on kernel signature
torch.cuda.synchronize()

# 6. Verify correctness
np.testing.assert_allclose(ref_out, triton_out.cpu().numpy(), rtol=1e-4, atol=1e-4)
print(" Correctness test passed!")
```
Run it inside your interactive node:
```bash
# On 5060 Ti (x86):
export UV_PROJECT_ENVIRONMENT=.venv-x86
uv run python .scratch/test_quick.py

# On GB10 (ARM):
export UV_PROJECT_ENVIRONMENT=.venv-arm64
uv run python .scratch/test_quick.py
```

#### 4. Validate with the HPCAgent-Bench Harness
Once the quick test passes, run the official benchmark harness.

To validate against the compiled reference (which uses OpenMP), the harness enforces specific shell environment limits before starting:
- The stack limit must be raised to its hard limit: `ulimit -s "$(ulimit -H -s)"`
- OpenMP thread and stack settings: `export OMP_STACKSIZE=512M` and `export OMP_THREAD_LIMIT=$(nproc)`
- Disable namespace seal on the student cluster: `export HPCAGENT_BENCH_GRADING_SEAL=false`

Run this once in your interactive GPU session before benchmarking:
```bash
ulimit -s "$(ulimit -H -s)"
export OMP_STACKSIZE=512M
export OMP_THREAD_LIMIT=$(nproc)
export HPCAGENT_BENCH_GRADING_SEAL=false
```

Then run the validation:
```bash
# Validate against reference at preset S (3 repetitions)
uv run hpcagent-bench run-framework -f triton -b <kernel> -p S -d float32 -r 3
```
Look for `validation: SUCCESS` in the output.

#### 5. Compare with Other Baselines & View Results
You can run the same benchmark under the NumPy baseline to compare performance:
```bash
# Compare against the NumPy reference implementation (e.g. for pagerank)
uv run hpcagent-bench run-framework -f numpy -b pagerank -p S -d float32 -r 3
```


**TIP**: Preset **`S`** is very small and intended primarily as a quick sanity check. Use the **`-p M`** parameter (medium dataset) or the default fuzzed preset to observe representative speedups on GPU:
 ```bash
 # Run Triton on preset M
 uv run hpcagent-bench run-framework -f triton -b <kernel> -p M -d float32 -r 3

 # Run default fuzzed timing preset (official benchmark evaluation)
 uv run hpcagent-bench run-framework -f triton -b <kernel> -d float32
 ```

Whenever you run benchmarks, rows are recorded into the SQLite results database.

- **Check run status & validation counts (pass/fail gate)**:
  ```bash
  uv run hpcagent-bench run-framework --summarize results/hpcagent_bench0.db
  ```
- **Inspect the exact timing numbers (`median_ms`)**:
  ```bash
  uv run python -c "
  import sqlite3
  conn = sqlite3.connect('results/hpcagent_bench0.db')
  print(f'{\"framework\":<12} {\"kernel\":<12} {\"preset\":<8} {\"datatype\":<10} {\"median_ms\":>12} {\"validated\":>10}')
  print('-' * 68)
  for col, kern, pre, dt, med, val in conn.execute('SELECT column, kernel, preset, datatype, median_ms, validated FROM canon'):
      med_str = f'{med:.3f} ms' if med is not None else 'N/A'
      print(f'{col:<12} {kern:<12} {pre:<8} {dt:<10} {med_str:>12} {val:>10}')
  "
  ```

---

### Phase 5: Container Runs (For Final Benchmarks Only)

Before reporting official numbers or submitting PRs, run inside the cluster containers to ensure complete reproducibility.

Container image mapping:
- **RTX 5060 Ti (x86)**: `docker://spcleth/hpcagent-bench:judge-nvidia-x86_64-latest`
- **GB10 (ARM64)**: `docker://spcleth/hpcagent-bench:judge-nvidia-latest`


**WARNING**: By default, Apptainer unpacks temporary build layers into `/tmp`, which quickly runs out of space on compute nodes (`write /tmp/build-temp-...: no space left on device`).
Always redirect `APPTAINER_TMPDIR` and `APPTAINER_CACHEDIR` away from `/tmp`, and pre-build the `.sif` image once.

#### Step 1: Pre-build the `.sif` Container (Do this once)
Run this once on a node with sufficient disk space:
```bash
# Redirect Apptainer temporary and cache directories away from /tmp
export APPTAINER_TMPDIR="${SCRATCH:-$HOME}/.apptainer_tmp"
export APPTAINER_CACHEDIR="${SCRATCH:-$HOME}/.apptainer_cache"
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"

# On 5060 Ti (x86):
apptainer build ~/judge-nvidia-x86_64.sif docker://spcleth/hpcagent-bench:judge-nvidia-x86_64-latest

# On GB10 (ARM64 - run this on a GB10 node!):
apptainer build ~/judge-nvidia-arm64.sif docker://spcleth/hpcagent-bench:judge-nvidia-latest
```

#### Step 2: Run Benchmarks with the Pre-built `.sif`
Running from a `.sif` file mounts the image directly without unpacking into `/tmp`:

##### Option A: Interactive Run (via `srun` & Apptainer)
Inside an allocated GPU session:
```bash
# On 5060 Ti:
apptainer exec --nv ~/judge-nvidia-x86_64.sif \
    hpcagent-bench run-framework -f triton -b <kernel> -d float32

# On GB10:
apptainer exec --nv ~/judge-nvidia-arm64.sif \
    hpcagent-bench run-framework -f triton -b <kernel> -d float32
```

##### Option B: Automated Batch Run (via `sbatch` - Recommended for Sweeps)
Submitting an `sbatch` job is recommended for final benchmarks because you don't need to keep an active terminal open, avoiding accidental idle GPU hours:

Create a submission script, e.g. `run_final_bench.sbatch`:
```bash
#!/bin/bash
#SBATCH --job-name=final_bench
#SBATCH -A dphpc
#SBATCH --gpus=5060ti:1        # or gb10:1 on GB10
#SBATCH --time=00:30:00
#SBATCH --output=bench_%j.out
#SBATCH --error=bench_%j.err

# Execute inside the pre-built container
apptainer exec --nv ~/judge-nvidia-x86_64.sif \
    hpcagent-bench run-framework -f triton -b <kernel> -d float32
```
Submit with:
```bash
sbatch run_final_bench.sbatch
```
*(Note on node exclusivity: Slurm isolates your assigned GPU with cgroups so no one else can access it. Do not pass `--exclusive`—requesting an entire multi-GPU node violates the one-GPU-per-student limit and burns through your quota).*

---

### Phase 6: Results Analysis, Statistics & Plotting

HPCAgent-Bench provides built-in statistical testing and plotting utilities following the **Hoefler & Belli (SC'15)** benchmarking standard:

#### 1. Inspect Exact Timing Numbers
To inspect measured runtimes across all tested frameworks and presets:
```bash
uv run python -c "
import sqlite3
conn = sqlite3.connect('results/hpcagent_bench0.db')
print(f'{\"framework\":<14} {\"kernel\":<14} {\"preset\":<8} {\"datatype\":<10} {\"median_ms\":>12} {\"validated\":>10}')
print('-' * 70)
for col, kern, pre, dt, med, val in conn.execute('SELECT column, kernel, preset, datatype, median_ms, validated FROM canon'):
    med_str = f'{med:.3f} ms' if med is not None else 'N/A'
    print(f'{col:<14} {kern:<14} {pre:<8} {dt:<10} {med_str:>12} {val:>10}')
"
```

#### 2. Generate Publication-Quality Speedup Plots
Draws $\log_2$ speedup ratio bar/dot charts comparing frameworks against a baseline (e.g. Triton vs. NumPy):
```bash
uv run hpcagent-bench plot kernel-speedups \
    --canon-db results/hpcagent_bench0.db \
    --canon-columns triton,numba \
    --baseline numpy \
    --out results/plots/kernel_speedups.pdf
```
*(Produces both vector `.pdf` and raster `.png` under `results/plots/`).*

#### 3. Statistical Analysis & Confidence Intervals
HPCAgent-Bench includes tools for computing non-parametric statistics (Mann-Whitney rank tests, Fisher's exact tests, and geometric mean speedup ratios with 95% BCa bootstrap confidence intervals):
```python
# In Python or Jupyter
from hpcagent_bench.stats import summary

# Example: calculate geometric mean speedup with 95% BCa bootstrap CI
speedups = [1.85, 2.10, 1.45, 3.20]
geomean, (ci_low, ci_high) = summary.geomean_ci(speedups)
print(f"Geomean Speedup: {geomean:.2f}x [95% CI: {ci_low:.2f}x - {ci_high:.2f}x]")
```

#### 4. Export Observations to CSV
Extract raw benchmark records to flat CSV files for custom analysis in Pandas, R, or Excel:
```bash
uv run hpcagent-bench extract --db results/hpcagent_bench.db --out results/observations.csv
```

---

### Phase 7: Cluster Etiquette & Rules Checklist

- [ ] **Never run GPU jobs on login nodes**: Login nodes are strictly for editing, git, and building x86 dependencies.
- [ ] **Always set `-t <minutes>`**: Always specify a reasonable wall-time limit (e.g. `-t 30` or `-t 60`) on `srun` to prevent accidental billing if you forget about the terminal.
- [ ] **Exit interactive sessions immediately**: Slurm deducts from your 200 GPU hours for the **entire duration** of the allocation, even when completely idle! Type `exit` or `Ctrl+D` whenever you step away.
- [ ] **One GPU at a time**: You are allowed a maximum of **1 running job with 1 GPU** simultaneously.
- [ ] **Use the right node**:
  - `5060ti:1` for x86 / Blackwell CC 12.0
  - `gb10:1` for ARM64 CC 12.1 (build your ARM Python env here!)
  - Skip 1080 Ti; only use 2080 Ti for quick sanity checks if necessary.