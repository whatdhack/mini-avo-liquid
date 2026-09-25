# MiniAVO-Liquid

**GPU-less evolutionary search for Triton kernels, driven by a small Liquid AI model.**

![MiniAVO-Liquid loop: a parent kernel is mutated by the Liquid model into candidate kernels, which are filtered by a Python syntax check and an ahead-of-time Triton compile, scored statically, and the best becomes the next parent; what the checks found is turned into instructions for the next generation's prompt](assets/demo.gif)

## What it does

MiniAVO-Liquid runs an evolutionary loop over GPU kernels for
[GPU MODE](https://www.gpumode.com)-style problems (`matmul_v2`, `vectorsum_v2`,
`trimul_alphafold3`) on target GPUs from A100 to H100 and B200:

1. **Seed.** Start from a naive kernel in `seeds/naive/`, or search the web for a better one
   with [Nimble](https://docs.nimbleway.com) (`--web-seed`).
2. **Mutate.** A Liquid AI model (`liquid/lfm-2.5-2.6b:free` on OpenRouter by default) rewrites
   the current parent kernel, following an optimization strategy and optional steering prompt.
3. **Check.** Each candidate must parse as Python and compile as Triton for the target GPU.
4. **Score.** Candidates that compile get a static performance score.
5. **Select.** A candidate becomes the new parent only if it scores strictly higher than the
   current one.
6. **Feed back.** What the checks and the static analysis found becomes instructions in the next
   generation's prompt. These lessons are kept for the current run, or across runs in RawTree
   (`--feedback-memory rawtree`).

The loop repeats for `--generations` rounds.

Everything is written to a lineage JSON (every candidate, including failed ones and why they
failed) and a submission script containing the best kernel.

It started as the standalone Python part of `mini-avo-gs`, with the mutation agent moved from
Gemini to Liquid AI.

### No real GPU: how kernels are scored

Nothing in this repo runs a kernel. There is no GPU in the loop, so there is **no measured
latency and no numerical correctness check**. Instead, each candidate is compiled for the
target GPU and judged by what the compiler produces:

| Stage | What it catches | How |
|---|---|---|
| **Syntax** | Code that isn't valid Python (e.g. C-style `0.0f`) | `ast.parse` |
| **Triton compile** | Made-up `tl.*` functions, type and shape errors, bad `constexpr` use | `triton.compile` ahead of time for sm_80 / sm_89 / sm_90 / sm_100 (`triton_check.py`) |
| **Launch limits** | Kernels whose shared memory or registers can't fit on an SM | Per-SM limits of the target GPU |
| **Static score** (0–100) | How well the compiled kernel uses the hardware | `static_perf.py`, from the PTX, cubin and compile metadata |

The static score combines:

- **Registers and spills** (`cuobjdump` resource usage)
- **Theoretical occupancy**, from registers, shared memory and warps per block
- **Global load width**: 16-byte vector loads and async copies vs. scalar loads
- **Tensor cores**: `tcgen05` (Blackwell), `wgmma` (Hopper) or `mma.sync`
- **TMA** (`cp.async.bulk.tensor`) and **warp specialization** (`tl.range(..., warp_specialize=True)`) on Hopper and Blackwell
- **Tile reuse** for GEMM-like problems

GEMM-like and memory-bound problems use different weights. TMEM use is reported but not
scored separately, since Triton allocates it automatically with `tcgen05`.

Because argument types aren't known without a real launch, the compile guesses them from
parameter names (`*_ptr` → pointer, `M`/`N`/`K` → `i32`) and adds the specializations a real
launch would (16-byte alignment, unit innermost strides).

The score is a **proxy**. It reliably flags bad kernels: spills, launch failures, missing tensor
cores, scalar loads, low occupancy. It cannot rank two well-tuned kernels against each other;
that needs real GPU timing. The model's own latency estimate is shown as "Claimed" and never used
for selection.

**Safety:** kernels are model output, so their top-level code is never executed. Only
trusted imports, function definitions and literal constants are loaded before compiling.

### Feedback to the next generation

What the checks find is turned into instructions for the next prompt (`feedback.py`):

- **Rejected candidates** become rules. For example, a made-up `tl.*` name gets its Triton
  replacement (`tl.ceil_div` → `tl.cdiv`; no manual shared-memory allocation), C syntax gets
  "write Python, not C", a function defined inside a kernel gets "move helpers to module
  level", and a reply cut off at the token limit gets "return only the kernel".
- **Near misses** are reported: a candidate that compiled but didn't beat the parent is told
  which score components it lost on.
- **The parent's weak spots** from static analysis are listed as opportunities, such as narrow
  loads, no TMA, no warp specialization, spills or small tiles.

Repeated mistakes are merged with a "seen N×" count, and only the most recent few are kept.

**Feedback memory.** With `--feedback-memory local` (the default), lessons last for one run.
With `--feedback-memory rawtree`, every lesson and candidate outcome is also written as an event
to a [RawTree](https://rawtree.com) table (Tinybird's analytics database for unstructured data; the
table is created on first insert). The next run on the same problem and hardware loads those
lessons back with SQL, so it starts out knowing the mistakes earlier runs made. The same table
can be queried for analysis across runs, for example:

```sql
SELECT status, count() FROM miniavo_feedback
WHERE kind = 'candidate' AND problem = 'matmul_v2' GROUP BY status
```

If RawTree is unreachable or the key is wrong, the run prints one warning and continues with
in-memory feedback.

### Generation fan-out (planned)

Today the CLI produces **one candidate per generation**. The planned design fans out each
generation:

```
population = [seed]
for gen in 1..g:
    children  = parallel(liquid(parent, strategy_i) for parent in population for i in 1..n)
    survivors = [c for c in children if syntax_ok(c) and triton_compiles(c)]   # cheap filters
    population = top_k(population + score(survivors))                         # parents stay in the pool
```

- **Parallel drafts:** the free model is called n times per parent, each with a different
  strategy for variety, and duplicates are dropped.
- **Cheap filters:** the syntax check and compile remove most bad drafts before scoring.
- **Elitism:** keeping the top k (parents included) bounds the population and stops it from
  dying out when every child fails.
- **Rate limits:** calls are throttled to OpenRouter's free-tier limits.

## How to run

### 1. Environment

Python 3.12 with the latest PyTorch and Triton (PyTorch 2.14 pins Triton 3.8):

```bash
conda create -n mini-avo-liquid python=3.12 -y
conda activate mini-avo-liquid
pip install -r requirements.txt
```

No GPU or CUDA driver is needed. Triton ships the `ptxas` and `cuobjdump` used for the compile
and static analysis.

If `/tmp` is a small in-memory filesystem (common on WSL), point pip at a disk-backed temp
directory for the large CUDA wheels: `TMPDIR=~/.cache/pip-tmp pip install -r requirements.txt`.

### 2. Keys

```bash
cp .env.example .env
```

| Variable | Needed for |
|---|---|
| `OPENROUTER_API_KEY` | Liquid model mutations. Without it, each generation uses a deterministic simulator. |
| `NIMBLE_API_KEY` | `--web-seed` only |
| `RAWTREE_API_KEY` (+ optional `RAWTREE_DATABASE`, `RAWTREE_FEEDBACK_TABLE`) | `--feedback-memory rawtree` only; the key needs `read_write` permission |
| `LIQUID_MODEL`, `LIQUID_MAX_TOKENS`, `LIQUID_REASONING_EFFORT` | Optional overrides (defaults: `liquid/lfm-2.5-2.6b:free`, `8192`, `high`) |

### 3. Run

```bash
# Memory-bound reduction on H100, 10 generations
python run_evolution.py --problem vectorsum_v2 --hardware h100 --generations 10

# GEMM on B200, starting from a kernel found on the web
python run_evolution.py --problem matmul_v2 --hardware b200 --web-seed --generations 10

# Search the web again instead of using the cached seed in seeds/web/
python run_evolution.py --problem matmul_v2 --web-seed --refresh-seed

# Keep check-derived lessons across runs in RawTree
python run_evolution.py --problem vectorsum_v2 --hardware h100 --feedback-memory rawtree

# Steer the mutations, or pick another model
python run_evolution.py --problem matmul_v2 --steer "Use TMA descriptors and warp specialization"
python run_evolution.py --model liquid/lfm-2.5-1.2b-instruct:free
```

| Flag | Values | Default |
|---|---|---|
| `--problem` | `matmul_v2`, `vectorsum_v2`, `trimul_alphafold3` | `matmul_v2` |
| `--hardware` | `b200`, `h100`, `a100`, `l4`, `rtx4090` | `b200` |
| `--generations` | number of generations | `5` |
| `--web-seed` / `--refresh-seed` | start from a Nimble-found kernel / ignore the cache | off |
| `--steer` | free-text guidance added to the mutation prompt | none |
| `--model` | any OpenRouter model ID | `liquid/lfm-2.5-2.6b:free` |
| `--feedback-memory` | `local` (this run only) or `rawtree` (persisted across runs) | `local` |

### 4. Read the output

Each generation prints one of three outcomes:

- **`★ NEW ELITE`**: the candidate compiled and scored higher than the parent.
- **`✓ COMPILES, NOT BETTER`**: the candidate compiled but didn't beat the parent.
- **`✗ FAILED`**: the candidate failed a check. The Python or Triton error is shown, with the
  failing source line and the argument types that were guessed.

Files written to the current directory:

- `evolution_result_<problem>_<hw>.json`: the full lineage. Each entry has its code, status,
  `check_error`, `static` metrics and `static_score`, and the seed's source URL is included.
- `submission_<problem>_<variant>.py`: the best kernel. It is **not benchmarked**, so time it on
  a real GPU before submitting it anywhere.

## Repository layout

| Path | Purpose |
|---|---|
| `run_evolution.py` | CLI and evolution loop, Liquid/OpenRouter client, problem and hardware catalog |
| `triton_check.py` | Safe loading plus ahead-of-time Triton compile for the target GPU |
| `static_perf.py` | Static metrics (PTX / cubin / metadata) and the 0–100 score |
| `feedback.py` | Turns check and static-analysis findings into instructions for the next prompt |
| `rawtree_store.py` | Optional RawTree backend that persists feedback memory across runs |
| `seed_search.py` | Nimble web search for seed kernels; slices out each `@triton.jit` kernel with its helpers |
| `seeds/naive/` | Built-in starter kernels, one `.py` per problem |
| `seeds/web/` | Cached web seeds (`--web-seed`) |
| `assets/make_demo_gif.py` | Regenerates `assets/demo.gif` |
