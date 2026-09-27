# MiniAVO-Liquid

**Evolutionary search for Triton kernels: compile-checked without a GPU, timed on one when there is one.**

![MiniAVO-Liquid loop: a parent kernel is mutated by an LLM into candidate kernels, which are filtered by a Python syntax check and an ahead-of-time Triton compile, scored statically, and every Nth generation run on a real GPU for correctness and time; the best becomes the next parent, and what the checks and the GPU run found is turned into instructions for the next generation's prompt](assets/demo.gif)

## What it does

MiniAVO-Liquid runs an evolutionary loop over GPU kernels for
[GPU MODE](https://www.gpumode.com)-style problems (`matmul_v2`, `vectorsum_v2`, `cholesky`,
`trimul_alphafold3`) on target GPUs from A100 to H100, B200 and the GB10:

1. **Seed.** Start from a naive kernel in `seeds/naive/`, or search the web for a better one
   with [Nimble](https://docs.nimbleway.com) (`--web-seed`).
2. **Mutate.** An LLM (`liquid/lfm-2.5-2.6b:free` on OpenRouter by default, or `glm-5.3` on
   Vultr Serverless Inference with `--provider vultr`) rewrites
   the current parent kernel, following an optimization strategy and optional steering prompt.
3. **Check.** Each candidate must parse as Python and compile as Triton for the target GPU.
4. **Score.** Candidates that compile get a static performance score. Every Nth generation,
   the candidate is also run on a real GPU (`--gpu-bench-every N`): checked against a torch
   reference, then timed.
5. **Select.** A candidate becomes the new parent only if it scores strictly higher than the
   current one — or, once there is a real timing, only if it was timed too and ran faster.
6. **Feed back.** What the checks, the static analysis and any real GPU run found becomes
   instructions in the next generation's prompt. These are kept for the current run, or across
   runs in a local JSONL file (`--feedback-memory file`) or RawTree (`--feedback-memory rawtree`).

The loop repeats for `--generations` rounds.

Everything is written to a lineage JSON (every candidate, including failed ones and why they
failed) and a submission script containing the best kernel.

It started as the standalone Python part of `mini-avo-gs`, with the mutation agent moved from
Gemini to Liquid AI.

### How kernels are judged

The search needs no GPU. Every candidate is compiled for the target architecture and judged by
what the compiler produces — that alone catches most of what a model gets wrong, and it works on
a laptop. It cannot tell you whether a kernel is *fast*, or whether it computes the right answer.
When there is a GPU on the machine, `--gpu-bench-every N` adds both, and the measured time then
outranks the score.

| Stage | What it catches | How |
|---|---|---|
| **Syntax** | Code that isn't valid Python (e.g. C-style `0.0f`) | `ast.parse` |
| **Triton compile** | Made-up `tl.*` functions, type and shape errors, bad `constexpr` use | `triton.compile` ahead of time for sm_80 / sm_89 / sm_90 / sm_100 / sm_121 (`triton_check.py`) |
| **Launch limits** | Kernels whose shared memory or registers can't fit on an SM | Per-SM limits of the target GPU |
| **Static score** (0–100) | How well the compiled kernel uses the hardware | `static_perf.py`, from the PTX, cubin and compile metadata |
| **GPU run** (optional) | Kernels that compile and score well but compute the wrong thing, hang or crash | `gpu_bench.py`, a separate interpreter with a timeout (`--gpu-bench-every N`) |

The static score combines:

- **Registers and spills** (`cuobjdump` resource usage). Spills score zero. Register *count*
  is scored in a band: credit ramps in from 32, is full from 96 to 224, and falls back to zero
  at the 255 cap, where the compiler is one allocation away from spilling rather than blocking
  well. Without this the occupancy term alone penalizes exactly the register-blocked kernels
  that win — the fastest GEMM measured on the GB10 uses 226 registers at 12% occupancy.
- **Theoretical occupancy**, from registers, shared memory and warps per block
- **Global load width**: 16-byte vector loads and async copies vs. scalar loads
- **Tensor cores**: `tcgen05` (Blackwell), `wgmma` (Hopper) or `mma.sync`
- **No global atomics** on GEMM-like problems: atomics there mean split-K accumulating into C,
  and every split-K variant measured on the GB10 landed in the slow tail
- **TMA** (`cp.async.bulk.tensor`) and **warp specialization** (`tl.range(..., warp_specialize=True)`)
  on Hopper and datacenter Blackwell (sm_90, sm_100) — **not** on sm_121, where measurement showed
  they buy nothing: the fastest 4096³ GEMM found on the GB10 uses neither and beats the best
  TMA + warp-specialized kernel by 7.6%, while the Hopper-style weights scored it 77.5 against
  that kernel's 100
- **Tile reuse** for GEMM-like problems

GEMM-like and memory-bound problems use different weights. TMEM use is reported but not
scored separately, since Triton allocates it automatically with `tcgen05`.

Even with those corrections the score is a weak ranker: across 14 measured GB10 GEMM kernels
spanning 1480–5135 µs, 8 still tie at 100. Treat it as a gate on what deserves a GPU run, not
as an ordering.

Because argument types aren't known without a real launch, the compile guesses them from
parameter names (`*_ptr` → pointer, `M`/`N`/`K` → `i32`) and adds the specializations a real
launch would (16-byte alignment, unit innermost strides).

Tuning constants are not guessed. They are read from the host wrapper's launch —
`kernel[grid](..., BLOCK_SIZE=4096, num_warps=8)`, following names bound to literals in the
wrapper's body or its parameter defaults — and from a `triton.Config` in an `@triton.autotune`
decorator when there is one. The launch site wins, then autotune, then the parameter default,
then a name-based guess. Without this the wrapper is never run, every variant compiles at
`BLOCK_SIZE=128` / 4 warps, and retuned kernels all score the same.

The score is a **proxy**. It reliably flags bad kernels: spills, launch failures, missing tensor
cores, scalar loads, low occupancy. It cannot rank two well-tuned kernels against each other, and
it cannot tell whether a kernel computes the right answer — a 100/100 GEMM candidate with TMA and
warp specialization turned out to be off by 2.4x the peak value when it was actually run. For both
of those, use `--gpu-bench-every N`. The model's own latency estimate is shown as "Claimed" and
never used for selection.

### Running candidates on a real GPU

`--gpu-bench-every N` runs generation N, 2N, … on the local GPU, and the seed once at the start:

- **Inputs and reference** come from a per-problem harness in `gpu_bench.py`. `vectorsum_v2` and
  `matmul_v2` and `cholesky` have one; `trimul_alphafold3` does not, because its seed is a stub
  with no defined semantics, and the flag is ignored for it.
- **Extra cases**: a harness may list further inputs that are checked but not timed, because a
  kernel tuned against one shape can be wrong on every other one. Both winners so far failed
  here — the GEMM one hardcoded row-major strides and returns garbage for a transposed operand,
  the Cholesky one misses the reconstruction gate at n=128 while passing at n=256.
- **Entry point**: the harness calls the wrapper the mutation prompt asks for — `solve(x)` for
  `vectorsum_v2`, `solve(a, b)` for `matmul_v2`, `custom_kernel(A)` for `cholesky`, matching the
  upstream GPU MODE task — and falls back to other top-level functions
  (`run`, `launch`, anything else that is not a `@triton.jit` kernel), trying `fn(*inputs)` and
  `fn(*inputs, out)`. The first one that reproduces the reference result is used.
- **Timing** is the median of `triton.testing.do_bench`, reported as "Measured" alongside the
  speedup over the seed's own measured latency (not the problem table's `baseline_latency_us`,
  which describes a different GPU).
- **Isolation**: generated code is executed, so each run happens in its own interpreter with a
  wall-clock timeout (`--gpu-bench-timeout`, default 300s). A kernel that hangs, crashes the
  driver or loops forever takes its subprocess down and nothing else.
- **Selection**: a candidate the GPU rejected is never promoted. Once the elite has a timing,
  only a timed and faster candidate replaces it, so an off-cycle generation cannot swap a
  measured elite for an unmeasured one.
- **Feedback**: a wrong result or a slower timing becomes an instruction in the next prompt,
  the same way compile errors do.

**Safety:** kernels are model output. The compile path never executes their top-level code —
only trusted imports, function definitions and literal constants are loaded before compiling.
The GPU path does execute them, which is the point of it, so every run is a separate interpreter
with a wall-clock timeout.

#### Remembering measurements across runs

A static score can be recomputed from the code at any time; a timing cannot — it only exists
because something was run. With `--feedback-memory file` (or `rawtree`), every measured latency
and every GPU rejection is persisted alongside the lessons and read back by the next run on the
same problem and hardware, so later prompts start with:

```
Real GPU measurements from this and earlier runs (median wall-clock, lower is better; these are
timings, not estimates):
- 1615.9 μs: v0_naive_baseline [baseline]
- 1657.9 μs: v1_two_stage_blockreduce [warp_shuffle]
- 1707.9 μs: v3_multitile_partial_buffer [tile_tuning]
- wrong or unrunnable on the GPU: v2_gridstride_twopass [vectorization] — output contains NaN or Inf
The kernel to beat ran in 1615.9 μs. The strategies listed above have already been measured at
that speed or slower, so a variation of them will not win: change the memory access pattern or
the amount of work per program.
```

The fastest run per kernel name is kept, this run's result supersedes a remembered one of the
same name, and a store never reads back its own run. The file defaults to
`.miniavo_memory.jsonl` next to the script (`FEEDBACK_FILE` overrides it) and is gitignored.

### What it has actually produced

Measured on an NVIDIA GB10 (DGX Spark, sm_121) with `glm-5.3` on Vultr Serverless Inference.
"Best found" is the fastest kernel the search produced that passed the GPU harness at the time.

| Problem | Seed | Best found | Library reference |
|---|---|---|---|
| `matmul_v2` (4096³ fp16) | 3337 µs / 40 TFLOPS | 1603 µs / 86 TFLOPS (gen 12 of 100) | 1530 µs / 90 TFLOPS (cuBLAS) |
| `cholesky` (batch 64, n 256) | 2229 µs | 825 µs (gen 1) | 671 µs (cuSOLVER) |
| `vectorsum_v2` (100M fp32) | 1616 µs / 247 GB/s | none faster in 10 generations | 1658 µs (`torch.sum`) |

Read those with the caveats:

- **`vectorsum_v2` has no headroom.** The naive seed already reads at 247 GB/s against a 244 GB/s
  measured stream ceiling. Every rewrite adds a launch or a pass and loses. A search that reports
  progress here is reporting noise.
- **Both winners initially passed only because the harness was too narrow.** The GEMM kernel
  hardcodes row-major strides and returns garbage for a transposed operand; the Cholesky kernel
  misses the reconstruction gate at n=128. The `extra_cases()` inputs now catch both, and both are
  rejected by the current harness.
- **The GEMM search converged at generation 12 and then stalled for 88 generations.** 20 of its 30
  timed candidates were the same five launch configurations under different names, and its final
  kernel is 7.6% slower than the same code at `128×256×64 / 8 warps / 3 stages` — two constants
  away, never tried.

### Feedback to the next generation

What the checks find is turned into instructions for the next prompt (`feedback.py`):

- **Rejected candidates** become rules. For example, a made-up `tl.*` name gets its Triton
  replacement (`tl.ceil_div` → `tl.cdiv`; no manual shared-memory allocation), C syntax gets
  "write Python, not C", a function defined inside a kernel gets "move helpers to module
  level", and a reply cut off at the token limit gets "return only the kernel".
- **Near misses** are reported: a candidate that compiled but didn't beat the parent is told
  which score components it lost on, or — when both were timed — the two latencies.
- **GPU rejections** become rules of their own: a kernel that passed every static check and then
  returned NaNs, missed the tolerance or timed out says so in the next prompt.
- **The parent's weak spots** from static analysis are listed as opportunities, such as narrow
  loads, no TMA, no warp specialization, spills or small tiles.

Repeated mistakes are merged with a "seen N×" count, and only the most recent few are kept.

**Feedback memory.** With `--feedback-memory local` (the default), lessons last for one run.
With `--feedback-memory file`, they are appended to `.miniavo_memory.jsonl` next to the script
and read back by the next run on the same problem and hardware — measured timings included.
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
- **Rate limits:** calls are throttled to OpenRouter's free-tier limits; Vultr Serverless
  Inference is paid per token and has no free-tier throttle.

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
| `OPENROUTER_API_KEY` | Mutations via OpenRouter (Liquid models). |
| `VULTR_SERVERLESS_INFERENCE_API_KEY` | Mutations via Vultr Serverless Inference. `VULTR_INFERENCE_API_KEY` and `VULTR_API_KEY` also work. |
| `NIMBLE_API_KEY` | `--web-seed` only |
| `RAWTREE_API_KEY` (+ optional `RAWTREE_DATABASE`, `RAWTREE_FEEDBACK_TABLE`) | `--feedback-memory rawtree` only; the key needs `read_write` permission |
| `LIQUID_MODEL`, `LIQUID_MAX_TOKENS`, `LIQUID_REASONING_EFFORT` | Optional OpenRouter overrides (defaults: `liquid/lfm-2.5-2.6b:free`, `8192`, `high`) |
| `VULTR_MODEL`, `VULTR_REASONING_EFFORT`, `LLM_MAX_TOKENS` | Optional Vultr overrides (defaults: `glm-5.3`, `low`, `32768`) |
| `LLM_PROVIDER` | Pins the provider (`openrouter` or `vultr`); same as `--provider` |

With no provider key, every generation uses the deterministic simulator. With both keys set,
OpenRouter is used unless `--provider vultr` or `LLM_PROVIDER=vultr` says otherwise. Only the
base URL, key and model default differ between the two: both are called through the OpenAI SDK,
and the OpenRouter-only `X-Title` header and `reasoning.effort` body field are dropped for Vultr.
`response_format={"type":"json_object"}` is attempted first and retried without it on a 400/422,
then retried again without the thinking-budget field, so providers that reject either still work.

The thinking budget is provider-shaped: OpenRouter takes `{"reasoning": {"effort": …}}` and
defaults to `high`; Vultr takes a flat `reasoning_effort` and defaults to `low`, because
`glm-5.3` at the default effort spends all 32k output tokens reasoning about the mutation
prompt and never emits the JSON object.

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

# Keep lessons and real GPU timings across runs in a local file
python run_evolution.py --problem vectorsum_v2 --hardware gb10 --gpu-bench-every 2 --feedback-memory file

# Steer the mutations, or pick another model
python run_evolution.py --problem matmul_v2 --steer "Use TMA descriptors and warp specialization"
python run_evolution.py --model liquid/lfm-2.5-1.2b-instruct:free

# Mutate with Vultr Serverless Inference instead of OpenRouter
python run_evolution.py --problem matmul_v2 --provider vultr
python run_evolution.py --provider vultr --model qwen3.8-27b

# What can each provider serve? (no API key needed)
python run_evolution.py --list-models vultr

# Time every 3rd generation on the local GB10 (DGX Spark, sm_121)
python run_evolution.py --problem vectorsum_v2 --hardware gb10 --generations 9 --gpu-bench-every 3

# Batched Cholesky, timed every other generation, remembering timings across runs
python run_evolution.py --problem cholesky --hardware gb10 --generations 20 \
    --gpu-bench-every 2 --feedback-memory file
```

`--gpu-bench-every` needs a CUDA GPU on the machine, and `--hardware` should name it so the
compile check targets the same architecture it will run on.

| Flag | Values | Default |
|---|---|---|
| `--problem` | `matmul_v2`, `vectorsum_v2`, `cholesky`, `trimul_alphafold3` | `matmul_v2` |
| `--hardware` | `b200`, `gb10`, `h100`, `a100`, `l4`, `rtx4090` | `b200` |
| `--generations` | number of generations | `5` |
| `--web-seed` / `--refresh-seed` | start from a Nimble-found kernel / ignore the cache | off |
| `--steer` | free-text guidance added to the mutation prompt | none |
| `--provider` | `openrouter`, `vultr` | whichever API key is set (OpenRouter first) |
| `--model` | any model ID the provider serves | provider default: `liquid/lfm-2.5-2.6b:free`, `glm-5.3` |
| `--list-models` | `openrouter`, `vultr` | prints the catalog and exits |
| `--gpu-bench-every` | run generation N, 2N, … on a real GPU (`0` disables); env `GPU_BENCH_EVERY` | `0` |
| `--gpu-bench-timeout` | seconds one GPU benchmark subprocess may take | `300` |
| `--feedback-memory` | `local` (this run only), `file` (JSONL next to the script) or `rawtree` | `local` |

### 4. Read the output

Each generation prints one of three outcomes:

- **`★ NEW ELITE`**: the candidate compiled and scored higher than the parent.
- **`✓ COMPILES, NOT BETTER`**: the candidate compiled but didn't beat the parent.
- **`✗ FAILED`**: the candidate failed a check. The Python or Triton error is shown, with the
  failing source line and the argument types that were guessed.

Files written to the current directory:

- `evolution_result_<problem>_<hw>.json`: the full lineage. Each entry has its code, status,
  `check_error`, `static` metrics and `static_score`, and the seed's source URL is included.
- `submission_<problem>_<variant>.py`: the best kernel. Without `--gpu-bench-every` it is **not
  benchmarked and not checked for correctness**, so run it on a real GPU before submitting it
  anywhere. With it, the lineage also carries `measured_latency_us`, `measured_speedup` and
  `bench_status` per candidate.

## Repository layout

| Path | Purpose |
|---|---|
| `run_evolution.py` | CLI and evolution loop, provider registry (OpenRouter, Vultr), problem and hardware catalog |
| `triton_check.py` | Safe loading plus ahead-of-time Triton compile for the target GPU |
| `gpu_bench.py` | Optional real-GPU correctness check and timing, in a subprocess with a timeout |
| `static_perf.py` | Static metrics (PTX / cubin / metadata) and the 0–100 score |
| `feedback.py` | Turns check, static-analysis and GPU-run findings into instructions for the next prompt |
| `local_store.py` | Feedback memory in a local JSONL file (`--feedback-memory file`) |
| `rawtree_store.py` | Optional RawTree backend that persists feedback memory across runs |
| `seed_search.py` | Nimble web search for seed kernels; slices out each `@triton.jit` kernel with its helpers |
| `seeds/naive/` | Built-in starter kernels, one `.py` per problem |
| `seeds/web/` | Cached web seeds (`--web-seed`) |
| `assets/make_demo_gif.py` | Regenerates `assets/demo.gif` |
