"""Real-GPU correctness check and timing for generated kernels.

The static score is a proxy: it flags bad kernels but cannot rank two well-tuned ones. This
module runs a candidate on the GPU instead — it builds the problem's inputs, calls the
kernel's host wrapper, compares the result against a torch reference, and times it with
`triton.testing.do_bench`.

Generated code is executed here, unlike triton_check.py, so every run happens in a separate
interpreter with a wall-clock timeout: a kernel that hangs the GPU, crashes the driver or
loops forever takes down only its own subprocess.

A problem needs a harness entry below to be benchmarked; `trimul_alphafold3` has none,
because its reference seed is a stub with no defined semantics.

A harness defines `make_inputs()`, `reference(*inputs)` and `TOL`, and may define
`check(got, want, inputs)` when elementwise comparison is the wrong gate, and `extra_cases()`
returning (label, inputs) pairs that are checked for correctness but not timed. Extra cases
exist because a kernel tuned against one input shape can be wrong on every other one: the
first GEMM winner here hardcoded row-major strides, and the first Cholesky winner missed the
accuracy gate at n=128 while passing at n=256.
"""
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import uuid
from typing import Any, Dict, Optional

DEFAULT_TIMEOUT_S = 300
DEFAULT_WARMUP_MS = 25
DEFAULT_REP_MS = 100
BENCH_MARKER = "__MINIAVO_BENCH__"

# Host wrapper the mutation prompt asks for, per problem. Discovery falls back to other
# top-level functions, but a kernel that follows this is always callable.
ENTRY_SIGNATURE = {
    "vectorsum_v2": "def solve(x: torch.Tensor) -> torch.Tensor  # x: (100_000_000,) fp32, returns the scalar sum as a 1-element tensor",
    "matmul_v2": "def solve(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor  # a, b: (4096, 4096) fp16, returns (4096, 4096)",
    "cholesky": ("def custom_kernel(data: torch.Tensor) -> torch.Tensor  # data: (64, 256, 256) fp32 symmetric "
                 "positive definite, returns the lower-triangular L with A = L @ L.T"),
}

# Inputs, reference result and comparison tolerance, per problem. Executed in the subprocess.
_HARNESS = {
    "vectorsum_v2": '''
def make_inputs():
    torch.manual_seed(0)
    return (torch.randn(100_000_000, device="cuda", dtype=torch.float32),)

def reference(x):
    return x.double().sum().reshape(1).float()

# 100M fp32 summed in a different order: compare on relative error against the fp64 sum
TOL = dict(rtol=2e-3, atol=5e-2)
''',
    "matmul_v2": '''
def make_inputs():
    torch.manual_seed(0)
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    b = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    return (a, b)

def reference(a, b):
    return a.float() @ b.float()

# fp16 inputs with fp32 accumulation over K=4096
TOL = dict(rtol=2e-2, atol=2e-1)

def extra_cases():
    torch.manual_seed(1)
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    bt = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    small_a = torch.randn(1024, 2048, device="cuda", dtype=torch.float16)
    small_b = torch.randn(2048, 512, device="cuda", dtype=torch.float16)
    return [("transposed b (non-contiguous)", (a, bt.t())),
            ("non-square 1024x2048 @ 2048x512", (small_a, small_b))]
''',
    "cholesky": '''
# One of the leaderboard's benchmark specs (batch 64, n 256, cond 2, seed 41256)
BATCH, N, DAMPING = 64, 256, 1.0e-2

def _spd(batch, n, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    a = torch.randn((batch, n, n), device="cuda", dtype=torch.float32, generator=gen)
    out = (a @ a.transpose(-1, -2)) / float(n)
    out.diagonal(dim1=-2, dim2=-1).add_(DAMPING)
    return (0.5 * (out + out.transpose(-1, -2))).contiguous()

def make_inputs():
    return (_spd(BATCH, N, 41256),)

def reference(A):
    return torch.linalg.cholesky_ex(A, check_errors=False).L

TOL = dict(rtol=1e-3, atol=1e-4)

def extra_cases():
    return [("batch=256 n=128", (_spd(256, 128, 41128),)),
            ("batch=16 n=512", (_spd(16, 512, 41512),))]

def check(got, want, inputs):
    """The upstream task checks properties, not elementwise agreement with one library: lower
    triangular, positive diagonal, and L @ L.T reconstructing A. A correct factorization that
    rounds differently from cuSOLVER is still correct."""
    A = inputs[0]
    if got.shape != A.shape:
        return float("inf"), "shape {} != input {}".format(tuple(got.shape), tuple(A.shape))
    if got.dtype != torch.float32:
        return float("inf"), "output dtype must be float32, got {}".format(got.dtype)
    if not torch.isfinite(got).all():
        return float("inf"), "output contains NaN or Inf"
    eps = torch.finfo(torch.float32).eps
    n = A.shape[-1]
    scale = torch.linalg.matrix_norm(A, ord=1, dim=(-2, -1)).clamp_min(torch.finfo(torch.float32).tiny)
    upper = torch.linalg.matrix_norm(torch.triu(got, diagonal=1), ord=1, dim=(-2, -1))
    if torch.any(upper > 8.0 * n * eps * scale):
        return (upper / scale).amax().item(), "output is not lower triangular"
    if torch.any(torch.diagonal(got, dim1=-2, dim2=-1) <= 0):
        return float("inf"), "output diagonal must be strictly positive"
    residual = torch.linalg.matrix_norm(got @ got.transpose(-1, -2) - A, ord=1, dim=(-2, -1))
    rel = (residual / scale).amax().item()
    allowed = 20.0 * n * eps
    return rel, None if rel <= allowed else "L @ L.T does not reconstruct A: relative residual {:.3g} > {:.3g}".format(rel, allowed)
''',
}

_RUNNER = '''
import json, sys, inspect, importlib.util, traceback
import torch, triton

{harness}

PREFERRED = ("solve", "custom_kernel", "run", "launch", "main")

def load_candidate(path):
    spec = importlib.util.spec_from_file_location("_miniavo_candidate", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_miniavo_candidate"] = module
    spec.loader.exec_module(module)
    return module

def entry_candidates(module):
    """Host wrappers worth trying, preferred names first, @triton.jit kernels excluded."""
    from triton.runtime.jit import JITFunction
    funcs = {{k: v for k, v in vars(module).items()
             if inspect.isfunction(v) and not isinstance(v, JITFunction) and not k.startswith("_")}}
    ordered = [funcs.pop(n) for n in PREFERRED if n in funcs]
    ordered += [v for k, v in funcs.items() if k.startswith(("solve", "run", "call", "launch"))]
    ordered += [v for k, v in funcs.items() if not k.startswith(("solve", "run", "call", "launch"))]
    return ordered

def invoke(fn, inputs, out_like):
    """Call a wrapper as fn(*inputs) or fn(*inputs, out); return its output tensor."""
    errors = []
    for with_out in (False, True):
        args = inputs + ((torch.zeros_like(out_like),) if with_out else ())
        try:
            result = fn(*args)
        except TypeError as e:
            errors.append(str(e))
            continue
        if isinstance(result, torch.Tensor):
            return result, args
        if with_out:
            return args[-1], args
        errors.append("returned {{}}, not a tensor".format(type(result).__name__))
    raise TypeError("; ".join(errors) or "not callable with the problem's inputs")

def mismatch(got, want):
    got = got.float().reshape(want.shape) if got.numel() == want.numel() else got.float()
    if got.shape != want.shape:
        return float("inf"), "shape {{}} != reference {{}}".format(tuple(got.shape), tuple(want.shape))
    if not torch.isfinite(got).all():
        return float("inf"), "output contains NaN or Inf"
    # Relative to the largest reference magnitude: a per-element ratio explodes wherever the
    # reference is near zero, which says nothing about whether the kernel is right
    err = (got - want).abs().max()
    rel = (err / want.abs().max().clamp_min(1e-12)).item()
    ok = torch.allclose(got, want, **TOL)
    return rel, None if ok else "max error {{:.3g}} ({{:.3g}} of peak) exceeds rtol={{}} atol={{}}".format(
        err.item(), rel, TOL["rtol"], TOL["atol"])

def main(path, warmup, rep):
    module = load_candidate(path)
    inputs = make_inputs()
    want = reference(*inputs)
    tried = []
    for fn in entry_candidates(module):
        try:
            got, args = invoke(fn, inputs, want)
        except Exception as e:
            tried.append("{{}}: {{}}".format(fn.__name__, str(e).splitlines()[0][:120]))
            continue
        torch.cuda.synchronize()
        rel, why = check(got, want, inputs) if "check" in globals() else mismatch(got, want)
        if why:
            return {{"ok": False, "entry": fn.__name__, "error": why, "rel_error": rel}}
        for label, extra in (extra_cases() if "extra_cases" in globals() else []):
            try:
                other, _ = invoke(fn, extra, reference(*extra))
            except Exception as e:
                return {{"ok": False, "entry": fn.__name__,
                         "error": "raised on {{}}: {{}}".format(label, str(e).splitlines()[0][:100])}}
            torch.cuda.synchronize()
            want2 = reference(*extra)
            _, why2 = check(other, want2, extra) if "check" in globals() else mismatch(other, want2)
            if why2:
                return {{"ok": False, "entry": fn.__name__, "rel_error": rel,
                         "error": "wrong on {{}}: {{}}".format(label, why2)}}
        ms = triton.testing.do_bench(lambda: fn(*args), warmup=warmup, rep=rep, return_mode="median")
        return {{"ok": True, "entry": fn.__name__, "latency_us": ms * 1000.0, "rel_error": rel}}
    return {{"ok": False, "error": "no host wrapper produced the reference result"
             + (" (tried " + "; ".join(tried) + ")" if tried else " (no non-kernel function found)")}}

if __name__ == "__main__":
    try:
        result = main(sys.argv[1], float(sys.argv[2]), float(sys.argv[3]))
    except Exception as e:
        result = {{"ok": False, "error": "{{}}: {{}}".format(type(e).__name__, str(e).splitlines()[0][:200]),
                   "traceback": traceback.format_exc()[-600:]}}
    print("{marker}" + json.dumps(result))
'''


def supported(problem_id: str) -> bool:
    return problem_id in _HARNESS


def bench_variant(code: str, problem_id: str, timeout_s: int = DEFAULT_TIMEOUT_S,
                  warmup_ms: float = DEFAULT_WARMUP_MS, rep_ms: float = DEFAULT_REP_MS) -> Dict[str, Any]:
    """Run `code` on the GPU. Returns {ok, latency_us?, entry?, error?, max_rel_error?}.

    Never raises: a crash, hang or wrong answer comes back as ok=False with a reason.
    """
    if not supported(problem_id):
        return {"ok": False, "error": f"no GPU harness for problem {problem_id}"}
    with tempfile.TemporaryDirectory() as tmp:
        candidate = os.path.join(tmp, f"candidate_{uuid.uuid4().hex}.py")
        runner = os.path.join(tmp, "runner.py")
        with open(candidate, "w") as f:
            f.write(code)
        with open(runner, "w") as f:
            f.write(_RUNNER.format(harness=textwrap.dedent(_HARNESS[problem_id]), marker=BENCH_MARKER))
        # Triton shells out to its bundled ptxas; keep the running interpreter's bin dir on PATH
        env = dict(os.environ)
        env["PATH"] = os.path.dirname(sys.executable) + os.pathsep + env.get("PATH", "")
        try:
            proc = subprocess.run([sys.executable, runner, candidate, str(warmup_ms), str(rep_ms)],
                                  cwd=tmp, env=env, capture_output=True, text=True, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": f"timed out after {timeout_s}s on the GPU"}
        for line in proc.stdout.splitlines():
            if line.startswith(BENCH_MARKER):
                return json.loads(line[len(BENCH_MARKER):])
        tail = (proc.stderr or proc.stdout).strip().splitlines()
        detail = tail[-1][:200] if tail else f"exit code {proc.returncode}"
        return {"ok": False, "error": f"benchmark subprocess failed: {detail}"}
