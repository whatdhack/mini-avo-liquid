"""GPU-less Triton compile check for generated kernels.

Compiles every entry @triton.jit kernel in a snippet ahead-of-time for the target GPU
(triton.compile + ASTSource), which catches nonexistent tl.* APIs, type/shape errors and bad
constexpr usage without needing a GPU.

The snippet is model output, so its top-level code is never run: only imports of trusted
modules, function definitions (non-jit decorators stripped) and literal constant assignments
are kept. Kernel argument types are not known without a real launch, so they are guessed from
parameter names and the problem's default dtype, together with the specializations a real
launch would add (16-byte aligned pointers/sizes, innermost strides == 1). The error message
includes the signature used.
"""
import ast
import importlib.util
import inspect
import os
import sys
import tempfile
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

TRUSTED_MODULES = {"torch", "triton", "math", "typing"}

# Target compute capability per hardware id in run_evolution.HARDWARE_CATALOG
HW_CAPABILITY = {"b200": 100, "h100": 90, "a100": 80, "l4": 89, "rtx4090": 89}

# Pointer element type per problem (store/load convert implicitly, so one type is enough)
PROBLEM_PTR_DTYPE = {"matmul_v2": "fp16", "vectorsum_v2": "fp32", "trimul_alphafold3": "fp32"}

FLOAT_SCALAR_NAMES = ("eps", "alpha", "beta", "scale", "momentum", "temperature")
POINTER_NAMES = {"x", "y", "z", "a", "b", "c", "w", "out", "output", "input", "dx", "dy", "dw", "db"}
DEFAULT_NUM_WARPS = 4
DEFAULT_NUM_STAGES = 3


@dataclass
class CompiledEntry:
    name: str
    compiled: Any  # triton.compiler.CompiledKernel
    signature: Dict[str, str]
    constexprs: Dict[str, Any]


def _is_jit(node: ast.AST) -> bool:
    return isinstance(node, ast.FunctionDef) and any("triton.jit" in ast.unparse(d) for d in node.decorator_list)


def _is_literal(node: ast.AST) -> bool:
    """A literal, or tl.constexpr(<literal>)."""
    try:
        ast.literal_eval(node)
        return True
    except ValueError:
        pass
    return (isinstance(node, ast.Call) and ast.unparse(node.func) in ("tl.constexpr", "triton.language.constexpr")
            and len(node.args) == 1 and not node.keywords and _is_literal(node.args[0]))


def _safe_module_source(code: str) -> str:
    """Rebuild the snippet keeping only definitions that are safe to import."""
    tree = ast.parse(code)
    lines = code.splitlines()
    out: List[str] = []
    for n in tree.body:
        if isinstance(n, ast.Import) and all(a.name.split(".")[0] in TRUSTED_MODULES for a in n.names):
            out.append(ast.unparse(n))
        elif isinstance(n, ast.ImportFrom) and (n.module or "").split(".")[0] in TRUSTED_MODULES and n.level == 0:
            out.append(ast.unparse(n))
        elif isinstance(n, ast.FunctionDef):
            # Default values are evaluated at def time, so only allow literal defaults
            if not all(_is_literal(d) for d in n.args.defaults + [d for d in n.args.kw_defaults if d]):
                continue
            # Keep the original def and body text (Triton compiles from source); drop other decorators
            decorator = "@triton.jit\n" if _is_jit(n) else ""
            out.append(decorator + "\n".join(lines[n.lineno - 1:n.end_lineno]))
        elif isinstance(n, (ast.Assign, ast.AnnAssign)) and n.value is not None and _is_literal(n.value):
            out.append(ast.unparse(n))
    return "import triton\nimport triton.language as tl\n\n" + "\n\n".join(out) + "\n"


def _first_autotune_config(code: str) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """(constexprs, launch options) from the first literal triton.Config({...}, num_warps=.., num_stages=..)."""
    for n in ast.walk(ast.parse(code)):
        if isinstance(n, ast.Call) and ast.unparse(n.func).endswith("Config") and n.args and isinstance(n.args[0], ast.Dict):
            try:
                constexprs = ast.literal_eval(n.args[0])
            except ValueError:
                continue
            options = {}
            for kw in n.keywords:
                if kw.arg in ("num_warps", "num_stages"):
                    try:
                        options[kw.arg] = int(ast.literal_eval(kw.value))
                    except ValueError:
                        pass
            return constexprs, options
    return {}, {}


def _guess_constexpr(name: str) -> Any:
    upper = name.upper()
    if "BLOCK" in upper or upper.startswith("TILE"):
        return 128
    if "GROUP" in upper:
        return 8
    if "ACTIVATION" in upper:
        return ""
    if "WARPS" in upper or "STAGES" in upper:
        return 4
    return 1


def _unit_strides(names: List[str]) -> set:
    """For each stride_<tensor><dim> group (e.g. stride_am, stride_ak) the last one is the contiguous dim."""
    groups: Dict[str, List[str]] = {}
    for n in names:
        if n.startswith("stride_") and len(n) > len("stride_") + 1:
            groups.setdefault(n[len("stride_"):-1], []).append(n)
    return {g[-1] for g in groups.values() if len(g) >= 2}


def _guess_signature(kernel, ptr_dtype: str, autotune: Dict[str, Any]):
    """Return (signature, constexprs, attrs) approximating what a real launch would specialize."""
    sig: Dict[str, str] = {}
    constexprs: Dict[str, Any] = {}
    attrs: Dict[Tuple[int], List[List[Any]]] = {}
    aligned = [["tt.divisibility", 16]]
    unit = _unit_strides([p.name for p in kernel.params])
    for i, p in enumerate(kernel.params):
        name = p.name
        if p.is_constexpr:
            sig[name] = "constexpr"
            has_default = p.default is not inspect.Parameter.empty
            constexprs[name] = autotune.get(name, p.default if has_default else _guess_constexpr(name))
        elif name in unit:
            # A real launch specializes integer arguments equal to 1 into constants
            sig[name] = "constexpr"
            constexprs[name] = 1
        # Lowercase short names (a, x, out) are pointers; uppercase ones (M, N, K, B, C) are sizes
        elif "ptr" in name.lower() or name in POINTER_NAMES:
            sig[name] = f"*{ptr_dtype}"
            attrs[(i,)] = aligned
        elif any(k in name.lower() for k in FLOAT_SCALAR_NAMES):
            sig[name] = "fp32"
        else:
            sig[name] = "i32"
            attrs[(i,)] = aligned  # problem sizes and strides are multiples of 16
    return sig, constexprs, attrs


def compile_entries(code: str, problem_id: str, hardware_id: str) -> Tuple[Optional[str], List[CompiledEntry]]:
    """Compile every entry kernel for the target GPU. Returns (error or None, compiled entries).

    Returns (None, []) if Triton is not installed, so callers skip the check.
    """
    try:
        import triton
        from triton.backends.compiler import GPUTarget
        from triton.compiler import ASTSource
    except ImportError:
        return None, []

    try:
        source = _safe_module_source(code)
    except SyntaxError as e:
        return f"syntax: line {e.lineno}: {e.msg}", []

    with tempfile.TemporaryDirectory() as tmp:
        mod_name = f"_triton_check_{uuid.uuid4().hex}"
        path = os.path.join(tmp, f"{mod_name}.py")
        with open(path, "w") as f:
            f.write(source)
        try:
            spec = importlib.util.spec_from_file_location(mod_name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[mod_name] = module
            spec.loader.exec_module(module)
        except Exception as e:
            sys.modules.pop(mod_name, None)
            return f"import: {type(e).__name__}: {e}", []

        try:
            kernels = {k: v for k, v in vars(module).items() if isinstance(v, triton.runtime.jit.JITFunction)}
            if not kernels:
                return "no @triton.jit kernel found", []
            # Helpers called from other kernels are compiled as part of their callers
            tree = ast.parse(source)
            called = {x.id for n in tree.body if _is_jit(n) for x in ast.walk(n)
                      if isinstance(x, ast.Name) and x.id in kernels and x.id != n.name}
            entries = [v for k, v in kernels.items() if k not in called]

            target = GPUTarget("cuda", HW_CAPABILITY.get(hardware_id, 90), 32)
            ptr_dtype = PROBLEM_PTR_DTYPE.get(problem_id, "fp32")
            autotune, launch = _first_autotune_config(code)
            options = {"num_warps": launch.get("num_warps", DEFAULT_NUM_WARPS),
                       "num_stages": launch.get("num_stages", DEFAULT_NUM_STAGES)}
            compiled: List[CompiledEntry] = []
            for kernel in entries:
                sig, constexprs, attrs = _guess_signature(kernel, ptr_dtype, autotune)
                try:
                    ck = triton.compile(ASTSource(kernel, sig, constexprs, attrs), target=target, options=options)
                except Exception as e:
                    msg = [line for line in str(e).strip().splitlines() if line.strip()]
                    # Triton's CompilationError is "at <line>:<col>:", source lines, a "^" marker, then the cause
                    caret = next((i for i, line in enumerate(msg) if line.strip() == "^"), None)
                    where = f"{msg[0]} {msg[caret - 1].strip()} ... " if caret and caret > 1 else ""
                    detail = where + msg[-1] if msg else type(e).__name__
                    return f"triton ({kernel.__name__}, sig={sig}): {detail[:300]}", []
                compiled.append(CompiledEntry(kernel.__name__, ck, sig, constexprs))
            return None, compiled
        finally:
            sys.modules.pop(mod_name, None)


def check_triton(code: str, problem_id: str, hardware_id: str) -> Optional[str]:
    """Return None if every entry kernel compiles for the target GPU, else a one-line error."""
    return compile_entries(code, problem_id, hardware_id)[0]
