"""Find a good initial Triton seed kernel on the web via Nimble Search (https://docs.nimbleway.com)."""
import ast
import json
import os
import re
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

NIMBLE_SEARCH_URL = "https://sdk.nimbleway.com/v2/search"
SEED_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "seeds", "web")

# Per-problem search query and code features a relevant kernel should contain
SEED_QUERIES: Dict[str, Dict[str, Any]] = {
    "matmul_v2": {
        "query": "triton matmul kernel tl.dot autotune GROUP_SIZE_M fp16 example",
        "must_have": ["tl.dot"],
        "must_not": [],
        "bonus": ["triton.autotune", "GROUP_SIZE_M", "num_stages", "tl.float32"],
        "min_score": 2.0,
    },
    "vectorsum_v2": {
        "query": "triton reduction kernel tl.sum large vector sum example",
        "must_have": ["tl.sum"],
        # reject matmuls, norms and softmax-style row reductions that merely contain tl.sum
        "must_not": ["tl.dot", "rstd", "tl.exp", "tl.max(", "mean"],
        "bonus": ["triton.autotune", "tl.atomic_add", "mask"],
        "min_score": 1.5,
    },
    "trimul_alphafold3": {
        "query": "triton triangle multiplicative update kernel alphafold",
        "must_have": ["tl.dot"],
        "must_not": [],
        "bonus": ["triton.autotune", "tl.sigmoid", "layer_norm"],
        "min_score": 1.5,
    },
}

TRITON_HEADER = "import torch\nimport triton\nimport triton.language as tl\n\n"
FENCE_RE = re.compile(r"```[a-zA-Z0-9_+-]*\n(.*?)```", re.DOTALL)
GITHUB_BLOB_RE = re.compile(r"https://github\.com/([^/]+)/([^/]+)/blob/(.+\.py)$")
# triton-lang.org tutorial pages are rendered from python/tutorials/<name>.py in the Triton repo
TRITON_TUTORIAL_RE = re.compile(r"https://triton-lang\.org/[^/]+/getting-started/tutorials/([\w-]+)\.html$")


def nimble_search(query: str, api_key: str, max_results: int = 8) -> List[Dict[str, Any]]:
    body = json.dumps({
        "query": query,
        "max_results": max_results,
        "full_content": True,
        "output_format": "markdown",
        "include_domains": ["github.com", "triton-lang.org", "pytorch.org", "gist.github.com"],
    }).encode()
    req = urllib.request.Request(
        NIMBLE_SEARCH_URL, data=body, method="POST",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.load(resp).get("results", [])
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Nimble search HTTP {e.code}: {e.read().decode(errors='replace')[:300]}") from e


def raw_source_url(url: str) -> Optional[str]:
    url = url.split("#")[0].split("?")[0]
    m = GITHUB_BLOB_RE.match(url)
    if m:
        return f"https://raw.githubusercontent.com/{m.group(1)}/{m.group(2)}/{m.group(3)}"
    m = TRITON_TUTORIAL_RE.match(url)
    if m:
        return f"https://raw.githubusercontent.com/triton-lang/triton/main/python/tutorials/{m.group(1)}.py"
    return None


def fetch_github_raw(url: str) -> Optional[str]:
    """Scraped pages often lose code formatting; fetch the raw .py source instead."""
    raw = raw_source_url(url)
    if not raw:
        return None
    try:
        with urllib.request.urlopen(raw, timeout=30) as resp:
            return resp.read().decode(errors="replace")
    except Exception:
        return None


def _is_triton_kernel(node: ast.AST) -> bool:
    return isinstance(node, ast.FunctionDef) and any("triton.jit" in ast.unparse(d) for d in node.decorator_list)


def _defined_names(node: ast.AST) -> List[str]:
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
        return [node.name]
    if isinstance(node, (ast.Assign, ast.AnnAssign)):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        return [n.id for t in targets for n in ast.walk(t) if isinstance(n, ast.Name)]
    return []


def slice_kernels(code: str) -> List[str]:
    """Split a module into one self-contained snippet per @triton.jit kernel.

    Each snippet keeps the imports, the kernel, the top-level helpers/configs it references
    (transitively), and host wrappers that launch it; benchmarks and tests are dropped.
    """
    tree = ast.parse(code)
    lines = code.splitlines()
    top = [n for n in tree.body if not isinstance(n, (ast.Expr, ast.If))]  # drop docstrings and __main__
    defs = {name: n for n in top for name in _defined_names(n)}

    def source(n: ast.AST) -> str:
        start = min([n.lineno] + [d.lineno for d in getattr(n, "decorator_list", [])])
        return "\n".join(lines[start - 1:n.end_lineno])

    def used_names(n: ast.AST) -> set:
        """Global names read by n (function parameters and locals shadow module-level names)."""
        loads = {x.id for x in ast.walk(n) if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Load)}
        if isinstance(n, ast.FunctionDef):
            args = n.args
            params = {a.arg for a in args.posonlyargs + args.args + args.kwonlyargs}
            params |= {a.arg for a in (args.vararg, args.kwarg) if a}
            stores = {x.id for x in ast.walk(n) if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Store)}
            loads -= params | stores
        return loads

    imports = [source(n) for n in top if isinstance(n, (ast.Import, ast.ImportFrom))]
    kernels = [n for n in top if _is_triton_kernel(n)]
    snippets = []
    for kernel in kernels:
        keep = {id(kernel): kernel}
        # host wrappers that launch this kernel (but not other kernels' wrappers)
        for n in top:
            if isinstance(n, ast.FunctionDef) and not _is_triton_kernel(n) and kernel.name in used_names(n):
                keep[id(n)] = n
        work = list(keep.values())
        while work:
            for name in used_names(work.pop()):
                dep = defs.get(name)
                if dep is not None and id(dep) not in keep:
                    keep[id(dep)] = dep
                    work.append(dep)
        body = [source(n) for n in top if id(n) in keep]
        snippets.append("\n".join(imports) + "\n\n\n" + "\n\n\n".join(body) + "\n")
    return snippets


def extract_triton_snippets(text: str) -> List[str]:
    """Return parseable Python snippets, one per @triton.jit kernel."""
    blocks = FENCE_RE.findall(text) or [text]
    snippets = []
    for block in blocks:
        if "@triton.jit" not in block:
            continue
        code = block if "import triton" in block else TRITON_HEADER + block
        try:
            snippets.extend(slice_kernels(code))
        except (SyntaxError, ValueError):
            continue
    return snippets


def score_snippet(code: str, spec: Dict[str, Any]) -> float:
    if not all(k in code for k in spec["must_have"]) or any(k in code for k in spec["must_not"]):
        return -1.0
    n_lines = code.count("\n")
    if n_lines > 400:  # whole libraries make poor seeds for a small mutation model
        return -1.0
    return sum(1.0 for k in spec["bonus"] if k in code) + min(n_lines, 120) / 120.0


def find_web_seed(problem_id: str, refresh: bool = False, verbose: bool = True) -> Optional[Dict[str, Any]]:
    """Return {"code", "source_url", "query", "score"} for the best web seed, or None.

    Results are cached in seeds/web/<problem_id>.json so repeated runs don't spend search credits.
    """
    log = print if verbose else (lambda *a, **k: None)
    cache_path = os.path.join(SEED_CACHE_DIR, f"{problem_id}.json")
    if not refresh and os.path.isfile(cache_path):
        with open(cache_path) as f:
            seed = json.load(f)
        log(f"[*] Web seed: using cached {cache_path} (from {seed['source_url']})")
        return seed

    spec = SEED_QUERIES.get(problem_id)
    api_key = os.getenv("NIMBLE_API_KEY")
    if not spec or not api_key:
        log(f"[!] Web seed skipped: {'NIMBLE_API_KEY not set' if spec else f'no seed query for {problem_id}'}")
        return None

    log(f"[*] Web seed: Nimble search \"{spec['query']}\"")
    try:
        results = nimble_search(spec["query"], api_key)
    except Exception as e:
        log(f"[!] Web seed search failed: {e}")
        return None

    best: Optional[Dict[str, Any]] = None
    for r in results:
        url = r.get("url", "")
        texts = [r.get("content") or ""]
        raw = fetch_github_raw(url)
        if raw:
            texts.insert(0, f"```python\n{raw}\n```")
        for text in texts:
            for code in extract_triton_snippets(text):
                score = score_snippet(code, spec)
                if score >= spec["min_score"] and (best is None or score > best["score"]):
                    best = {"code": code, "source_url": url, "query": spec["query"], "score": round(score, 2)}
    log(f"    └─ {len(results)} results; " + (f"best seed from {best['source_url']} (score {best['score']})" if best else "no kernel passed the relevance checks; keeping the naive seed"))

    if best:
        os.makedirs(SEED_CACHE_DIR, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(best, f, indent=2)
    return best
