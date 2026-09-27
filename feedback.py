"""Turn check and static-analysis findings into instructions for the next generation's prompt.

Three sources feed the prompt:
  * failures: syntax / Triton compile / launch-limit errors and truncated agent replies, mapped
    to concrete rules (e.g. a nonexistent tl.* API -> what to use instead)
  * near misses: candidates that compiled but did not beat the parent, with the score parts
    where they fell short
  * the current parent's static analysis: the score parts it is missing, as concrete opportunities
Lessons are deduplicated and the most recent ones are kept, so repeated mistakes are called out once
with a repeat count.

Memory backends: by default lessons live only for the current run. With a store
(local_store.LocalStore or rawtree_store.RawTreeStore) every lesson and candidate outcome is also
persisted, and both the lessons and the real-GPU timings from earlier runs on the same problem and
hardware seed the first prompt: a measurement is the one thing a later run cannot recompute from
the code, so what was already timed, and what it cost, is carried forward.
"""
import re
from collections import OrderedDict
from typing import Any, Dict, List, Optional

# Nonexistent tl.* names the model tends to invent (CUDA habits) -> what to do in Triton instead
TL_REPLACEMENTS = {
    "ceil_div": "use tl.cdiv(a, b)",
    "cdiv_": "use tl.cdiv(a, b)",
    "size": "pass the element count as a kernel argument (e.g. n_elements)",
    "numel": "pass the element count as a kernel argument (e.g. n_elements)",
    "sizeof": "use tl.constexpr sizes; Triton has no sizeof",
    "shared": "Triton has no manual shared-memory allocation: operate on block tensors "
              "(tl.zeros, tl.load) and the compiler stages them through shared memory",
    "alloc": "Triton has no manual shared-memory allocation: operate on block tensors "
             "(tl.zeros, tl.load) and the compiler stages them through shared memory",
    "shfl": "Triton has no warp intrinsics: reduce a whole block with tl.sum / tl.max / tl.reduce",
    "ballot": "Triton has no warp intrinsics: reduce a whole block with tl.sum / tl.max / tl.reduce",
    "warp": "Triton has no warp intrinsics: reduce a whole block with tl.sum / tl.max / tl.reduce",
    "thread": "Triton has no thread indices: use tl.program_id(0) and tl.arange(0, BLOCK)",
    "tid": "Triton has no thread indices: use tl.program_id(0) and tl.arange(0, BLOCK)",
    "block_dim": "Triton has no thread indices: use tl.program_id(0) and tl.arange(0, BLOCK)",
    "max_vector_width": "Triton vectorizes loads itself when a program loads a large contiguous block",
    "max_threads_per_block": "choose num_warps in triton.Config / the launch instead",
    "syncthreads": "Triton inserts barriers itself; there is no __syncthreads",
    "memcpy": "use tl.load / tl.store (or a TMA descriptor's .load/.store)",
    "atomic": "use tl.atomic_add(ptr, value)",
}
VALID_TL = ("tl.program_id, tl.num_programs, tl.arange, tl.zeros, tl.full, tl.load, tl.store, tl.sum, "
            "tl.max, tl.min, tl.dot, tl.where, tl.cdiv, tl.atomic_add, tl.make_tensor_descriptor, "
            "tl.range, tl.static_range, tl.constexpr, tl.float16, tl.float32")

MISSING_ATTR_RE = re.compile(r"has no attribute '([\w.]+)'")
SOURCE_AT_RE = re.compile(r"at \d+:\d+: (.*?) \.\.\. ")
SYNTAX_RE = re.compile(r"^line (\d+): (.*)$")
VALID_TL_HINT = f"Only use real triton.language APIs, e.g. {VALID_TL}."


def _tl_advice(name: str) -> str:
    low = name.lower()
    for key, advice in TL_REPLACEMENTS.items():
        if key in low:
            return advice
    return "it does not exist in Triton"


def _syntax_parts(check_error: str, code: str):
    """Split check_syntax()'s "line N: <msg>: <source>" using the code itself (msg may contain ':')."""
    m = SYNTAX_RE.match(check_error)
    if not m:
        return None
    lineno, rest = int(m.group(1)), m.group(2)
    lines = code.splitlines()
    source = lines[lineno - 1].strip() if 0 < lineno <= len(lines) else ""
    msg = rest[:-len(source)].rstrip(": ") if source and rest.endswith(source) else rest
    return msg, source or rest


def lessons_from_failure(check_error: Optional[str], agent_error: Optional[str], code: str = "") -> List[str]:
    """Rules derived from one failed candidate (or a failed agent call)."""
    lessons: List[str] = []
    if agent_error:
        if "finish_reason=length" in agent_error or "truncated" in agent_error:
            lessons.append("Your reply was cut off at the output-token limit, so no code was received. "
                           "Keep reasoning short and return only the kernel and its launch wrapper, "
                           "without comments, docstrings, benchmarks or tests.")
        elif "Unterminated string" in agent_error or "Expecting" in agent_error:
            lessons.append("Your reply was not valid JSON. Return one JSON object; escape newlines in "
                           "\"code\" as \\n and quotes as \\\".")
        return lessons
    if not check_error:
        return lessons

    m = MISSING_ATTR_RE.search(check_error)
    if m:
        name = m.group(1)
        lessons.append(f"tl.{name} does not exist in Triton: {_tl_advice(name)}.")
        return lessons
    if "Unsupported function referenced" in check_error and "jit" in check_error:
        lessons.append("Do not define or decorate @triton.jit functions inside a kernel; define helper "
                       "kernels at module level and call them.")
        return lessons
    src = SOURCE_AT_RE.search(check_error)
    if src and src.group(1).lstrip().startswith("def "):
        lessons.append("Do not define functions inside a @triton.jit kernel "
                       f"(`{src.group(1).strip()[:60]}`); move helpers to module level with @triton.jit.")
        return lessons
    if check_error.startswith("cannot launch"):
        lessons.append(f"The kernel exceeded per-SM limits ({check_error.split(': ', 1)[-1]}). "
                       "Use smaller BLOCK sizes or fewer num_stages so shared memory fits.")
        return lessons
    parts = _syntax_parts(check_error, code)
    if parts:
        msg, line = parts
        if re.search(r"\d\.\d*f\b", line) or re.match(r"(int|float|double|auto|const)\s+\w+\s*=", line):
            lessons.append(f"Write Python, not C/CUDA: no type declarations and no 'f' float suffix "
                           f"(bad: `{line[:70]}`).")
        else:
            lessons.append(f"Python syntax error ({msg[:70]}) in `{line[:80]}`. Double-check parentheses "
                           "and keyword arguments in calls, especially the kernel launch.")
        return lessons
    if check_error.startswith("triton ("):
        detail = check_error.rsplit("): ", 1)[-1]
        where = SOURCE_AT_RE.search(detail)
        cause = detail.rsplit(" ... ", 1)[-1].strip()
        text = f"`{where.group(1).strip()[:80]}`" if where else ""
        if cause and cause != "^":
            text = f"{text}: {cause[:160]}" if text else cause[:160]
        lessons.append(f"Triton rejected {text or 'the kernel'}.")
    return lessons


def opportunities(static: Dict[str, Any], gemm: bool, capability: int) -> List[str]:
    """Concrete improvements for the kernel whose static metrics are given (score parts below max)."""
    if not static or "score_parts" not in static:
        return []
    ops: List[str] = []
    wl = static.get("wide_load_frac")
    if wl is not None and wl < 1.0:
        ops.append("Global loads are narrow (scalar). Have each program load a large contiguous block "
                   "(e.g. BLOCK_SIZE >= 1024 with num_warps=4, i.e. 8 fp32 per thread) so Triton emits "
                   "16-byte vector loads.")
    if static.get("spill_bytes", 0) > 0:
        ops.append(f"Registers spill ({static['spill_bytes']} B). Use smaller tiles or fewer num_stages.")
    occ = static.get("occupancy", 1.0)
    if not gemm and occ < 0.5:
        ops.append("Occupancy is low for a memory-bound kernel. Use fewer registers/shared memory per "
                   "program or 4-8 warps.")
    if gemm:
        if capability >= 90 and not static.get("tma"):
            ops.append("No TMA. Build device-side descriptors with tl.make_tensor_descriptor(ptr, shape=[M, K], "
                       "strides=[K, 1], block_shape=[BLOCK_M, BLOCK_K]) and load tiles with desc.load([m, k]); "
                       "the host wrapper must call triton.set_allocator(...).")
        if capability >= 90 and not static.get("warp_specialized"):
            ops.append("No warp specialization. Write the K loop as "
                       "`for k in tl.range(k_tiles, warp_specialize=True):` so producer warps overlap TMA "
                       "loads with MMAs.")
        native = {100: static.get("tcgen05"), 90: static.get("wgmma")}.get(capability, static.get("mma_sync"))
        if not native:
            ops.append("Tensor cores are not used natively. Use tl.dot on fp16 tiles (>= 64 in each dim) "
                       "with an fp32 accumulator.")
        tm, tn = static.get("tile_m"), static.get("tile_n")
        if tm and tn and 2.0 / (1.0 / tm + 1.0 / tn) < 128:
            ops.append(f"Output tile {tm}x{tn} gives poor data reuse; use BLOCK_M, BLOCK_N >= 128.")
    return ops


class Feedback:
    """Accumulates lessons across generations and renders them for the next prompt."""

    def __init__(self, gemm: bool, capability: int, max_lessons: int = 8, max_measured: int = 6,
                 store: Any = None, context: Optional[Dict[str, Any]] = None):
        self.gemm = gemm
        self.capability = capability
        self.max_lessons = max_lessons
        self.store = store  # optional persistent memory, e.g. rawtree_store.RawTreeStore
        self.context = context or {}  # problem / hardware / model, attached to every stored event
        self._lessons: "OrderedDict[str, int]" = OrderedDict()  # lesson -> times seen (most recent last)
        self._last_near_miss: Optional[str] = None
        self._measured: List[Dict[str, Any]] = []  # real-GPU outcomes from earlier runs
        self._measured_this_run: Dict[str, Dict[str, Any]] = {}  # and from this one, by kernel name
        self.loaded_from_memory = 0
        self.measured_from_memory = 0
        if store is not None:
            problem, hardware = self.context.get("problem", ""), self.context.get("hardware", "")
            for lesson, n in store.load_lessons(problem, hardware, limit=max_lessons):
                self._lessons[lesson] = n
            self.loaded_from_memory = len(self._lessons)
            if hasattr(store, "load_benchmarks"):
                self._measured = store.load_benchmarks(problem, hardware, limit=max_measured)
            self.measured_from_memory = len(self._measured)

    def _add(self, lesson: str, generation: Optional[int] = None) -> None:
        self._lessons[lesson] = self._lessons.pop(lesson, 0) + 1
        while len(self._lessons) > self.max_lessons:
            self._lessons.popitem(last=False)
        if self.store is not None:
            self.store.record("lesson", self.context, generation=generation, lesson=lesson)

    def observe_seed(self, variant: Any) -> None:
        """Record the seed's own GPU timing. It is the number every candidate has to beat, so it
        belongs in memory even though the seed is not a candidate and has no parent."""
        if not getattr(variant, "bench_status", ""):
            return
        self._measured_this_run[variant.name] = {
            "name": variant.name, "optimization_type": variant.optimization_type,
            "latency_us": getattr(variant, "measured_latency_us", None), "bench_status": variant.bench_status}
        if self.store is not None:
            self.store.record("candidate", self.context, generation=0, name=variant.name,
                              status=variant.status, optimization_type=variant.optimization_type,
                              static_score=variant.static_score,
                              measured_latency_us=getattr(variant, "measured_latency_us", None),
                              bench_status=variant.bench_status)

    def observe(self, variant: Any, parent: Any) -> None:
        """Record what the checks found for `variant`, which was generated from `parent`."""
        if self.store is not None:
            self.store.record("candidate", self.context, generation=variant.generation, name=variant.name,
                              status=variant.status, optimization_type=variant.optimization_type,
                              check_error=variant.check_error or "",
                              agent_error=variant.agent_error or "", static_score=variant.static_score,
                              parent_score=parent.static_score,
                              measured_latency_us=getattr(variant, "measured_latency_us", None),
                              bench_status=getattr(variant, "bench_status", ""))
        if variant.status == "failed" or variant.agent_error:
            for lesson in lessons_from_failure(variant.check_error, variant.agent_error, variant.code):
                self._add(lesson, variant.generation)
            return
        bench_status = getattr(variant, "bench_status", "")
        if bench_status:
            self._measured_this_run[variant.name] = {
                "name": variant.name, "optimization_type": variant.optimization_type,
                "latency_us": getattr(variant, "measured_latency_us", None), "bench_status": bench_status}
        if bench_status not in ("", "ok"):
            # It compiled and scored well, and was still wrong or unrunnable on the real GPU
            self._add(f"A kernel that passed every static check was rejected when run on the GPU: {bench_status}. "
                      "Cover the whole input exactly once, zero-initialise any buffer you accumulate into, "
                      "and make the host wrapper return the final result.", variant.generation)
            return
        measured, parent_measured = (getattr(variant, "measured_latency_us", None),
                                     getattr(parent, "measured_latency_us", None))
        if measured is not None and parent_measured is not None and measured >= parent_measured:
            self._last_near_miss = (f"Your previous kernel was timed on the real GPU at {measured} μs versus the "
                                    f"parent's {parent_measured} μs, so it was not kept. The static score cannot "
                                    f"separate these two; change the memory access pattern, not the block size alone.")
            return
        if variant.static_score is not None and parent.static_score is not None \
                and variant.static_score <= parent.static_score and variant.code != parent.code:
            parts, base = variant.static.get("score_parts", {}), parent.static.get("score_parts", {})
            worse = [k for k in base if parts.get(k, 0) < base[k]]
            self._last_near_miss = (f"Your previous kernel compiled but scored {variant.static_score} vs the "
                                    f"parent's {parent.static_score}"
                                    + (f" (lower on: {', '.join(worse)})" if worse else " (no component improved)")
                                    + ". Change something that raises a missing score component.")

    def _measured_section(self) -> str:
        """What has actually been timed on this GPU, from earlier runs and this one."""
        rows: Dict[str, Dict[str, Any]] = {r["name"]: r for r in self._measured}
        rows.update(self._measured_this_run)  # this run's result wins for the same kernel name
        timed = sorted((r for r in rows.values() if r.get("latency_us")), key=lambda r: r["latency_us"])
        rejected = [r for r in rows.values() if not r.get("latency_us") and r.get("bench_status") not in ("", "ok")]
        if not timed and not rejected:
            return ""
        lines = []
        for r in timed[:6]:
            strategy = f" [{r['optimization_type']}]" if r.get("optimization_type") else ""
            lines.append(f"- {r['latency_us']} μs: {r['name']}{strategy}")
        for r in rejected[:3]:
            strategy = f" [{r['optimization_type']}]" if r.get("optimization_type") else ""
            lines.append(f"- wrong or unrunnable on the GPU: {r['name']}{strategy} — {r['bench_status']}")
        header = ("Real GPU measurements from this and earlier runs (median wall-clock, lower is better; "
                  "these are timings, not estimates):")
        footer = (f"\nThe kernel to beat ran in {timed[0]['latency_us']} μs. The strategies listed above have "
                  f"already been measured at that speed or slower, so a variation of them will not win: "
                  f"change the memory access pattern or the amount of work per program."
                  if timed else "")
        return header + "\n" + "\n".join(lines) + footer

    def render(self, parent: Any) -> str:
        sections = []
        if self._lessons:
            rules = [f"- {lesson}" + (f" (seen {n}x)" if n > 1 else "") for lesson, n in reversed(self._lessons.items())]
            if any(lesson.startswith(("tl.", "Triton rejected")) for lesson in self._lessons):
                rules.append(f"- {VALID_TL_HINT}")
            sections.append("Compiler feedback from previous generations (these mistakes were rejected; do not repeat them):\n"
                            + "\n".join(rules))
        measured = self._measured_section()
        if measured:
            sections.append(measured)
        if self._last_near_miss:
            sections.append(self._last_near_miss)
        ops = opportunities(parent.static, self.gemm, self.capability)
        if ops:
            sections.append("Static analysis of the current kernel found these improvement opportunities:\n"
                            + "\n".join(f"- {o}" for o in ops))
        return "\n\n".join(sections)
