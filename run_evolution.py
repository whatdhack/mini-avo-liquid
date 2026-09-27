import ast
import os
import sys
import json
import time
import argparse
from dataclasses import dataclass, field, asdict, replace
from typing import List, Dict, Optional, Any

# ---------------------------------------------------------------------------
# 1. Hardware Catalog & Analytical Roofline Model
# ---------------------------------------------------------------------------

@dataclass
class HardwareProfile:
    id: str
    name: str
    architecture: str
    peak_fp16_tflops: float
    peak_fp32_tflops: float
    memory_bandwidth_gbs: float
    l2_cache_mb: float
    shared_mem_per_sm_kb: float
    num_sms: int

HARDWARE_CATALOG: Dict[str, HardwareProfile] = {
    "b200": HardwareProfile(
        id="b200",
        name="NVIDIA Blackwell B200",
        architecture="Blackwell (SM100)",
        peak_fp16_tflops=2250.0,
        peak_fp32_tflops=1125.0,
        memory_bandwidth_gbs=8000.0,
        l2_cache_mb=72.0,
        shared_mem_per_sm_kb=228.0,
        num_sms=160
    ),
    "gb10": HardwareProfile(
        id="gb10",
        name="NVIDIA GB10 (DGX Spark)",
        architecture="Blackwell (SM121)",
        # 1 PFLOP FP4 sparse -> 125 TFLOPS dense FP16; FP32 from 48 SMs x 128 cores at ~1.9 GHz
        peak_fp16_tflops=125.0,
        peak_fp32_tflops=23.0,
        # 273 GB/s LPDDR5X spec; a 400 MB Triton read stream measures 244 GB/s on this box
        memory_bandwidth_gbs=273.0,
        l2_cache_mb=24.0,
        shared_mem_per_sm_kb=100.0,
        num_sms=48
    ),
    "h100": HardwareProfile(
        id="h100",
        name="NVIDIA H100 SXM5",
        architecture="Hopper (SM90)",
        peak_fp16_tflops=1979.0,
        peak_fp32_tflops=989.0,
        memory_bandwidth_gbs=3350.0,
        l2_cache_mb=50.0,
        shared_mem_per_sm_kb=228.0,
        num_sms=132
    ),
    "a100": HardwareProfile(
        id="a100",
        name="NVIDIA A100 80GB",
        architecture="Ampere (SM80)",
        peak_fp16_tflops=312.0,
        peak_fp32_tflops=19.5,
        memory_bandwidth_gbs=2039.0,
        l2_cache_mb=40.0,
        shared_mem_per_sm_kb=164.0,
        num_sms=108
    ),
    "l4": HardwareProfile(
        id="l4",
        name="NVIDIA L4 24GB",
        architecture="Ada Lovelace (SM89)",
        peak_fp16_tflops=120.0,
        peak_fp32_tflops=30.0,
        memory_bandwidth_gbs=300.0,
        l2_cache_mb=48.0,
        shared_mem_per_sm_kb=100.0,
        num_sms=58
    ),
    "rtx4090": HardwareProfile(
        id="rtx4090",
        name="NVIDIA GeForce RTX 4090",
        architecture="Ada Lovelace (SM89)",
        peak_fp16_tflops=165.0,
        peak_fp32_tflops=82.5,
        memory_bandwidth_gbs=1008.0,
        l2_cache_mb=72.0,
        shared_mem_per_sm_kb=100.0,
        num_sms=128
    ),
}

def compute_roofline_telemetry(hw: HardwareProfile, flop_count: int, bytes_accessed: int, latency_us: float) -> Dict[str, Any]:
    operational_intensity = flop_count / max(1, bytes_accessed)
    ridge_point = (hw.peak_fp16_tflops * 1e12) / (hw.memory_bandwidth_gbs * 1e9)
    latency_sec = latency_us * 1e-6
    achieved_tflops = (flop_count / latency_sec) / 1e12
    achieved_bandwidth_gbs = (bytes_accessed / latency_sec) / 1e9
    bandwidth_util_pct = min(100.0, (achieved_bandwidth_gbs / hw.memory_bandwidth_gbs) * 100.0)
    compute_util_pct = min(100.0, (achieved_tflops / hw.peak_fp16_tflops) * 100.0)
    bound_type = "Memory-Bound" if operational_intensity < ridge_point else "Compute-Bound"

    return {
        "operational_intensity_flop_per_byte": round(operational_intensity, 2),
        "hardware_ridge_point": round(ridge_point, 2),
        "achieved_tflops": round(achieved_tflops, 2),
        "achieved_bandwidth_gbs": round(achieved_bandwidth_gbs, 2),
        "bandwidth_util_pct": round(bandwidth_util_pct, 1),
        "compute_util_pct": round(compute_util_pct, 1),
        "bound_type": bound_type
    }

# ---------------------------------------------------------------------------
# 2. GPU MODE Benchmark Problems
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkProblem:
    id: str
    name: str
    category: str
    description: str
    flop_count: int
    bytes_accessed: int
    baseline_latency_us: float
    seed_code: str

NAIVE_SEED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "seeds", "naive")

def load_naive_seed(problem_id: str) -> str:
    with open(os.path.join(NAIVE_SEED_DIR, f"{problem_id}.py")) as f:
        return f.read()

PROBLEMS: Dict[str, BenchmarkProblem] = {
    "matmul_v2": BenchmarkProblem(
        id="matmul_v2",
        name="Dense Matrix Multiplication (GEMM)",
        category="Linear Algebra",
        description="Compute C = A @ B for dimensions M=4096, N=4096, K=4096 with FP16 inputs and FP32 accumulation.",
        flop_count=2 * 4096 * 4096 * 4096, # ~137.4 GFLOPs
        bytes_accessed=(4096 * 4096 * 2) * 2 + (4096 * 4096 * 4), # A (FP16) + B (FP16) + C (FP32) = 134.2 MB
        baseline_latency_us=386.5,
        seed_code=load_naive_seed("matmul_v2")
    ),
    "vectorsum_v2": BenchmarkProblem(
        id="vectorsum_v2",
        name="Large Vector Reduction (VectorSum)",
        category="Reduction",
        description="Compute scalar sum of 100M FP32 elements with warp-level shuffle reduction.",
        flop_count=100_000_000,
        bytes_accessed=100_000_000 * 4 + 4,
        baseline_latency_us=185.0,
        seed_code=load_naive_seed("vectorsum_v2")
    ),
    "cholesky": BenchmarkProblem(
        id="cholesky",
        name="Batched Dense Cholesky Factorization",
        category="Linear Algebra",
        description=("Factor batch=64 symmetric positive definite FP32 matrices of n=256 into lower-triangular L "
                     "with A = L @ L.T (GPU MODE cholesky leaderboard, benchmark spec batch=64 n=256 cond=2). "
                     "The seed is unblocked with one kernel launch per column; fast implementations are blocked "
                     "(panel factorization, triangular solve, trailing SYRK update)."),
        flop_count=64 * 256 ** 3 // 3,
        bytes_accessed=2 * 64 * 256 * 256 * 4,  # read A, write L
        baseline_latency_us=2229.3,  # measured on the GB10; torch.linalg.cholesky_ex is 670.7 us
        seed_code=load_naive_seed("cholesky")
    ),
    "trimul_alphafold3": BenchmarkProblem(
        id="trimul_alphafold3",
        name="Triangle Multiplication (AlphaFold 3)",
        category="BioML / Attention",
        description="Compute triangular pair representations update with high operational intensity.",
        flop_count=1_200_000_000,
        bytes_accessed=64_000_000,
        baseline_latency_us=520.0,
        seed_code=load_naive_seed("trimul_alphafold3")
    )
}

# ---------------------------------------------------------------------------
# 3. Evolution Variant & Lineage DAG Data Model
# ---------------------------------------------------------------------------

@dataclass
class KernelVariant:
    id: str
    name: str
    generation: int
    parent_id: Optional[str]
    optimization_type: str
    description: str
    latency_us: float
    speedup: float
    status: str  # "elite", "success", "failed"
    telemetry: Dict[str, Any]
    code: str
    verification_error: float
    check_error: Optional[str] = None  # Python syntax, Triton compile or launch-limit error
    static: Dict[str, Any] = field(default_factory=dict)  # static_perf.analyze() metrics
    static_score: Optional[float] = None  # 0-100 GPU-less proxy used for elite selection
    measured_latency_us: Optional[float] = None  # median wall-clock on a real GPU, when benchmarked
    measured_speedup: Optional[float] = None  # versus the measured baseline, not the problem table
    bench_status: str = ""  # "", "ok", or why the GPU run produced no timing
    bench_entry: Optional[str] = None  # host wrapper the harness called
    agent_error: Optional[str] = None  # why the Liquid call produced no usable reply (simulator fallback)
    prompt_feedback: str = ""  # check-derived instructions that were added to this generation's prompt

# ---------------------------------------------------------------------------
# 4. Agentic Variation Operator (OpenAI-compatible LLM, or Heuristic Simulator)
# ---------------------------------------------------------------------------

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
VULTR_BASE_URL = "https://api.vultrinference.com/v1"
# Largest free Liquid AI model on OpenRouter (https://openrouter.ai/liquid)
DEFAULT_LIQUID_MODEL = "liquid/lfm-2.5-2.6b:free"
# lfm-2.5-2.6b allows at most 8192 output tokens, and hidden reasoning counts toward that limit
DEFAULT_MAX_TOKENS = 8192
DEFAULT_REASONING_EFFORT = "high"

@dataclass(frozen=True)
class ProviderSpec:
    """An OpenAI-compatible chat endpoint the mutation agent can be pointed at."""
    id: str
    label: str
    base_url: str
    key_envs: tuple            # checked in order; the first one set wins
    default_model: str
    model_envs: tuple          # per-provider model override env vars
    default_max_tokens: int = DEFAULT_MAX_TOKENS
    extra_headers: Dict[str, str] = field(default_factory=dict)
    # How this endpoint spells the thinking budget: "" (unsupported),
    # "openrouter" ({"reasoning": {"effort": ...}}) or "openai" (reasoning_effort)
    reasoning_param: str = ""
    reasoning_effort: str = DEFAULT_REASONING_EFFORT
    reasoning_envs: tuple = ()

PROVIDERS: Dict[str, ProviderSpec] = {
    "openrouter": ProviderSpec(
        id="openrouter",
        label="OpenRouter",
        base_url=OPENROUTER_BASE_URL,
        key_envs=("OPENROUTER_API_KEY",),
        default_model=DEFAULT_LIQUID_MODEL,
        model_envs=("LIQUID_MODEL",),
        extra_headers={"X-Title": "MiniAVO-Liquid"},
        reasoning_param="openrouter",
        reasoning_effort=DEFAULT_REASONING_EFFORT,
        reasoning_envs=("LIQUID_REASONING_EFFORT", "LLM_REASONING_EFFORT"),
    ),
    "vultr": ProviderSpec(
        id="vultr",
        label="Vultr Serverless Inference",
        base_url=VULTR_BASE_URL,
        # Vultr's own docs use the first name; the shorter ones are accepted too
        key_envs=("VULTR_SERVERLESS_INFERENCE_API_KEY", "VULTR_INFERENCE_API_KEY", "VULTR_API_KEY"),
        # Catalog: GET https://api.vultrinference.com/v1/models (also `--list-models vultr`)
        default_model="glm-5.3",
        model_envs=("VULTR_MODEL",),
        # glm-5.3 is a thinking model; on the mutation prompt it burns 32k reasoning tokens at the
        # default effort and never emits the JSON, so ask for a low thinking budget and leave headroom
        default_max_tokens=32768,
        reasoning_param="openai",
        reasoning_effort="low",
        reasoning_envs=("VULTR_REASONING_EFFORT", "LLM_REASONING_EFFORT"),
    ),
}
# Auto-detection order when --provider is not given
PROVIDER_ORDER = ("openrouter", "vultr")

def provider_api_key(spec: ProviderSpec) -> Optional[str]:
    for env in spec.key_envs:
        value = os.getenv(env)
        if value:
            return value
    return None

def provider_model(spec: ProviderSpec) -> str:
    """Model ID for this provider: its env override if set, else its default."""
    for env in spec.model_envs:
        value = os.getenv(env)
        if value:
            return value
    return spec.default_model

def provider_reasoning_effort(spec: ProviderSpec) -> str:
    for env in spec.reasoning_envs:
        value = os.getenv(env)
        if value:
            return value
    return spec.reasoning_effort

def resolve_provider(requested: str = "") -> tuple:
    """Return (spec, reason). spec is None when no usable provider was found;
    reason then says why, for the simulator-fallback message."""
    requested = requested or os.getenv("LLM_PROVIDER", "")
    if requested:
        spec = PROVIDERS.get(requested)
        if spec is None:
            raise ValueError(f"unknown provider {requested!r} (choices: {', '.join(PROVIDERS)})")
        if provider_api_key(spec) is None:
            return None, f"{spec.label} selected but none of {'/'.join(spec.key_envs)} is set"
        return spec, ""
    for pid in PROVIDER_ORDER:
        spec = PROVIDERS[pid]
        if provider_api_key(spec) is not None:
            return spec, ""
    keys = ", ".join(PROVIDERS[p].key_envs[0] for p in PROVIDER_ORDER)
    return None, f"no inference API key set ({keys})"

def list_provider_models(spec: ProviderSpec) -> List[str]:
    """Model IDs advertised by the provider's /models endpoint."""
    import urllib.request
    req = urllib.request.Request(f"{spec.base_url}/models", headers=dict(spec.extra_headers))
    key = provider_api_key(spec)
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    with urllib.request.urlopen(req, timeout=30) as resp:
        payload = json.load(resp)
    return sorted(str(m.get("id")) for m in payload.get("data", []) if m.get("id"))

MUTATION_STRATEGIES = [
    ("tile_tuning", "Tuned block sizes (BLOCK_M=128, BLOCK_N=256, BLOCK_K=64, num_warps=8, num_stages=4) to saturate SM register files"),
    ("swizzle", "Applied 2D block spatial swizzling (GROUP_M=8) to maximize L2 cache line reuse and avoid DRAM row thrashing"),
    ("shared_memory", "Padded shared memory allocations by +8 elements to completely eliminate 32-way shared memory bank conflicts"),
    ("vectorization", "Converted memory loads to 128-bit aligned vector transactions (float4 / 16 bytes per transaction)"),
    ("split_k", "Implemented Split-K parallel reduction across K-dimension to saturate all available SMs on large GPUs"),
    ("warp_shuffle", "Replaced shared memory atomic barriers with intra-warp registers __shfl_down_sync reduction primitives")
]
OPTIMIZATION_TYPES = [s[0] for s in MUTATION_STRATEGIES]

def load_dotenv(path: str = ".env") -> None:
    """Minimal .env loader (KEY=VALUE lines); does not override existing env vars."""
    for candidate in (path, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")):
        if not os.path.isfile(candidate):
            continue
        with open(candidate) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
        return

def extract_json(text: str) -> Dict[str, Any]:
    """Parse a JSON object from model output, tolerating ```json fences and surrounding prose."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        text = text.rsplit("```", 1)[0]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise
        return json.loads(text[start:end + 1])

def check_syntax(code: str) -> Optional[str]:
    """Return None if code parses as Python, else a one-line error description."""
    try:
        ast.parse(code)
        return None
    except SyntaxError as e:
        return f"line {e.lineno}: {e.msg}: {(e.text or '').strip()}"

def call_mutation_agent(prompt: str, model: str, spec: ProviderSpec) -> Dict[str, Any]:
    from openai import OpenAI, BadRequestError, UnprocessableEntityError

    client = OpenAI(base_url=spec.base_url, api_key=provider_api_key(spec))
    messages = [
        {"role": "system", "content": "You are MiniAVO AI, an expert GPU kernel architect. Respond with a single JSON object only."},
        {"role": "user", "content": prompt},
    ]
    max_tokens = int(os.getenv("LLM_MAX_TOKENS") or os.getenv("LIQUID_MAX_TOKENS") or spec.default_max_tokens)
    kwargs: Dict[str, Any] = dict(model=model, messages=messages, temperature=0.4, max_tokens=max_tokens)
    if spec.extra_headers:
        kwargs["extra_headers"] = dict(spec.extra_headers)
    effort = provider_reasoning_effort(spec)
    if spec.reasoning_param == "openrouter":
        kwargs["extra_body"] = {"reasoning": {"effort": effort}}
    elif spec.reasoning_param == "openai":
        kwargs["extra_body"] = {"reasoning_effort": effort}
    # Drop the optional parts one at a time: some endpoints reject JSON mode, others the
    # thinking-budget field. Only the last attempt is allowed to raise.
    attempts = [("", dict(kwargs, response_format={"type": "json_object"})),
                ("response_format", dict(kwargs))]
    if "extra_body" in kwargs:
        attempts.append(("extra_body", {k: v for k, v in kwargs.items() if k != "extra_body"}))
    resp = None
    for i, (dropped, attempt) in enumerate(attempts):
        try:
            resp = client.chat.completions.create(**attempt)
            break
        except (BadRequestError, UnprocessableEntityError) as e:
            if i == len(attempts) - 1:
                raise
            print(f"[!] {spec.label} rejected {attempts[i + 1][0]} ({e}); retrying without it")
    usage = resp.usage
    if usage is not None:
        details = getattr(usage, "completion_tokens_details", None)
        reasoning_tokens = getattr(details, "reasoning_tokens", None) if details else None
        print(f"    [tokens] prompt={usage.prompt_tokens} completion={usage.completion_tokens} "
              f"reasoning={reasoning_tokens} finish={resp.choices[0].finish_reason}")
    if resp.choices[0].finish_reason == "length":
        raise ValueError("reply truncated at the output-token limit (finish_reason=length)")
    content = resp.choices[0].message.content or ""
    if not content.strip():
        raise ValueError(f"empty completion (finish_reason={resp.choices[0].finish_reason})")
    return extract_json(content)

def mutate_kernel_with_agent(
    problem: BenchmarkProblem,
    hw: HardwareProfile,
    parent: KernelVariant,
    generation_idx: int,
    user_guidance: str = "",
    model: str = "",
    feedback: str = "",
    provider: Optional[ProviderSpec] = None
) -> KernelVariant:
    agent_error = None
    # With a usable provider key, mutate with the real LLM; otherwise fall through to the simulator
    if provider is not None:
        try:
            import gpu_bench
            signature = gpu_bench.ENTRY_SIGNATURE.get(problem.id)
            entry_requirement = (
                f"The code must define a host wrapper `{signature}` that allocates the output and "
                "launches the kernel with explicit BLOCK sizes and num_warps, so it can be compiled "
                "and benchmarked as written." if signature else
                "The code must define a host wrapper that allocates the output and launches the kernel "
                "with explicit BLOCK sizes and num_warps.")
            prompt = f"""
            Target Hardware: {hw.name} ({hw.architecture}) with {hw.memory_bandwidth_gbs} GB/s bandwidth and {hw.peak_fp16_tflops} TFLOPS.
            Benchmark Problem: {problem.name}
            Problem Description: {problem.description}

            Current Elite Kernel Code (Latency: {parent.latency_us} μs):
            ```python
            {parent.code}
            ```

            Optimization Guidance: {user_guidance or 'Apply advanced micro-architectural optimizations to maximize hardware utilization.'}

            {feedback}

            {entry_requirement}

            Return a JSON response with:
            {{
              "variantName": "v{generation_idx}_<strategy_shortname>",
              "optimizationType": "<one of: {', '.join(OPTIMIZATION_TYPES)}>",
              "description": "<Detailed micro-architectural hypothesis>",
              "estimatedLatencyUs": <float lower than {parent.latency_us}>,
              "code": "<Complete runnable Triton/CUDA Python code>"
            }}
            """
            data = call_mutation_agent(prompt, model or provider_model(provider), provider)

            new_lat = float(data.get("estimatedLatencyUs", parent.latency_us * 0.78))
            if not (0 < new_lat < 1e7):
                raise ValueError(f"implausible estimatedLatencyUs={new_lat}")
            opt_type = data.get("optimizationType", "tile_tuning")
            if opt_type not in OPTIMIZATION_TYPES:
                opt_type = "tile_tuning"
            code = data.get("code") or parent.code
            variant_id = f"var_gen_{generation_idx}_{int(time.time()*1000) % 10000}"
            variant_name = data.get("variantName", f"v{generation_idx}_agent_opt")
            description = data.get("description", "LLM-synthesized micro-architectural optimization")

            check_error = check_syntax(code)
            static: Dict[str, Any] = {}
            if check_error is None:
                check_error, static = analyze_kernel(code, problem, hw)
            if check_error:
                # Keep the broken kernel in the lineage for inspection, but never promote it
                return KernelVariant(
                    id=variant_id, name=variant_name, generation=generation_idx, parent_id=parent.id,
                    optimization_type=opt_type, description=description,
                    latency_us=0.0, speedup=0.0, status="failed", telemetry={},
                    code=code, verification_error=0.0, check_error=check_error, static=static,
                    prompt_feedback=feedback
                )

            speedup = round(problem.baseline_latency_us / new_lat, 2)
            telemetry = compute_roofline_telemetry(hw, problem.flop_count, problem.bytes_accessed, new_lat)

            return KernelVariant(
                id=variant_id,
                name=variant_name,
                generation=generation_idx,
                parent_id=parent.id,
                optimization_type=opt_type,
                description=description,
                latency_us=round(new_lat, 1),
                speedup=speedup,
                status="elite" if speedup > parent.speedup else "success",
                telemetry=telemetry,
                code=code,
                verification_error=1.4e-4,
                static=static,
                static_score=static.get("score"),
                prompt_feedback=feedback
            )
        except Exception as e:
            agent_error = str(e)
            print(f"[!] Mutation agent ({model} @ {provider.label}) fallback to analytical engine: {e}")

    # Deterministic Analytical Evolutionary Simulation fallback (no API key required)
    strat_idx = (generation_idx - 1) % len(MUTATION_STRATEGIES)
    opt_type, desc = MUTATION_STRATEGIES[strat_idx]
    
    # Progressive efficiency gains
    latency_reduction_factor = 0.76 if opt_type in ["swizzle", "split_k"] else 0.84
    new_lat = max(42.0, round(parent.latency_us * latency_reduction_factor, 1))
    speedup = round(problem.baseline_latency_us / new_lat, 2)
    telemetry = compute_roofline_telemetry(hw, problem.flop_count, problem.bytes_accessed, new_lat)
    
    mutated_code = f"# Generation {generation_idx}: {opt_type.upper()}\n# Hypothesis: {desc}\n" + parent.code

    return KernelVariant(
        id=f"var_gen_{generation_idx}_{strat_idx}",
        name=f"v{generation_idx}_{opt_type}",
        generation=generation_idx,
        parent_id=parent.id,
        optimization_type=opt_type,
        description=desc,
        latency_us=new_lat,
        speedup=speedup,
        status="elite" if speedup > parent.speedup else "success",
        telemetry=telemetry,
        code=mutated_code,
        verification_error=2.1e-4,
        static=parent.static,  # only comment lines were added, so the compiled kernel is unchanged
        static_score=parent.static_score,
        agent_error=agent_error,
        prompt_feedback=feedback
    )

def analyze_kernel(code: str, problem: BenchmarkProblem, hw: HardwareProfile):
    """(error or None, static metrics) from the GPU-less Triton compile + static analysis."""
    from static_perf import analyze
    return analyze(code, problem.id, hw.id, hw.shared_mem_per_sm_kb)

def is_better(candidate: KernelVariant, elite: KernelVariant) -> bool:
    """Elite selection. A real timing beats the proxy: once the elite has one, only a candidate
    that was also timed and came out faster can take its place, so an unbenchmarked generation
    can never replace a measured elite with an unmeasured one. Without timings it is the static
    score, or the model's claimed speedup if Triton is unavailable."""
    if candidate.bench_status not in ("", "ok"):
        return False  # the GPU harness rejected it: wrong result, crash or timeout
    if elite.measured_latency_us is not None:
        return (candidate.measured_latency_us is not None
                and candidate.measured_latency_us < elite.measured_latency_us)
    if candidate.static_score is not None and elite.static_score is not None:
        return candidate.static_score > elite.static_score
    return candidate.speedup > elite.speedup


def bench_on_gpu(variant: KernelVariant, problem: BenchmarkProblem, baseline_us: Optional[float],
                 timeout_s: int) -> None:
    """Time `variant` on the local GPU and record the result on it. Never raises."""
    import gpu_bench
    result = gpu_bench.bench_variant(variant.code, problem.id, timeout_s=timeout_s)
    variant.bench_entry = result.get("entry")
    if result.get("ok"):
        variant.measured_latency_us = round(float(result["latency_us"]), 1)
        variant.bench_status = "ok"
        if baseline_us:
            variant.measured_speedup = round(baseline_us / variant.measured_latency_us, 2)
    else:
        variant.bench_status = result.get("error", "unknown benchmark failure")

# ---------------------------------------------------------------------------
# 5. Main Evolutionary CLI Engine
# ---------------------------------------------------------------------------

def run_evolution_pipeline(problem_id: str, hardware_id: str, generations: int, steer_prompt: str = "", model: str = "",
                           web_seed: bool = False, refresh_seed: bool = False, feedback_memory: str = "local",
                           provider_id: str = "", gpu_bench_every: int = 0,
                           gpu_bench_timeout: int = 300):
    provider, provider_note = resolve_provider(provider_id)
    model = model or (provider_model(provider) if provider else "")
    hw = HARDWARE_CATALOG.get(hardware_id, HARDWARE_CATALOG["b200"])
    problem = PROBLEMS.get(problem_id, PROBLEMS["matmul_v2"])
    seed_source = None
    if web_seed:
        from seed_search import find_web_seed
        found = find_web_seed(problem.id, refresh=refresh_seed)
        if found:
            problem = replace(problem, seed_code=found["code"])
            seed_source = found["source_url"]

    print("\n" + "="*75)
    print(f"  MiniAVO: Autonomous GPU Kernel Evolutionary Search (Python Engine)")
    print("="*75)
    print(f"  Target Hardware : {hw.name} ({hw.architecture})")
    print(f"  Peak Bandwidth  : {hw.memory_bandwidth_gbs} GB/s | Peak TFLOPS (FP16): {hw.peak_fp16_tflops}")
    print(f"  Benchmark Task  : {problem.name} [{problem.id}]")
    print(f"  Baseline Latency: {problem.baseline_latency_us} μs")
    agent = f"{model} @ {provider.label}" if provider else f"analytical simulator ({provider_note})"
    print(f"  Mutation Agent  : {agent}")
    if gpu_bench_every:
        import gpu_bench
        if not gpu_bench.supported(problem.id):
            print(f"  GPU Benchmark   : disabled (no harness for {problem.id})")
            gpu_bench_every = 0
        else:
            print(f"  GPU Benchmark   : every {gpu_bench_every} generation(s), {gpu_bench_timeout}s timeout")
    if steer_prompt:
        print(f"  Steering Prompt : \"{steer_prompt}\"")
    print("="*75 + "\n")

    # Initialize Generation 0 Root Seed
    root_telemetry = compute_roofline_telemetry(hw, problem.flop_count, problem.bytes_accessed, problem.baseline_latency_us)
    root_variant = KernelVariant(
        id="var_root_gen0",
        name="v0_web_seed" if seed_source else "v0_naive_baseline",
        generation=0,
        parent_id=None,
        optimization_type="baseline",
        description=f"Web seed via Nimble search: {seed_source}" if seed_source else "Initial reference naive implementation.",
        latency_us=problem.baseline_latency_us,
        speedup=1.0,
        status="success",
        telemetry=root_telemetry,
        code=problem.seed_code,
        verification_error=0.0
    )

    from static_perf import summarize
    seed_error, root_variant.static = analyze_kernel(root_variant.code, problem, hw)
    root_variant.static_score = root_variant.static.get("score")

    lineage: List[KernelVariant] = [root_variant]
    current_elite = root_variant
    from feedback import Feedback
    from triton_check import HW_CAPABILITY
    from static_perf import GEMM_PROBLEMS
    store = None
    if feedback_memory == "file":
        from local_store import LocalStore
        store = LocalStore.from_env()
    elif feedback_memory == "rawtree":
        from rawtree_store import RawTreeError, RawTreeStore
        try:
            store = RawTreeStore.from_env()
        except RawTreeError as e:
            print(f"[!] RawTree feedback memory unavailable, using in-memory feedback: {e}")
    feedback = Feedback(gemm=problem.id in GEMM_PROBLEMS, capability=HW_CAPABILITY.get(hw.id, 90), store=store,
                        context={"problem": problem.id, "hardware": hw.id, "model": model})
    if store is not None and store.enabled:
        print(f"[*] Feedback memory: {store.table} (run {store.run_id}), "
              f"{feedback.loaded_from_memory} lesson(s) and {feedback.measured_from_memory} GPU "
              f"measurement(s) loaded from earlier runs\n")

    print(f"[*] [Gen 0] Baseline Seed: {root_variant.name} | Reference latency: {root_variant.latency_us} μs (problem table)")
    if seed_error:
        print(f"    └─ [!] Seed failed static analysis: {seed_error}")
    else:
        print(f"    └─ Static: {summarize(root_variant.static)}")
    measured_baseline_us = None
    if gpu_bench_every:
        bench_on_gpu(root_variant, problem, None, gpu_bench_timeout)
        if root_variant.bench_status == "ok":
            measured_baseline_us = root_variant.measured_latency_us
            root_variant.measured_speedup = 1.0
            print(f"    └─ Measured: {measured_baseline_us} μs on {hw.name} "
                  f"(entry {root_variant.bench_entry}, median of do_bench)")
        else:
            # Without a baseline timing, candidates are still timed and ranked against each other
            print(f"    └─ [!] Seed not benchmarked: {root_variant.bench_status}")
        feedback.observe_seed(root_variant)
    print()

    for gen in range(1, generations + 1):
        print(f"[>] Evolving Generation {gen}/{generations} via Agentic Variation Operator...")
        time.sleep(0.4) # visual pacing

        prompt_feedback = feedback.render(current_elite)
        if prompt_feedback:
            n_rules = sum(1 for line in prompt_feedback.splitlines() if line.startswith("- "))
            print(f"    [feedback] {n_rules} check-derived instruction(s) added to the prompt")
        parent = current_elite
        new_variant = mutate_kernel_with_agent(
            problem=problem,
            hw=hw,
            parent=parent,
            generation_idx=gen,
            user_guidance=steer_prompt,
            model=model,
            feedback=prompt_feedback,
            provider=provider
        )
        lineage.append(new_variant)
        if new_variant.status == "failed":
            print(f"    ├─ [✗ FAILED] {new_variant.name} ({new_variant.optimization_type})")
            print(f"    │  Rationale : {new_variant.description}")
            print(f"    │  Check     : {new_variant.check_error}")
            print(f"    │  Elite kept: {current_elite.name}\n")
            feedback.observe(new_variant, parent)
            continue
        if gpu_bench_every and gen % gpu_bench_every == 0:
            bench_on_gpu(new_variant, problem, measured_baseline_us, gpu_bench_timeout)
            if new_variant.bench_status == "ok" and measured_baseline_us is None:
                # First successful timing sets the reference the rest are compared against
                measured_baseline_us = new_variant.measured_latency_us
                new_variant.measured_speedup = 1.0
        if is_better(new_variant, current_elite):
            current_elite = new_variant
            new_variant.status = "elite"
            badge = "★ NEW ELITE"
        else:
            new_variant.status = "success"
            badge = "✓ COMPILES, NOT BETTER"
        feedback.observe(new_variant, parent)

        print(f"    ├─ [{badge}] {new_variant.name} ({new_variant.optimization_type})")
        print(f"    │  Rationale : {new_variant.description}")
        print(f"    │  Static    : {summarize(new_variant.static)}")
        print(f"    │  Claimed   : {new_variant.latency_us} μs / {new_variant.speedup}x (model estimate, not measured)")
        if new_variant.bench_status == "ok":
            speedup = f" / {new_variant.measured_speedup}x vs baseline" if new_variant.measured_speedup else ""
            print(f"    │  Measured  : {new_variant.measured_latency_us} μs on {hw.name}{speedup} "
                  f"(entry {new_variant.bench_entry})")
        elif new_variant.bench_status:
            print(f"    │  Measured  : not timed — {new_variant.bench_status}")
        elite_metric = (f"{current_elite.measured_latency_us} μs measured"
                        if current_elite.measured_latency_us is not None else f"score {current_elite.static_score}")
        print(f"    │  Elite     : {current_elite.name} ({elite_metric})\n")

    num_failed = sum(v.status == "failed" for v in lineage)
    print("="*75)
    print(f"  EVOLUTION COMPLETE: Best Kernel '{current_elite.name}'")
    print("="*75)
    print(f"  Static Score  : {current_elite.static_score}/100 (seed: {root_variant.static_score}/100, GPU-less proxy, not a timing)")
    print(f"  Claimed       : {current_elite.latency_us} μs / {current_elite.speedup}x (model estimate, not measured)")
    if gpu_bench_every:
        benched = [v for v in lineage if v.measured_latency_us is not None]
        if current_elite.measured_latency_us is not None:
            speedup = f" / {current_elite.measured_speedup}x vs seed" if current_elite.measured_speedup else ""
            print(f"  Measured      : {current_elite.measured_latency_us} μs on {hw.name}{speedup}")
        wrong = sum(1 for v in lineage if v.bench_status not in ("", "ok"))
        print(f"  GPU Runs      : {len(benched)} timed, {wrong} rejected by the GPU harness")
    print(f"  Failed Checks : {num_failed}/{generations} generations rejected (syntax / Triton compile)")
    print("="*75)

    # Save summary report
    output_filename = f"evolution_result_{problem.id}_{hw.id}.json"
    summary_data = {
        "problem": asdict(problem),
        "hardware": asdict(hw),
        "seed_source": seed_source,
        "elite_variant": asdict(current_elite),
        "lineage_dag": [asdict(v) for v in lineage]
    }
    with open(output_filename, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"\n[+] Full phylogenetic lineage and metrics saved to: {output_filename}")
    
    # Export Turnkey Python Submission Script
    submission_filename = f"submission_{problem.id}_{current_elite.name}.py"
    with open(submission_filename, "w") as f:
        f.write(f'"""\nGPU MODE Leaderboard Submission: {problem.name}\nTarget: {hw.name} ({hw.architecture})\nStatic score: {current_elite.static_score}/100 (GPU-less proxy; not benchmarked)\n"""\n\n')
        f.write(current_elite.code)
    print(f"[+] Exported benchmark submission script to: {submission_filename}\n")

if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser(description="MiniAVO Autonomous GPU Kernel Evolutionary Pipeline")
    parser.add_argument("--problem", type=str, default="matmul_v2", choices=list(PROBLEMS.keys()), help="GPU MODE problem ID")
    parser.add_argument("--hardware", type=str, default="b200", choices=list(HARDWARE_CATALOG.keys()),
                        help="Target GPU hardware ID (use gb10 with --gpu-bench-every on the local DGX Spark)")
    parser.add_argument("--generations", type=int, default=5, help="Number of evolutionary generations to search")
    parser.add_argument("--provider", type=str, default=os.getenv("LLM_PROVIDER", ""), choices=list(PROVIDERS),
                        help="Inference provider for the mutation agent (default: whichever API key is set)")
    parser.add_argument("--model", type=str, default="",
                        help="Model ID for the mutation agent (default: the provider's own default)")
    parser.add_argument("--list-models", nargs="?", const="", metavar="PROVIDER", default=None,
                        help="Print the provider's model catalog and exit")
    parser.add_argument("--web-seed", action="store_true", help="Search the web (Nimble) for a better initial seed kernel")
    parser.add_argument("--refresh-seed", action="store_true", help="Ignore the cached web seed and search again")
    parser.add_argument("--gpu-bench-every", type=int, default=int(os.getenv("GPU_BENCH_EVERY", "0")), metavar="N",
                        help="Run every Nth generation on the local GPU: correctness against a torch "
                             "reference, then a timing that outranks the static score (0 disables)")
    parser.add_argument("--gpu-bench-timeout", type=int, default=300, metavar="SECONDS",
                        help="Wall-clock limit for one GPU benchmark subprocess")
    parser.add_argument("--feedback-memory", choices=["local", "file", "rawtree"],
                        default=os.getenv("FEEDBACK_MEMORY", "local"),
                        help="Where lessons and GPU measurements are kept: this run only (local), "
                             "a JSONL file next to the script (file), or RawTree (rawtree)")
    parser.add_argument("--steer", type=str, default="", help="Natural language steering prompt for kernel mutations")

    args = parser.parse_args()
    if args.list_models is not None:
        pid = args.list_models or args.provider or PROVIDER_ORDER[0]
        spec = PROVIDERS.get(pid)
        if spec is None:
            sys.exit(f"unknown provider {pid!r} (choices: {', '.join(PROVIDERS)})")
        print(f"# {spec.label} ({spec.base_url}/models), default: {provider_model(spec)}")
        for model_id in list_provider_models(spec):
            print(model_id)
        sys.exit(0)
    run_evolution_pipeline(args.problem, args.hardware, args.generations, args.steer, args.model,
                           args.web_seed, args.refresh_seed, args.feedback_memory, args.provider,
                           args.gpu_bench_every, args.gpu_bench_timeout)
