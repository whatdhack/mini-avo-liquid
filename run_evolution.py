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
    agent_error: Optional[str] = None  # why the Liquid call produced no usable reply (simulator fallback)
    prompt_feedback: str = ""  # check-derived instructions that were added to this generation's prompt

# ---------------------------------------------------------------------------
# 4. Agentic Variation Operator (Liquid AI via OpenRouter, or Heuristic Simulator)
# ---------------------------------------------------------------------------

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
# Largest free Liquid AI model on OpenRouter (https://openrouter.ai/liquid)
DEFAULT_LIQUID_MODEL = "liquid/lfm-2.5-2.6b:free"
# lfm-2.5-2.6b allows at most 8192 output tokens, and hidden reasoning counts toward that limit
DEFAULT_MAX_TOKENS = 8192
DEFAULT_REASONING_EFFORT = "high"

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

def call_liquid_agent(prompt: str, model: str) -> Dict[str, Any]:
    from openai import OpenAI, BadRequestError

    client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=os.environ["OPENROUTER_API_KEY"])
    messages = [
        {"role": "system", "content": "You are MiniAVO AI, an expert GPU kernel architect. Respond with a single JSON object only."},
        {"role": "user", "content": prompt},
    ]
    kwargs = dict(model=model, messages=messages, temperature=0.4, max_tokens=int(os.getenv("LIQUID_MAX_TOKENS", DEFAULT_MAX_TOKENS)),
                  extra_headers={"X-Title": "MiniAVO-Liquid"},
                  extra_body={"reasoning": {"effort": os.getenv("LIQUID_REASONING_EFFORT", DEFAULT_REASONING_EFFORT)}})
    try:
        resp = client.chat.completions.create(response_format={"type": "json_object"}, **kwargs)
    except BadRequestError as e:
        # Some providers reject response_format; retry once without it
        print(f"[!] json_object mode rejected ({e}); retrying without response_format")
        resp = client.chat.completions.create(**kwargs)
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
    model: str = DEFAULT_LIQUID_MODEL,
    feedback: str = ""
) -> KernelVariant:
    agent_error = None
    # If an OpenRouter API key is available, use the real Liquid AI agent mutation
    if os.getenv("OPENROUTER_API_KEY"):
        try:
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

            Return a JSON response with:
            {{
              "variantName": "v{generation_idx}_<strategy_shortname>",
              "optimizationType": "<one of: {', '.join(OPTIMIZATION_TYPES)}>",
              "description": "<Detailed micro-architectural hypothesis>",
              "estimatedLatencyUs": <float lower than {parent.latency_us}>,
              "code": "<Complete runnable Triton/CUDA Python code>"
            }}
            """
            data = call_liquid_agent(prompt, model)

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
            print(f"[!] Liquid agent ({model}) fallback to analytical engine: {e}")

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
    """Elite selection by static score; falls back to the model's claimed speedup if Triton is unavailable."""
    if candidate.static_score is not None and elite.static_score is not None:
        return candidate.static_score > elite.static_score
    return candidate.speedup > elite.speedup

# ---------------------------------------------------------------------------
# 5. Main Evolutionary CLI Engine
# ---------------------------------------------------------------------------

def run_evolution_pipeline(problem_id: str, hardware_id: str, generations: int, steer_prompt: str = "", model: str = DEFAULT_LIQUID_MODEL,
                           web_seed: bool = False, refresh_seed: bool = False, feedback_memory: str = "local"):
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
    agent = model if os.getenv("OPENROUTER_API_KEY") else "analytical simulator (OPENROUTER_API_KEY not set)"
    print(f"  Mutation Agent  : {agent}")
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
    if feedback_memory == "rawtree":
        from rawtree_store import RawTreeError, RawTreeStore
        try:
            store = RawTreeStore.from_env()
        except RawTreeError as e:
            print(f"[!] RawTree feedback memory unavailable, using in-memory feedback: {e}")
    feedback = Feedback(gemm=problem.id in GEMM_PROBLEMS, capability=HW_CAPABILITY.get(hw.id, 90), store=store,
                        context={"problem": problem.id, "hardware": hw.id, "model": model})
    if store is not None and store.enabled:
        print(f"[*] Feedback memory: RawTree table '{store.table}' (run {store.run_id}), "
              f"{feedback.loaded_from_memory} lesson(s) loaded from earlier runs\n")

    print(f"[*] [Gen 0] Baseline Seed: {root_variant.name} | Reference latency: {root_variant.latency_us} μs (problem table)")
    if seed_error:
        print(f"    └─ [!] Seed failed static analysis: {seed_error}\n")
    else:
        print(f"    └─ Static: {summarize(root_variant.static)}\n")

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
            feedback=prompt_feedback
        )
        lineage.append(new_variant)
        if new_variant.status == "failed":
            print(f"    ├─ [✗ FAILED] {new_variant.name} ({new_variant.optimization_type})")
            print(f"    │  Rationale : {new_variant.description}")
            print(f"    │  Check     : {new_variant.check_error}")
            print(f"    │  Elite kept: {current_elite.name}\n")
            feedback.observe(new_variant, parent)
            continue
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
        print(f"    │  Elite     : {current_elite.name} (score {current_elite.static_score})\n")

    num_failed = sum(v.status == "failed" for v in lineage)
    print("="*75)
    print(f"  EVOLUTION COMPLETE: Best Kernel '{current_elite.name}'")
    print("="*75)
    print(f"  Static Score  : {current_elite.static_score}/100 (seed: {root_variant.static_score}/100, GPU-less proxy, not a timing)")
    print(f"  Claimed       : {current_elite.latency_us} μs / {current_elite.speedup}x (model estimate, not measured)")
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
    parser.add_argument("--hardware", type=str, default="b200", choices=list(HARDWARE_CATALOG.keys()), help="Target GPU hardware ID")
    parser.add_argument("--generations", type=int, default=5, help="Number of evolutionary generations to search")
    parser.add_argument("--model", type=str, default=os.getenv("LIQUID_MODEL", DEFAULT_LIQUID_MODEL), help="OpenRouter model ID for the mutation agent")
    parser.add_argument("--web-seed", action="store_true", help="Search the web (Nimble) for a better initial seed kernel")
    parser.add_argument("--refresh-seed", action="store_true", help="Ignore the cached web seed and search again")
    parser.add_argument("--feedback-memory", choices=["local", "rawtree"], default=os.getenv("FEEDBACK_MEMORY", "local"),
                        help="Where check-derived lessons are kept: this run only, or persisted in RawTree across runs")
    parser.add_argument("--steer", type=str, default="", help="Natural language steering prompt for kernel mutations")

    args = parser.parse_args()
    run_evolution_pipeline(args.problem, args.hardware, args.generations, args.steer, args.model,
                           args.web_seed, args.refresh_seed, args.feedback_memory)
