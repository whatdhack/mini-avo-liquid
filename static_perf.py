"""GPU-less static performance estimate for Triton kernels.

Compiles the kernel for the target GPU (see triton_check.compile_entries), then reads:
  * cubin resource usage (cuobjdump): registers per thread, local-memory spill bytes
  * compile metadata: dynamic shared memory, num_warps, num_stages
  * PTX instruction mix: tensor-core ops (tcgen05 / wgmma / mma.sync), global load/store widths,
    async copies (cp.async / TMA bulk copies), atomics, setmaxnreg
  * TTGIR / metadata: warp specialization (ttg.warp_specialize), TMEM columns (tmem_size),
    global scratch (device-side TMA descriptors need triton.set_allocator at launch)
and turns them into theoretical occupancy and a 0-100 static score.

This is a proxy, not a timing: it flags clearly bad kernels (spills, can't launch, no tensor cores
on a GEMM, scalar loads, low occupancy) but cannot rank well-tuned kernels reliably.
"""
import os
import re
import subprocess
import tempfile
from typing import Any, Dict, Optional, Tuple

from triton_check import HW_CAPABILITY, compile_entries

# Problems dominated by tensor-core math vs. by memory bandwidth
# Cholesky is scored GEMM-like: a fast implementation is blocked, and its trailing update is a
# SYRK, so tensor cores and tile reuse are the right things to reward
GEMM_PROBLEMS = {"matmul_v2", "trimul_alphafold3", "cholesky"}

# Per-SM limits by compute capability: (max warps, max blocks); 64K registers on all of these
SM_LIMITS = {80: (64, 32), 89: (48, 24), 90: (64, 32), 100: (64, 32)}
REGS_PER_SM = 65536
SPILL_BYTES_ZERO_SCORE = 512  # spill bytes at which the spill component reaches 0
# GEMM DRAM traffic scales with (1/BM + 1/BN); a harmonic-mean tile of 128+ gets full reuse credit
FULL_REUSE_TILE = 128

GLOBAL_MEM_RE = re.compile(r"\b(ld|st)\.global((?:\.[\w]+)*)")
ASYNC_COPY_RE = re.compile(r"\bcp\.async\.(bulk[\w.]*|c[ag]\.shared\.global[\w.]*)")
TYPE_BITS_RE = re.compile(r"^[bfsu](\d+)$")


def _cuobjdump_path() -> str:
    import triton
    return os.path.join(os.path.dirname(triton.__file__), "backends", "nvidia", "bin", "cuobjdump")


def _resource_usage(cubin: bytes) -> Dict[str, int]:
    """REG / LOCAL (spill) / SHARED (static) from `cuobjdump --dump-resource-usage`."""
    with tempfile.NamedTemporaryFile(suffix=".cubin", delete=False) as f:
        f.write(cubin)
        path = f.name
    try:
        out = subprocess.run([_cuobjdump_path(), "--dump-resource-usage", path],
                             capture_output=True, text=True, timeout=60).stdout
    finally:
        os.unlink(path)
    usage = {}
    for key in ("REG", "LOCAL", "SHARED", "STACK"):
        m = re.search(rf"\b{key}:(\d+)", out)
        usage[key] = int(m.group(1)) if m else 0
    return usage


def _access_bytes(suffixes: str) -> int:
    """Bytes per thread for a ld/st.global instruction, e.g. '.v4.b32' -> 16."""
    parts = [p for p in suffixes.split(".") if p]
    vec = next((int(p[1:]) for p in parts if re.fullmatch(r"v\d", p)), 1)
    bits = next((int(TYPE_BITS_RE.match(p).group(1)) for p in reversed(parts) if TYPE_BITS_RE.match(p)), 32)
    return vec * bits // 8


def _ptx_mix(ptx: str) -> Dict[str, Any]:
    loads = {"wide": 0, "medium": 0, "narrow": 0}
    stores = {"wide": 0, "medium": 0, "narrow": 0}
    for kind, suffixes in GLOBAL_MEM_RE.findall(ptx):
        nbytes = _access_bytes(suffixes)
        bucket = "wide" if nbytes >= 16 else "medium" if nbytes >= 8 else "narrow"
        (loads if kind == "ld" else stores)[bucket] += 1
    async_copies = ASYNC_COPY_RE.findall(ptx)
    # 16-byte cp.async.cg and TMA bulk copies are wide global loads that land directly in shared memory
    loads["wide"] += sum(1 for c in async_copies if c.startswith("bulk") or c.startswith("cg"))
    loads["medium"] += sum(1 for c in async_copies if c.startswith("ca"))
    return {
        "loads": loads,
        "stores": stores,
        "async_copy": bool(async_copies),
        "tma": any(c.startswith("bulk") for c in async_copies),
        "tma_tensor_copies": len(re.findall(r"\bcp\.async\.bulk\.tensor\b", ptx)),
        "setmaxnreg": len(re.findall(r"\bsetmaxnreg\b", ptx)),
        "tcgen05": len(re.findall(r"\btcgen05\.mma\b", ptx)),
        "wgmma": len(re.findall(r"\bwgmma\.mma_async\b", ptx)),
        "mma_sync": len(re.findall(r"\bmma\.sync\b", ptx)),
        "atomics": len(re.findall(r"\b(?:atom|red)\.global\b", ptx)),
        "instructions": sum(1 for line in ptx.splitlines() if line.strip().endswith(";")),
    }


def _width_score(counts: Dict[str, int]) -> Optional[float]:
    total = sum(counts.values())
    return None if total == 0 else (counts["wide"] + 0.5 * counts["medium"]) / total


def _occupancy(regs: int, shared_bytes: int, num_warps: int, capability: int, smem_per_sm: int) -> Tuple[int, float]:
    max_warps, max_blocks = SM_LIMITS.get(capability, (64, 32))
    regs_per_warp = ((max(regs, 1) + 7) // 8 * 8) * 32
    by_regs = (REGS_PER_SM // regs_per_warp) // num_warps
    by_smem = smem_per_sm // shared_bytes if shared_bytes else max_blocks
    by_warps = max_warps // num_warps
    blocks = max(0, min(by_regs, by_smem, by_warps, max_blocks))
    return blocks, blocks * num_warps / max_warps


# Registers per thread: the hardware cap is 255, and a kernel that lands there is not blocking
# well, it is one allocation away from spilling. Credit peaks in the healthy band and falls to
# zero at the cap. Measured on the GB10: the fastest 4096^3 GEMM found so far uses 226 registers
# at 12% occupancy, which the occupancy term alone penalizes.
REG_MAX = 255
REG_BAND = (32, 96, 224)  # zero below the first, full between the middle and the third


def _register_band(regs: int, spill_bytes: int) -> float:
    if spill_bytes or regs < REG_BAND[0]:
        return 0.0
    if regs < REG_BAND[1]:
        return (regs - REG_BAND[0]) / (REG_BAND[1] - REG_BAND[0])
    if regs <= REG_BAND[2]:
        return 1.0
    return max(0.0, (REG_MAX - regs) / (REG_MAX - REG_BAND[2]))


def _score(problem_id: str, capability: int, m: Dict[str, Any]) -> Dict[str, float]:
    """Weighted 0-100 score; component weights differ for GEMM-like vs memory-bound problems."""
    spill = max(0.0, 1.0 - m["spill_bytes"] / SPILL_BYTES_ZERO_SCORE)
    load_width = m["wide_load_frac"] if m["wide_load_frac"] is not None else 1.0
    if problem_id in GEMM_PROBLEMS:
        native = {100: m["tcgen05"], 90: m["wgmma"]}.get(capability, m["mma_sync"])
        tensor = 1.0 if native else 0.5 if m["mma_sync"] or m["wgmma"] else 0.0
        occupancy = min(1.0, m["occupancy"] / 0.125)  # GEMMs run fine at 1 block of 8 warps per SM
        if m["tile_m"] and m["tile_n"]:
            reuse = min(1.0, 2.0 / (1.0 / m["tile_m"] + 1.0 / m["tile_n"]) / FULL_REUSE_TILE)
        else:
            reuse = 0.5  # tile sizes not identifiable from constexpr names
        registers = _register_band(m["registers"], m["spill_bytes"])
        # Global atomics in a GEMM mean split-K accumulating into C. Measured on the GB10, every
        # split-K variant landed in the slow tail (3196, 3498, 5135 us) while every non-atomic
        # kernel came in at 2701 us or better.
        no_atomics = 0.0 if m["atomics"] else 1.0
        if capability >= 120:
            # Consumer Blackwell (sm_121, GB10): measured, TMA and warp specialization buy nothing
            # here. The fastest 4096^3 GEMM measured on this GPU uses neither and beats the best
            # TMA + warp-specialized kernel by 7.6%, while the Hopper-style bonuses below scored it
            # 77.5 against that kernel's 100. So there is no bonus for either on this target.
            parts = {"tensor_cores": 20 * tensor, "tile_reuse": 25 * reuse, "load_width": 10 * load_width,
                     "occupancy": 10 * occupancy, "no_spills": 15 * spill, "registers": 10 * registers,
                     "no_atomics": 10 * no_atomics}
        elif capability >= 90:
            # Hopper/datacenter Blackwell: TMA bulk-tensor copies beat per-thread cp.async, and
            # warp-specialized producer/consumer partitions overlap TMA loads with tensor-core MMAs
            staged = m["num_stages"] >= 2
            movement = 1.0 if m["tma"] else 0.5 if m["async_copy"] and staged else 0.0
            parts = {"tensor_cores": 20 * tensor, "tile_reuse": 15 * reuse, "load_width": 10 * load_width,
                     "tma": 5 * movement, "warp_specialization": 5 * float(m["warp_specialized"]),
                     "occupancy": 10 * occupancy, "no_spills": 15 * spill, "registers": 10 * registers,
                     "no_atomics": 10 * no_atomics}
        else:
            pipelined = 1.0 if m["async_copy"] and m["num_stages"] >= 2 else 0.0
            parts = {"tensor_cores": 25 * tensor, "tile_reuse": 10 * reuse, "load_width": 15 * load_width,
                     "async_pipeline": 10 * pipelined, "occupancy": 5 * occupancy, "no_spills": 15 * spill,
                     "registers": 10 * registers, "no_atomics": 10 * no_atomics}
    else:
        occupancy = min(1.0, m["occupancy"] / 0.5)  # bandwidth-bound kernels need warps in flight
        parts = {"load_width": 50 * load_width, "occupancy": 30 * occupancy, "no_spills": 20 * spill}
    return {k: round(v, 1) for k, v in parts.items()}


def _tile_sizes(constexprs: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
    """Output tile (BM, BN) from constexprs like BLOCK_SIZE_M / BLOCK_M / BLOCK_N."""
    def find(dim: str) -> Optional[int]:
        for k, v in constexprs.items():
            if "BLOCK" in k.upper() and k.upper().rstrip("_").endswith(dim) and isinstance(v, int):
                return v
        return None
    return find("M"), find("N")


def analyze(code: str, problem_id: str, hardware_id: str, smem_per_sm_kb: float) -> Tuple[Optional[str], Dict[str, Any]]:
    """Return (error or None, static metrics incl. 'score') for the main kernel in `code`."""
    error, entries = compile_entries(code, problem_id, hardware_id)
    if error:
        return error, {}
    if not entries:
        return None, {}  # Triton not installed

    capability = HW_CAPABILITY.get(hardware_id, 90)
    # The main kernel is the entry with the most PTX instructions
    per_kernel = [(e, _ptx_mix(e.compiled.asm["ptx"])) for e in entries]
    entry, mix = max(per_kernel, key=lambda em: em[1]["instructions"])
    ck = entry.compiled
    usage = _resource_usage(ck.asm["cubin"])
    shared_bytes = ck.metadata.shared + usage["SHARED"]
    num_warps, num_stages = ck.metadata.num_warps, ck.metadata.num_stages
    blocks, occupancy = _occupancy(usage["REG"], shared_bytes, num_warps, capability, int(smem_per_sm_kb * 1024))

    metrics: Dict[str, Any] = {
        "kernel": entry.name,
        "target": f"sm_{capability}",
        "registers": usage["REG"],
        "spill_bytes": usage["LOCAL"],
        "shared_bytes": shared_bytes,
        "num_warps": num_warps,
        "num_stages": num_stages,
        "blocks_per_sm": blocks,
        "occupancy": round(occupancy, 3),
        "tcgen05": mix["tcgen05"],
        "wgmma": mix["wgmma"],
        "mma_sync": mix["mma_sync"],
        "async_copy": mix["async_copy"],
        "tma": mix["tma_tensor_copies"] > 0,
        "warp_specialized": "ttg.warp_specialize" in ck.asm.get("ttgir", ""),
        "setmaxnreg": mix["setmaxnreg"],
        "tmem_columns": getattr(ck.metadata, "tmem_size", 0) or 0,
        "needs_allocator": (getattr(ck.metadata, "global_scratch_size", 0) or 0) > 0,
        "wide_load_frac": None if _width_score(mix["loads"]) is None else round(_width_score(mix["loads"]), 2),
        "wide_store_frac": None if _width_score(mix["stores"]) is None else round(_width_score(mix["stores"]), 2),
        "atomics": mix["atomics"],
        "ptx_instructions": mix["instructions"],
    }
    metrics["tile_m"], metrics["tile_n"] = _tile_sizes(entry.constexprs)
    if blocks == 0:
        return (f"cannot launch on sm_{capability}: {shared_bytes} B shared / {usage['REG']} regs x "
                f"{num_warps} warps exceed per-SM limits"), metrics
    metrics["score_parts"] = _score(problem_id, capability, metrics)
    metrics["score"] = round(sum(metrics["score_parts"].values()), 1)
    return None, metrics


def summarize(m: Dict[str, Any]) -> str:
    """One-line human summary of analyze() metrics."""
    if not m:
        return "n/a"
    tc = ("tcgen05" if m["tcgen05"] else "wgmma" if m["wgmma"] else "mma.sync" if m["mma_sync"] else "none")
    wl = "n/a" if m["wide_load_frac"] is None else f"{m['wide_load_frac']:.0%}"
    return (f"score {m.get('score', 0)}/100 | regs {m['registers']} spill {m['spill_bytes']}B | "
            f"smem {m['shared_bytes'] // 1024}KB | {m['num_warps']}w x {m['num_stages']}st | "
            f"occ {m['occupancy']:.0%} ({m['blocks_per_sm']} blk/SM) | TC {tc} | wide loads {wl}"
            + (f" | tile {m['tile_m']}x{m['tile_n']}" if m.get("tile_m") and m.get("tile_n") else "")
            + (" | TMA" if m["tma"] else " | cp.async" if m["async_copy"] else "")
            + (" | warp-specialized" if m.get("warp_specialized") else "")
            + (f" | TMEM {m['tmem_columns']} cols" if m.get("tmem_columns") else "")
            + (" | needs triton.set_allocator" if m.get("needs_allocator") else ""))
