"""Persistent feedback memory in a local JSONL file.

Same interface as rawtree_store.RawTreeStore, for runs that should remember what earlier runs
found without a RawTree account. Every lesson and candidate outcome is appended as one JSON
object per line; at startup, lessons and real-GPU timings from earlier runs on the same problem
and hardware are read back.

Real measurements are the reason this exists: a static score is reproducible from the code, but
"this exact strategy was timed at 1673 us on this GPU and did not beat the seed" is only known
by having run it, and is worth carrying into the next run.

Configuration (environment or .env):
    FEEDBACK_FILE   optional; defaults to .miniavo_memory.jsonl next to this file

File errors never stop a run: they are reported once and the run continues in memory.
"""
import json
import os
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_FILENAME = ".miniavo_memory.jsonl"
MAX_READ_BYTES = 8 * 1024 * 1024  # a run appends a few KB; cap the read so the file can never stall a run


def rank_candidate_events(events: List[Dict[str, Any]], limit: int = 20) -> List[Dict[str, Any]]:
    """Fastest timing per kernel name, then the kernels the GPU harness rejected.

    Both memory backends rank with this, so they agree, and so neither has to aggregate a
    schemaless numeric column in SQL: `min(measured_latency_us)` over a dynamic JSON column is
    exactly the kind of thing that either errors or compares as strings ("1000" < "900").
    """
    best: Dict[str, Dict[str, Any]] = {}
    rejected: Dict[str, Dict[str, Any]] = {}
    for event in events:
        name = str(event.get("name") or "?")
        status = str(event.get("bench_status") or "")
        try:
            latency = float(event["measured_latency_us"])
        except (KeyError, TypeError, ValueError):
            latency = 0.0
        row = {"name": name, "optimization_type": str(event.get("optimization_type") or ""),
               "latency_us": latency if latency > 0 else None, "bench_status": status}
        if latency > 0:
            if name not in best or latency < best[name]["latency_us"]:
                best[name] = row
        elif status not in ("", "ok"):
            rejected[name] = row
    ranked = sorted(best.values(), key=lambda r: r["latency_us"])
    return (ranked + [r for n, r in rejected.items() if n not in best])[:limit]


class LocalStore:
    def __init__(self, path: Optional[str] = None):
        self.path = path or os.getenv("FEEDBACK_FILE") or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), DEFAULT_FILENAME)
        self.table = self.path  # for the startup message, which reports where memory lives
        self.run_id = uuid.uuid4().hex[:12]
        self.enabled = True

    @classmethod
    def from_env(cls) -> "LocalStore":
        return cls()

    def _disable(self, err: Exception) -> None:
        if self.enabled:
            print(f"    [!] Local feedback memory disabled for this run: {err}")
        self.enabled = False

    def insert(self, events: List[Dict[str, Any]]) -> None:
        if not self.enabled or not events:
            return
        try:
            with open(self.path, "a") as f:
                for event in events:
                    f.write(json.dumps(event, default=str) + "\n")
        except OSError as e:
            self._disable(e)

    def record(self, kind: str, context: Dict[str, Any], **fields: Any) -> None:
        """Append one event: kind is 'lesson' or 'candidate'; context holds problem/hardware/model."""
        self.insert([{"ts": time.time(), "run_id": self.run_id, "kind": kind, **context, **fields}])

    def _events(self, kind: str, problem: str, hardware: str) -> List[Dict[str, Any]]:
        """Earlier runs' events of one kind for this problem and hardware, oldest first."""
        if not self.enabled or not os.path.isfile(self.path):
            return []
        try:
            with open(self.path) as f:
                raw = f.read(MAX_READ_BYTES)
        except OSError as e:
            self._disable(e)
            return []
        out = []
        for line in raw.splitlines():
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue  # a partially written line from an interrupted run
            if (event.get("kind") == kind and event.get("problem") == problem
                    and event.get("hardware") == hardware and event.get("run_id") != self.run_id):
                out.append(event)
        return out

    def load_lessons(self, problem: str, hardware: str, limit: int = 20) -> List[Tuple[str, int]]:
        """(lesson, times seen) from earlier runs on this problem/hardware, oldest first."""
        counts: Dict[str, int] = {}
        order: List[str] = []
        for event in self._events("lesson", problem, hardware):
            lesson = event.get("lesson")
            if not lesson:
                continue
            if lesson not in counts:
                order.append(lesson)
            counts[lesson] = counts.get(lesson, 0) + 1
        keep = order[-limit:]
        return [(lesson, counts[lesson]) for lesson in keep]

    def load_benchmarks(self, problem: str, hardware: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Real-GPU outcomes from earlier runs: the fastest timing per kernel name, plus the
        kernels the GPU harness rejected. Fastest first, rejects last."""
        return rank_candidate_events(self._events("candidate", problem, hardware), limit)
