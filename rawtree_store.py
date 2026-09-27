"""Persistent feedback memory in RawTree (https://rawtree.com, Tinybird's analytics DB for unstructured data).

Every lesson and candidate outcome is appended as a JSON event to one RawTree table (created on the
first insert, no schema needed). At startup, lessons from earlier runs on the same problem and hardware
are loaded back with SQL, so a new run starts with the mistakes previous runs already made.

Configuration (environment or .env):
    RAWTREE_API_KEY         API key (rt_...), read_write permission
    RAWTREE_DATABASE        optional; defaults to the key's default database
    RAWTREE_FEEDBACK_TABLE  optional; defaults to "miniavo_feedback"
    RAWTREE_API_URL         optional; defaults to https://api.rawtree.com

RawTree failures never stop a run: they are reported once and the run continues with in-memory feedback.
"""
import json
import os
import re
import time
import urllib.error
import urllib.request
import uuid
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_API_URL = "https://api.rawtree.com"
DEFAULT_TABLE = "miniavo_feedback"
IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")


def sql_string(value: str) -> str:
    """ClickHouse single-quoted string literal (the query API has no parameter binding)."""
    return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"


class RawTreeError(RuntimeError):
    pass


class RawTreeStore:
    def __init__(self, api_key: str, table: str = DEFAULT_TABLE, database: Optional[str] = None,
                 api_url: str = DEFAULT_API_URL, timeout: float = 20.0):
        if not IDENTIFIER_RE.match(table):
            raise ValueError(f"invalid RawTree table name: {table!r}")
        self._api_key = api_key
        self.table = table
        self.database = database
        self.api_url = api_url.rstrip("/")
        self.timeout = timeout
        self.run_id = uuid.uuid4().hex[:12]
        self.enabled = True

    @classmethod
    def from_env(cls) -> "RawTreeStore":
        key = os.getenv("RAWTREE_API_KEY")
        if not key:
            raise RawTreeError("RAWTREE_API_KEY is not set (create one with read_write permission in the "
                               "RawTree dashboard or `rtree key create --permission read_write`)")
        return cls(key, table=os.getenv("RAWTREE_FEEDBACK_TABLE", DEFAULT_TABLE),
                   database=os.getenv("RAWTREE_DATABASE") or None,
                   api_url=os.getenv("RAWTREE_API_URL", DEFAULT_API_URL))

    def _request(self, path: str, body: Any) -> Tuple[int, Any]:
        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}
        if self.database:
            headers["x-rawtree-database"] = self.database
        req = urllib.request.Request(self.api_url + path, data=json.dumps(body).encode(), headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read()
                return resp.status, json.loads(raw) if raw else None
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:300]
            raise RawTreeError(f"HTTP {e.code} on {path}: {detail}") from e
        except (urllib.error.URLError, TimeoutError) as e:
            raise RawTreeError(f"cannot reach {self.api_url}: {e}") from e

    def _disable(self, err: Exception) -> None:
        if self.enabled:
            print(f"    [!] RawTree feedback memory disabled for this run: {err}")
        self.enabled = False

    def insert(self, events: List[Dict[str, Any]]) -> None:
        if not self.enabled or not events:
            return
        try:
            self._request(f"/v1/tables/{self.table}", events)
        except RawTreeError as e:
            self._disable(e)

    def record(self, kind: str, context: Dict[str, Any], **fields: Any) -> None:
        """Append one event: kind is 'lesson' or 'candidate'; context holds problem/hardware/model."""
        self.insert([{"ts": time.time(), "run_id": self.run_id, "kind": kind, **context, **fields}])

    def load_benchmarks(self, problem: str, hardware: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Real-GPU outcomes from earlier runs: the fastest timing per kernel name, plus the
        kernels the GPU harness rejected. Fastest first, rejects last.

        The rows are ranked in Python rather than with min()/ORDER BY, because
        measured_latency_us is a dynamic column in a schemaless table: aggregating or comparing
        it in SQL depends on the type RawTree inferred, and rows written before GPU benchmarking
        existed do not have the key at all.
        """
        if not self.enabled:
            return []
        sql = (f"SELECT name, optimization_type, measured_latency_us, bench_status FROM {self.table} "
               f"WHERE kind = 'candidate' AND problem = {sql_string(problem)} "
               f"AND hardware = {sql_string(hardware)} AND run_id != {sql_string(self.run_id)} "
               f"ORDER BY ts DESC LIMIT {int(limit) * 20}")
        from local_store import rank_candidate_events
        return rank_candidate_events(self._query_rows(sql, fatal=False), limit)

    def _query_rows(self, sql: str, fatal: bool = True) -> List[Dict[str, Any]]:
        """Rows for one query. `fatal=False` keeps a failing read from switching off the whole
        memory backend: a query that cannot run is a reason to lose that one answer, not a
        reason to stop recording lessons for the rest of the run."""
        try:
            _, result = self._request("/v1/query", {"sql": sql, "format": "JSON"})
        except RawTreeError as e:
            # A missing table or column just means no earlier run has written this yet
            if "HTTP 404" in str(e) or "UNKNOWN_TABLE" in str(e) or "doesn't exist" in str(e) \
                    or "UNKNOWN_IDENTIFIER" in str(e):
                return []
            if not fatal:
                print(f"    [!] RawTree query failed, continuing without its result: {e}")
                return []
            self._disable(e)
            return []
        return (result or {}).get("data", [])

    def load_lessons(self, problem: str, hardware: str, limit: int = 20) -> List[Tuple[str, int]]:
        """(lesson, times seen) from earlier runs on this problem/hardware, oldest first."""
        if not self.enabled:
            return []
        sql = (f"SELECT lesson, count() AS n, max(ts) AS last_ts FROM {self.table} "
               f"WHERE kind = 'lesson' AND problem = {sql_string(problem)} AND hardware = {sql_string(hardware)} "
               f"GROUP BY lesson ORDER BY last_ts DESC LIMIT {int(limit)}")
        rows = self._query_rows(sql)
        return [(str(r["lesson"]), int(r["n"])) for r in reversed(rows) if r.get("lesson")]
