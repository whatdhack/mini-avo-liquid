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

    def load_lessons(self, problem: str, hardware: str, limit: int = 20) -> List[Tuple[str, int]]:
        """(lesson, times seen) from earlier runs on this problem/hardware, oldest first."""
        if not self.enabled:
            return []
        sql = (f"SELECT lesson, count() AS n, max(ts) AS last_ts FROM {self.table} "
               f"WHERE kind = 'lesson' AND problem = {sql_string(problem)} AND hardware = {sql_string(hardware)} "
               f"GROUP BY lesson ORDER BY last_ts DESC LIMIT {int(limit)}")
        try:
            _, result = self._request("/v1/query", {"sql": sql, "format": "JSON"})
        except RawTreeError as e:
            # A missing table just means no earlier run has written feedback yet
            if "HTTP 404" in str(e) or "UNKNOWN_TABLE" in str(e) or "doesn't exist" in str(e):
                return []
            self._disable(e)
            return []
        rows = (result or {}).get("data", [])
        return [(str(r["lesson"]), int(r["n"])) for r in reversed(rows) if r.get("lesson")]
