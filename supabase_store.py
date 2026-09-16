"""Minimal server-only Supabase REST adapter for production events."""

from __future__ import annotations

import json
import os
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


class SupabaseStore:
    def __init__(self) -> None:
        self.url = os.environ["SUPABASE_URL"].rstrip("/")
        self.key = os.environ["SUPABASE_SECRET_KEY"]

    def request(self, method: str, path: str, *, body: object | None = None, headers: dict[str, str] | None = None) -> tuple[object, dict[str, str]]:
        merged = {"apikey": self.key, "Authorization": f"Bearer {self.key}"}
        if headers:
            merged.update(headers)
        data = json.dumps(body).encode() if body is not None else None
        if data:
            merged["Content-Type"] = "application/json"
        request = Request(f"{self.url}/rest/v1/{path}", data=data, method=method, headers=merged)
        try:
            with urlopen(request, timeout=30) as response:
                raw = response.read()
                return (json.loads(raw) if raw else None), dict(response.headers.items())
        except HTTPError as error:
            detail = error.read().decode(errors="replace")
            raise RuntimeError(f"Supabase request failed ({error.code}): {detail}") from error

    def select_all(self, query: str, chunk_size: int = 1_000) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        start = 0
        while True:
            rows, _ = self.request("GET", query, headers={"Range": f"{start}-{start + chunk_size - 1}"})
            rows = rows or []
            result.extend(rows)
            if len(rows) < chunk_size:
                return result
            start += chunk_size

    def dashboard_payload(self, artifact: dict[str, Any]) -> dict[str, Any]:
        rows = self.select_all("process_events?select=event_at,pass_or_fail,probability,values_json&is_dashboard_sample=eq.true&order=event_at.asc")
        events = [{"t": row["event_at"], "d": row["event_at"][:10], "y": int(row["pass_or_fail"] or 0), "p": round(float(row["probability"]), 4), "v": row["values_json"]} for row in rows]
        first, _ = self.request("GET", "process_events?select=event_at&order=event_at.asc&limit=1")
        last, _ = self.request("GET", "process_events?select=event_at&order=event_at.desc&limit=1")
        _, meta = self.request("GET", "process_events?select=event_id&limit=1", headers={"Prefer": "count=exact"})
        total = int(next((value for key, value in meta.items() if key.lower() == "content-range"), "*/0").split("/")[-1])
        specs = artifact["specs"]
        return {"threshold": artifact["threshold"], "metrics": artifact["metrics"], "labels": {key: artifact["labels"].get(key, key) for key in specs}, "specs": specs, "events": events, "source": {"rows": total, "from": first[0]["event_at"], "to": last[0]["event_at"]}}

    def record_prediction(self, event_at: str, values: dict[str, float | None], probability: float) -> None:
        self.request("POST", "process_events", body={"event_at": event_at, "pass_or_fail": None, "probability": probability, "values_json": values, "is_dashboard_sample": True}, headers={"Prefer": "return=minimal"})
