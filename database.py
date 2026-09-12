"""Small SQLite store for process events and model predictions."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


BASE = Path(__file__).parent
DB_PATH = BASE / "data" / "foundry_v2.db"


def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def initialize() -> None:
    with connect() as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS process_events (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT, event_at TEXT NOT NULL, pass_or_fail INTEGER, probability REAL NOT NULL,
            values_json TEXT NOT NULL, created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )""")


def seed(events: pd.DataFrame, probabilities: np.ndarray, features: list[str]) -> int:
    initialize()
    rows = []
    for row, probability in zip(events.itertuples(index=False), probabilities, strict=True):
        values = {feature: float(getattr(row, feature)) if pd.notna(getattr(row, feature)) else None for feature in features}
        rows.append((row.event_at.strftime("%Y-%m-%d %H:%M:%S"), int(row.PassOrFail), float(probability), json.dumps(values)))
    with connect() as conn:
        # ponytail: this command rebuilds the historical baseline; incremental ingestion comes with MLOps.
        conn.execute("DELETE FROM process_events")
        conn.executemany("INSERT INTO process_events(event_at, pass_or_fail, probability, values_json) VALUES (?, ?, ?, ?)", rows)
    return len(rows)


def dashboard_payload(artifact: dict[str, Any], max_events: int = 8_000) -> dict[str, Any]:
    with connect() as conn:
        rows = conn.execute("SELECT event_at, pass_or_fail, probability, values_json FROM process_events ORDER BY event_at, event_id").fetchall()
    if len(rows) > max_events:
        step = max(1, len(rows) // max_events)
        indices = list(range(0, len(rows), step)) + list(range(max(0, len(rows) - 500), len(rows)))
        rows = [rows[index] for index in dict.fromkeys(indices)]
    events = [{"t": row["event_at"], "d": row["event_at"][:10], "y": int(row["pass_or_fail"] or 0), "p": round(row["probability"], 4), "v": json.loads(row["values_json"])} for row in rows]
    with connect() as conn:
        summary = conn.execute("SELECT COUNT(*) n, MIN(event_at) first_at, MAX(event_at) last_at FROM process_events").fetchone()
    specs = artifact["specs"]
    shown = list(specs)
    return {"threshold": artifact["threshold"], "metrics": artifact["metrics"], "labels": {key: artifact["labels"].get(key, key) for key in shown}, "specs": {key: specs[key] for key in shown}, "events": events, "source": {"rows": summary["n"], "from": summary["first_at"], "to": summary["last_at"]}}


def record_prediction(event_at: str, values: dict[str, float | None], probability: float) -> None:
    with connect() as conn:
        conn.execute("INSERT INTO process_events(event_at, pass_or_fail, probability, values_json) VALUES (?, NULL, ?, ?)", (event_at, probability, json.dumps(values)))
