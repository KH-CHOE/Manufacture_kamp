"""Copy the local SQLite history to Supabase. Requires server-only credentials."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
import json

from database import connect
from supabase_store import SupabaseStore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replace", action="store_true", help="Delete existing Supabase events before import")
    args = parser.parse_args()
    store = SupabaseStore()
    with connect() as conn:
        rows = conn.execute("SELECT event_id, event_at, pass_or_fail, probability, values_json FROM process_events ORDER BY event_at, event_id").fetchall()
    sample_step = max(1, len(rows) // 8_000)
    sample_indices = set(range(0, len(rows), sample_step)) | set(range(max(0, len(rows) - 500), len(rows)))
    if args.replace:
        store.request("DELETE", "process_events?event_id=gt.0", headers={"Prefer": "return=minimal"})
    for start in range(0, len(rows), 500):
        batch = [{"event_at": row["event_at"], "pass_or_fail": row["pass_or_fail"], "probability": row["probability"], "values_json": json.loads(row["values_json"]), "is_dashboard_sample": index in sample_indices} for index, row in enumerate(rows[start:start + 500], start)]
        store.request("POST", "process_events", body=batch, headers={"Prefer": "return=minimal"})
    print(f"Copied {len(rows):,} events to Supabase.")


if __name__ == "__main__":
    main()
