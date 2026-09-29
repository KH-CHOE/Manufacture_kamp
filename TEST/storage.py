"""Select local SQLite for development or Supabase when server credentials exist."""

from __future__ import annotations

import os
from typing import Any

from database import dashboard_payload as local_dashboard_payload
from database import initialize as local_initialize
from database import record_prediction as local_record_prediction


def remote_enabled() -> bool:
    return bool(os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_SECRET_KEY"))


def initialize() -> None:
    if not remote_enabled():
        local_initialize()


def dashboard_payload(artifact: dict[str, Any]) -> dict[str, Any]:
    if remote_enabled():
        from supabase_store import SupabaseStore
        return SupabaseStore().dashboard_payload(artifact)
    return local_dashboard_payload(artifact)


def record_prediction(event_at: str, values: dict[str, float | None], probability: float) -> None:
    if remote_enabled():
        from supabase_store import SupabaseStore
        SupabaseStore().record_prediction(event_at, values, probability)
    else:
        local_record_prediction(event_at, values, probability)
