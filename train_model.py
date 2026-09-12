"""One command to train the model, save its pkl artifact, and seed SQLite."""

from __future__ import annotations

from build_dashboard import train_and_save_model
from database import seed


if __name__ == "__main__":
    events, probabilities, artifact = train_and_save_model()
    count = seed(events, probabilities, artifact["features"])
    assert count == len(events)
    print(f"Saved model and loaded {count:,} process events into SQLite.")
