"""Export model output for the static browser dashboard.

Python stays on the data/model side; the interface itself lives in dist/.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from build_dashboard import build_payload


OUT = BASE / "dist" / "data.js"


def export() -> Path:
    OUT.parent.mkdir(exist_ok=True)
    payload = json.dumps(build_payload(), ensure_ascii=False, separators=(",", ":"))
    OUT.write_text(f"window.FOUNDRY_DATA={payload};\n", encoding="utf-8")
    return OUT


if __name__ == "__main__":
    output = export()
    assert output.exists() and output.stat().st_size > 100_000
    print(f"Wrote {output.name} ({output.stat().st_size:,} bytes)")
