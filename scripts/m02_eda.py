"""Build the master dataset and run the full Milestone 02 EDA (tables, figures, numbers JSON)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pfp.build import build_all  # noqa: E402
from pfp.run_eda import run_all  # noqa: E402

if __name__ == "__main__":
    r = run_all(build_all(write=True))
    print(f"figures: {len(r['figs'])}; numbers written")
