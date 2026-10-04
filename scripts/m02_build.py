"""Parse every training-week Annexure, merge external sources, and write the M02 master dataset."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pfp.build import build_all  # noqa: E402

if __name__ == "__main__":
    out = build_all(write=True)
    m = out["master"]
    print(f"master: {m.shape[0]} rows x {m.shape[1]} cols, {m.week_end.nunique()} weeks "
          f"({m.week_end.min().date()} to {m.week_end.max().date()})")
