"""Run the Phase 4A bounded baseline-plumbing smoke workflow."""

from __future__ import annotations

from pathlib import Path

from visionlab.experiments.phase4a import run_phase4a_smoke


if __name__ == "__main__":
    result = run_phase4a_smoke(Path("outputs") / "phase4a_smoke")
    print(result.to_dict())
