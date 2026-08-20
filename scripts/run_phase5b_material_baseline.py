"""Run the approved Phase 5B custom-CNN augmentation comparison."""

from __future__ import annotations

from pathlib import Path

from visionlab.experiments.phase5b import PHASE5B_RUN_ID, run_phase5b_material_baseline


if __name__ == "__main__":
    result = run_phase5b_material_baseline(Path("outputs") / PHASE5B_RUN_ID)
    print(result.to_dict())
