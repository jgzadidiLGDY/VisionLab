"""Run the approved Phase 4B custom-CNN CIFAR-10 material baseline."""

from __future__ import annotations

from pathlib import Path

from visionlab.experiments.phase4b import PHASE4B_RUN_ID, run_phase4b_material_baseline


if __name__ == "__main__":
    result = run_phase4b_material_baseline(Path("outputs") / PHASE4B_RUN_ID)
    print(result.to_dict())
