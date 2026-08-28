"""Run the approved Phase 8C-2B CIFAR-10.1 v6 cross-source material evaluation."""

from __future__ import annotations

import json

from visionlab.experiments.phase8c import run_phase8c2b_material_evaluation


if __name__ == "__main__":
    result = run_phase8c2b_material_evaluation()
    print(json.dumps(result.to_dict(), indent=2))
