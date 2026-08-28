"""Run Phase 8C-1 CIFAR-10.1 v6 local-availability and tiny-smoke preflight."""

from __future__ import annotations

import json
from pathlib import Path

from visionlab.data.cifar10_1 import write_phase8c1_preflight_artifacts


PHASE8C1_RUN_ID = "phase8c1-cifar10-1-registration-visual-qa-tiny-smoke"
PHASE8C1_OUTPUT_DIR = Path("outputs") / PHASE8C1_RUN_ID


if __name__ == "__main__":
    result = write_phase8c1_preflight_artifacts(PHASE8C1_OUTPUT_DIR)
    print(json.dumps(result, indent=2))
