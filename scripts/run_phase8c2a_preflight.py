"""Run Phase 8C-2A cross-source evaluation preflight only."""

from __future__ import annotations

import json

from visionlab.experiments.phase8c import run_phase8c2a_preflight


if __name__ == "__main__":
    result = run_phase8c2a_preflight()
    print(json.dumps(result.to_dict(), indent=2))
