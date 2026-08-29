"""Run Phase 9C review synthesis artifacts."""

from __future__ import annotations

import json

from visionlab.experiments.phase9c import run_phase9c_review_closeout


if __name__ == "__main__":
    result = run_phase9c_review_closeout()
    print(json.dumps(result.to_dict(), indent=2))
