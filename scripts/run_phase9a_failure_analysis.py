"""Run Phase 9A failure tables and error galleries."""

from __future__ import annotations

from visionlab.experiments.phase9a import run_phase9a_failure_analysis


if __name__ == "__main__":
    result = run_phase9a_failure_analysis()
    print(result.to_dict())
