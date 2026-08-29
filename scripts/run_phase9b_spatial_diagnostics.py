"""Run Phase 9B spatial diagnostics."""

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from visionlab.experiments.phase9b import PHASE9B_OUTPUT_DIR, run_phase9b_spatial_diagnostics


def main() -> None:
    result = run_phase9b_spatial_diagnostics(Path(PHASE9B_OUTPUT_DIR))
    print(f"Phase 9B status: {result.status}")
    print(f"Artifacts: {result.run_dir}")


if __name__ == "__main__":
    main()