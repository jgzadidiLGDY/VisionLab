import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from visionlab.experiments.phase8b import PHASE8B2A_OUTPUT_DIR, run_phase8b2a_preflight


if __name__ == "__main__":
    result = run_phase8b2a_preflight(Path(PHASE8B2A_OUTPUT_DIR))
    print(result.to_dict())
