import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from visionlab.experiments.phase7 import PHASE7_OUTPUT_DIR, run_phase7_evaluation


if __name__ == "__main__":
    result = run_phase7_evaluation(Path(PHASE7_OUTPUT_DIR))
    print(result.to_dict())
