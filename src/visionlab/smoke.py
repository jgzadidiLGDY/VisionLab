"""Minimal T0 smoke checks for the VisionLab repository."""

from __future__ import annotations

import sys
from pathlib import Path

from visionlab import __version__


def environment_summary() -> dict[str, str]:
    """Return cheap local environment facts without importing ML dependencies."""
    repo_root = Path(__file__).resolve().parents[2]
    return {
        "visionlab_version": __version__,
        "python_version": sys.version.split()[0],
        "repo_root": str(repo_root),
        "project_spec_exists": str((repo_root / "docs" / "project_specs.md").exists()),
    }


def main() -> int:
    summary = environment_summary()
    for key, value in summary.items():
        print(f"{key}: {value}")

    if summary["project_spec_exists"] != "True":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
