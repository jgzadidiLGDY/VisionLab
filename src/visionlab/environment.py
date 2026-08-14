"""Environment and optional PyTorch probes for VisionLab T1."""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import platform
import sys
from dataclasses import dataclass, asdict


OPTIONAL_PACKAGES = ("torch", "torchvision", "PIL", "numpy", "matplotlib")


@dataclass(frozen=True)
class PackageStatus:
    """Import and version status for an optional package."""

    name: str
    installed: bool
    version: str | None = None
    import_error: str | None = None


def _distribution_name(import_name: str) -> str:
    if import_name == "PIL":
        return "Pillow"
    return import_name


def package_status(import_name: str) -> PackageStatus:
    """Return whether an optional package imports and, if possible, its version."""
    try:
        module = importlib.import_module(import_name)
    except Exception as exc:  # pragma: no cover - exact optional failure varies by host.
        return PackageStatus(
            name=import_name,
            installed=False,
            import_error=f"{type(exc).__name__}: {exc}",
        )

    version = getattr(module, "__version__", None)
    if version is None:
        try:
            version = importlib.metadata.version(_distribution_name(import_name))
        except importlib.metadata.PackageNotFoundError:
            version = "unknown"
    return PackageStatus(name=import_name, installed=True, version=str(version))


def torch_device_probe() -> dict[str, object]:
    """Probe PyTorch enough to verify import, a CPU tensor op, and device reporting."""
    status = package_status("torch")
    result: dict[str, object] = {
        "available": status.installed,
        "version": status.version,
        "import_error": status.import_error,
        "cpu_tensor_op_ok": False,
        "cuda_available": False,
        "cuda_device_count": 0,
        "mps_available": False,
        "recommended_local_device": "unverified",
    }
    if not status.installed:
        result["recommended_local_device"] = "torch_not_installed"
        return result

    torch = importlib.import_module("torch")
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    result["cpu_tensor_op_ok"] = bool(torch.equal(x @ torch.eye(2), x))

    cuda_available = bool(torch.cuda.is_available())
    result["cuda_available"] = cuda_available
    result["cuda_device_count"] = int(torch.cuda.device_count())
    if cuda_available and result["cuda_device_count"]:
        result["cuda_device_name_0"] = torch.cuda.get_device_name(0)

    backends = getattr(torch, "backends", None)
    mps = getattr(backends, "mps", None) if backends is not None else None
    result["mps_available"] = bool(mps is not None and mps.is_available())

    if cuda_available:
        result["recommended_local_device"] = "cuda"
    elif result["mps_available"]:
        result["recommended_local_device"] = "mps"
    elif result["cpu_tensor_op_ok"]:
        result["recommended_local_device"] = "cpu"
    else:
        result["recommended_local_device"] = "torch_imported_but_tensor_probe_failed"
    return result


def environment_summary() -> dict[str, object]:
    """Return T1 environment facts without requiring ML dependencies."""
    packages = [asdict(package_status(name)) for name in OPTIONAL_PACKAGES]
    return {
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": packages,
        "torch_device": torch_device_probe(),
    }


def main() -> int:
    print(json.dumps(environment_summary(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
