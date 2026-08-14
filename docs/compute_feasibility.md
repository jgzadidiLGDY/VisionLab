# T1 Compute and PyTorch Feasibility

Status: T1 accepted feasibility note.

Date: 2026-08-14

## Local Environment Probes

Base interpreter:

- Command: `python --version`
- Result: `Python 3.14.5`
- Python launcher: `py -0p` reported only `C:\Python314\python.exe`.

Pre-install optional dependency probe:

- Command: `powershell -ExecutionPolicy Bypass -File scripts\probe_environment.ps1`
- Result: `torch`, `torchvision`, `PIL`, `numpy`, and `matplotlib` were not installed in the base interpreter.
- Device status: `recommended_local_device = torch_not_installed`.

Local GPU visibility:

- Command: `nvidia-smi`
- Result: command not found.
- Interpretation: no NVIDIA GPU path is currently visible from this local shell.

## PyTorch Installation Probe

An ignored local `.venv` was created for T1 feasibility testing.

Install command:

```powershell
.\.venv\Scripts\python -m pip install torch torchvision
```

Observed result:

- `torch 2.13.0+cpu` installed.
- `torchvision 0.28.0+cpu` installed.
- `numpy 2.5.2` and `Pillow 12.3.0` were installed as transitive dependencies.
- `matplotlib` was not installed.

Runtime probe:

```powershell
$env:PYTHONPATH='src'
.\.venv\Scripts\python -m visionlab.environment
```

Observed result:

- Python: `3.14.5`
- `cpu_tensor_op_ok = true`
- `cuda_available = false`
- `cuda_device_count = 0`
- `recommended_local_device = cpu`

Tiny PyTorch convolution check:

```powershell
.\.venv\Scripts\python -c "import torch, torchvision; import torch.nn.functional as F; x=torch.arange(25,dtype=torch.float32).reshape(1,1,5,5); k=torch.ones((1,1,3,3)); y=F.conv2d(x,k); print(torch.__version__); print(torchvision.__version__); print(tuple(y.shape)); print(torch.cuda.is_available())"
```

Observed result:

```text
2.13.0+cpu
0.28.0+cpu
(1, 1, 3, 3)
False
```

## External Compatibility Evidence

- PyPI package lookup listed `torch 2.13.0` and `torchvision 0.28.0` for the local Python 3.14 interpreter.
- The torchvision repository compatibility table lists Python `>=3.10, <=3.14` for recent torch/torchvision pairings including `torch 2.12` with `torchvision 0.27` and later main/nightly tracks: https://github.com/pytorch/vision
- The official PyTorch previous-versions page lists current CPU and CUDA wheel installation routes for recent releases: https://pytorch.org/get-started/previous-versions/

## T1 Interpretation

The T0 risk that Python 3.14.5 might block PyTorch is reduced but not eliminated.

Supported local development path established by T1:

- Python 3.14.5 local `.venv`
- PyTorch CPU wheels
- deterministic tests and tiny tensor/convolution probes on CPU

Not established by T1:

- local NVIDIA CUDA training;
- Colab or remote GPU runtime;
- material training performance;
- long-run reproducibility;
- final dependency pinning for later training phases.

## Recommended Compute Boundary

Use local CPU for:

- deterministic unit and contract tests;
- tiny tensor, image, and model-shape smoke checks;
- artifact schema checks;
- very small training smoke tests when later approved.

Use approved remote GPU or another explicit compute route for:

- material training runs;
- transfer-learning fine-tuning;
- repeated experiments;
- runs expected to consume meaningful wall time or storage.
