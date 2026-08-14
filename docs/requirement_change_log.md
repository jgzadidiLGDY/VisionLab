# VisionLab Requirement Change Log

Status: T1 accepted; approved T1 clarifications recorded.

Material requirement changes should be recorded here after builder review and approval. Minor clarifications may be summarized in the relevant phase closeout when they do not change scope, architecture, data integrity, experiments, or closure boundaries.

| Change ID | Date | Phase | Requirement | Decision | Evidence or rationale |
| --- | --- | --- | --- | --- | --- |
| None yet | 2026-08-13 | T0 | Initial project specification | No material requirement changes recorded. | T0 establishes governance before implementation. |

## Approved T1 Clarifications

These are approved clarifications from T1 evidence. They do not narrow the project-level Python requirement or select the applied domain.

| Recommendation ID | Date | Phase | Requirement area | Proposed clarification | Evidence or rationale | Builder decision |
| --- | --- | --- | --- | --- | --- | --- |
| T1-REC-001 | 2026-08-14 | T1 | Python and PyTorch support | Treat Python 3.14.5 with local `.venv` CPU PyTorch wheels as the verified local development path for T1/Phase 1 smoke work; do not narrow `requires-python` yet. | `.venv` installed `torch 2.13.0+cpu` and `torchvision 0.28.0+cpu`; CPU tensor and convolution probes passed; CUDA unavailable locally. | Approved 2026-08-14 |
| T1-REC-002 | 2026-08-14 | T1 | Deterministic local tests | Document `scripts/test.ps1` as the unambiguous local deterministic test command. | Raw `unittest` needs `PYTHONPATH=src`; `scripts/test.ps1` sets it explicitly and passed 8 tests. | Approved 2026-08-14 |
| T1-REC-003 | 2026-08-14 | T1 | Development dataset | Use CIFAR-10 as the provisional core development dataset for Phase 1 planning, without treating it as the applied-domain selection. | CIFAR-10 is RGB, balanced, small, widely supported, and suitable for custom-CNN fundamentals. | Approved 2026-08-14 |

## Entry Template

```markdown
## RC-YYYY-NNN — Short Title

- Date:
- Phase:
- Original requirement:
- Proposed revision:
- Technical or experimental evidence:
- Learning impact:
- Scope, architecture, test, and compute impact:
- Builder decision:
```
