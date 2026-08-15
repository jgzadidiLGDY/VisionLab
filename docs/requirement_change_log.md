# VisionLab Requirement Change Log

Status: Phase 1 complete; approved T1 and Phase 1 clarifications recorded.

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

## Approved Phase 1 Clarifications

These are approved clarifications from Phase 1 implementation evidence. They do not select the applied domain, begin model work, or authorize training.

| Recommendation ID | Date | Phase | Requirement area | Proposed clarification | Evidence or rationale | Builder decision |
| --- | --- | --- | --- | --- | --- | --- |
| P1-REC-001 | 2026-08-14 | Phase 1 | Core development dataset | Register CIFAR-10 as the provisional core development dataset for Phase 2 planning. | Phase 1B recorded source/version/provenance, class mapping, deterministic split policy, class counts, visual inspection artifacts, and limitations. | Approved 2026-08-14 |
| P1-REC-002 | 2026-08-14 | Phase 1 | Split policy | Use upstream CIFAR-10 train/test partitions, derive a 5,000-image validation split from upstream train only, stratified at 500 samples per class with seed `20260814`, and leave upstream test untouched. | Generated class counts are exactly 45,000 train, 5,000 validation, and 10,000 test, with balanced per-class counts. | Approved 2026-08-14 |
| P1-REC-003 | 2026-08-14 | Phase 1 | Sample identity | Define CIFAR-10 sample IDs from upstream partition and upstream source index rather than derived VisionLab split assignment. | Stable IDs preserve sample identity if a future approved split policy changes train/validation membership. | Approved 2026-08-14 |
| P1-REC-004 | 2026-08-14 | Phase 1 | Preprocessing profile | Use deterministic Phase 1B inspection preprocessing: RGB 32x32, `[0.0, 1.0]` value range, `(0.5, 0.5, 0.5)` mean/std, no augmentation. | Mirrors the simple PyTorch CIFAR-10 tutorial convention and is not computed from test-set statistics. | Approved 2026-08-14 |

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
