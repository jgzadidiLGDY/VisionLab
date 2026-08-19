# VisionLab

VisionLab is an AI-native computer-vision engineering laboratory that progresses from model fundamentals to controlled evaluation and a focused applied domain-transfer study.

The intended project will build and train a custom CNN, compare it with one pretrained vision model, evaluate both beyond aggregate accuracy, inspect their confidence and failures, measure robustness and out-of-distribution behavior, and then apply the resulting methodology to one controlled/synthetic-to-real or equivalent domain gap.

It is not intended to be a production inspection system, medical or forensic authority, broad computer-vision framework, vision-language-model project, or benchmark-chasing exercise.

This repository has completed **Phase 4 - Custom CNN Baseline Experiment**. CIFAR-10 is registered as the provisional core development dataset, and the first single-run custom CNN baseline is preserved as a controlled comparison reference; inference, pretrained models, augmentation, calibration, robustness, OOD, diagnostics, and applied-domain behavior should not yet be assumed unless repository inspection confirms otherwise.

---

## Project Direction

VisionLab follows a deliberately staged learning and engineering strategy:

> trustworthy data → custom model fundamentals → reproducible training → transfer learning → evaluation-centered MVP+ → applied domain-transfer capstone

Its central technical question is:

> How do model architecture and training-data distribution affect generalization, robustness, confidence, and failure behavior when a vision model moves from controlled training conditions toward less-controlled real-world images?

The project is designed to grow through four maturity levels:

```text
Learning Foundation
  → Model-Engineering MVP
  → Evaluation-Centered MVP+
  → Applied Domain-Transfer Capstone
```

The strong MVP+ core should be independently closable before the applied expansion begins. The applied capstone should extend a stable core rather than place the entire project at risk.

---

## Why This Project Exists

Many introductory vision projects start with a pretrained model, train it on a familiar benchmark, and report one accuracy number. That approach can produce a working demo without developing a strong understanding of image tensors, convolution, training dynamics, data leakage, calibration, or real-world failure.

Models can also appear successful while relying on narrow datasets, source-specific artifacts, background correlations, or overconfident predictions. In-distribution accuracy alone often hides:

- class-specific weaknesses;
- poor probability calibration;
- sensitivity to blur, compression, noise, or lighting;
- collapse on another data source;
- high-confidence mistakes;
- reliance on unintended visual shortcuts.

VisionLab addresses both the learning and engineering sides of this problem. It begins with a custom CNN and grows into a controlled model-comparison and domain-transfer laboratory.

The project does not require every experiment to produce an improvement. A null or mixed result is valid when the data, controls, artifacts, and interpretation are sound.

---

## Project Classification

VisionLab is planned as a **strong MVP+ to bounded-substantial, portfolio-grade computer-vision project**.

Its depth is expected to come from:

- image and convolution fundamentals;
- a custom PyTorch CNN;
- reproducible training and checkpointing;
- transfer learning and fine-tuning;
- controlled model comparison;
- class-wise evaluation and calibration;
- degradation robustness;
- OOD and cross-source testing;
- systematic failure analysis;
- model-appropriate diagnostics;
- data-centric intervention;
- controlled/synthetic-to-real evaluation;
- disciplined AI-native execution.

The project should remain narrow enough to finish. Detection, segmentation, video, edge hardware, vision-language models, and frequency-domain fusion are not part of the initial mandatory scope.

---

## Current Status

Status: **Phase 4 complete / accepted**

At this point:

- the project identity and maturity progression have been defined;
- the detailed project specification has been drafted;
- the AI-native tutor/build/check workflow has been defined;
- Codex working rules have been tailored to VisionLab;
- Git has been initialized for local version-control discipline;
- baseline governance documents now exist under `docs/`;
- a minimal Python smoke path exists for repository wiring only;
- T1 concept notes, environment probes, tiny convolution/image exercises, and dataset-candidate notes exist;
- Python 3.14.5 with an ignored local `.venv` and CPU PyTorch wheels has been verified for tiny tensor/convolution probes;
- CIFAR-10 has been downloaded locally into ignored `data/` and registered as the provisional core development dataset;
- training-run and phase approval boundaries have been established;
- the applied-domain decision has intentionally been deferred;
- Phase 1A dataset-contract code and committed tiny-fixture validation have been accepted;
- Phase 1B generated ignored CIFAR-10 manifest summaries, class counts, and visual inspection grids that were accepted after builder visual review;
- Phase 2 added a compact custom PyTorch CNN, shape-safe forward path, concise intermediate-shape inspection, parameter counting, and CPU forward/loss tests;
- Phase 3 added and closed a bounded CPU training engine with synthetic tiny-data verification, validation loop, optional scheduler support and learning-rate history, checkpoint save/restore with compatibility checks, minimal reproducibility/environment metadata, and non-finite loss failure status;
- Phase 4A added baseline experiment plumbing, registered split loader construction, DataLoader reproducibility policy, minimal prediction/evaluation artifacts, machine-readable history/curve artifacts, and a tiny non-material smoke route;
- Phase 4A smoke metrics are pipeline evidence only and are not official VisionLab baseline performance results;
- Phase 4B ran the approved single-run custom CNN CPU baseline on registered CIFAR-10, selected the best checkpoint by validation loss, restored that checkpoint, and evaluated the official test split once;
- Phase 4B produced a single-run baseline result, with restored-best official test loss `1.024515` and test accuracy `0.635900`;
- the Phase 4 baseline is not a tuned best result, not a variance estimate, and not a calibration, robustness, OOD, or broader generalization claim;
- no inference surface, pretrained model, augmentation experiment, calibration, robustness, OOD, diagnostics, or applied-domain behavior should yet be assumed.

The next project step should be a separate Phase 5 concept briefing and implementation plan for Augmentation and Generalization Controls. Phase 5 should not begin until that briefing and plan are reviewed and approved.

This status section should be updated as phases close. Historical detail belongs in the builder journal and phase closeout documents rather than accumulating here.

Core project documents:

- [`docs/project_specs.md`](docs/project_specs.md) — project identity, requirements, phases, closure boundaries, testing strategy, and risks
- [`AGENTS.md`](AGENTS.md) — durable repository-specific instructions for Codex and other coding agents
- [`AI_native_builder_journal.md`](AI_native_builder_journal.md) — evolving context, learning, decisions, and phase-closeout trail

T0 bootstrap documents now also include:

- [`docs/risk_register.md`](docs/risk_register.md) - current project risks and controls
- [`docs/requirement_change_log.md`](docs/requirement_change_log.md) - approved material requirement changes
- [`docs/phase_catalog.md`](docs/phase_catalog.md) - phase status and closeout trail
- [`docs/vision_foundations.md`](docs/vision_foundations.md) - T1 image tensor, convolution, pooling, feature-map, and split-role notes
- [`docs/compute_feasibility.md`](docs/compute_feasibility.md) - T1 Python/PyTorch/device probe evidence
- [`docs/development_dataset_candidates.md`](docs/development_dataset_candidates.md) - T1 development-dataset comparison and provisional recommendation
- [`docs/dataset_contract.md`](docs/dataset_contract.md) - Phase 1A dataset contract and tiny-fixture validation shape
- [`docs/cifar10_phase1b_registration.md`](docs/cifar10_phase1b_registration.md) - Phase 1B CIFAR-10 registration, split policy, class counts, and visual-inspection findings
- [`docs/phase_closeouts/Phase_4A_baseline_experiment_plumbing_and_smoke_verification.md`](docs/phase_closeouts/Phase_4A_baseline_experiment_plumbing_and_smoke_verification.md) - Phase 4A plumbing, smoke verification, limitations, and proposed Phase 4B approval plan
- [`docs/phase_closeouts/Phase_4_custom_cnn_baseline_experiment.md`](docs/phase_closeouts/Phase_4_custom_cnn_baseline_experiment.md) - accepted Phase 4 closeout, material baseline result, artifact inventory, and limitations

Additional documents expected during implementation include:

- vision glossary;
- architecture notes and ADRs;
- dataset feasibility and inspection reports;
- evaluation rubric;
- phase closeouts;
- model and experiment reports;
- applied-domain decision record.

---

## Planned Core Workflow

```text
Dataset and Split Validation
  → Model and Configuration Identity
  → Custom CNN Training or Checkpoint Restore
  → In-Distribution Evaluation
  → Transfer-Learning Comparison
  → Calibration and Confidence Analysis
  → Degradation Robustness Sweep
  → OOD / Cross-Source Evaluation
  → Failure Analysis and Diagnostics
  → Experiment Artifact and Model Comparison
  → Bounded Inference
```

Applied capstone loop:

```text
Applied-Domain Feasibility and Selection
  → Controlled or Synthetic Training Data
  → Independent Real Evaluation Data
  → Domain-Gap Baseline
  → Failure Diagnosis
  → One Approved Intervention
  → Controlled Retraining
  → Before/After Re-Evaluation
  → Honest Technical Report
```

The central experimental objects are registered data, model, configuration, prediction, and evaluation artifacts—not an attractive notebook or a single headline metric.

---

## Planned Primary Outputs

### Experiment artifacts

Each material training run is expected to preserve:

- experiment and run identity;
- dataset and split version;
- preprocessing and augmentation configuration;
- model configuration and parameter count;
- pretrained-weight identity where applicable;
- seed and environment information;
- optimizer, scheduler, and training budget;
- learning curves and metrics;
- selected and terminal checkpoints;
- evaluation results;
- warnings, failures, and stop reason;
- artifact inventory.

### Evaluation reports

Completed evaluation should eventually include:

- aggregate and per-class metrics;
- confusion matrix;
- calibration metric and reliability diagram;
- confidence distributions;
- representative correct predictions;
- false positives and false negatives;
- high-confidence mistakes;
- degradation curves;
- OOD or cross-source deltas;
- diagnostic artifacts;
- limitations and data-quality notes.

### Applied capstone report

The final applied study should describe:

- selected domain and task;
- controlled/synthetic and real data sources;
- capture or generation protocol;
- source-aware split design;
- baseline domain gap;
- failure patterns;
- intervention rationale;
- before/after results;
- null or mixed findings;
- operational and ethical limitations.

---

## Applied Domain Is Intentionally Deferred

The final application domain has not been selected.

This is deliberate. Domain selection will occur after the core model and evaluation capabilities exist, at a dedicated implementation-stage feasibility gate.

Candidate domains should be assessed using:

- data access and licensing;
- visual and label coherence;
- real-data collection feasibility;
- privacy and safety;
- controlled or synthetic generation feasibility;
- group-level leakage risks;
- compute and storage requirements;
- inspectability of failures;
- classification-first minimum scope;
- fallback viability.

Illustrative possibilities include surface-condition classification, waste-material classification, produce category or visible-condition classification, controlled-versus-real object-state classification, or a carefully bounded media-forensics study. These examples are not preselected commitments.

The selected domain and fallback should be recorded in an approved ADR or domain decision report.

---

## AI-Native Development Workflow

VisionLab is being developed through a paired tutor/build/check workflow:

```text
Concept Briefing
  → Builder Questions and Boundary Review
  → Bounded Implementation Plan
  → Builder Approval
  → Implementation
  → Automated Verification
  → Visual and Manual Verification
  → Result Interpretation
  → Phase Check and Context Synchronization
  → Phase Closeout
```

Material training introduces an additional approval boundary:

```text
Training Pipeline Smoke Test
  → Training Plan and Compute Estimate
  → Builder Approval
  → GPU or Material Training Run
  → Artifact Inspection
  → Interpretation Review
```

Codex may act as tutor, implementation partner, reviewer, debugger, experiment-design assistant, and documentation partner. The human builder retains authority over scope, architecture, training runs, experiments, interpretations, the applied domain, and project closure.

The file [`AI_native_builder_journal.md`](AI_native_builder_journal.md) is intended to be a first-class project artifact.

The README explains what VisionLab is and its current status.

The builder journal explains how the project, decisions, and builder understanding evolve during implementation.

---

## Initial Triage Stages

Before the main implementation phases, the project begins with two bounded stages.

### T0 — Project Bootstrap and Baseline Capture

Planned focus:

- establish repository structure and operating documents;
- create environment and version-control boundaries;
- establish risk and requirement-change records;
- define data, checkpoint, output, and secret policies;
- provide an initial smoke path.

### T1 — Vision Foundations and Feasibility Triage

Planned focus:

- review image tensors, channels, normalization, convolution, pooling, and receptive fields;
- verify PyTorch and the intended CPU/GPU path;
- complete small tensor and image-loading exercises;
- visualize one convolution or feature map;
- compare low-friction development datasets;
- record a provisional development-dataset and compute decision.

Triage should reduce uncertainty before the project commits to its dataset contract or a material training run.

Status: **Accepted.**

T1 verified a local CPU PyTorch path in an ignored `.venv`, added dependency-light foundation helpers and tests, generated tiny synthetic convolution artifacts, and recommended CIFAR-10 as the provisional Phase 1 development dataset candidate. The builder accepted the T1 clarifications on 2026-08-14, with final builder visual review of the generated foundation artifacts recorded as the remaining manual review condition. No dataset has been registered and no material training has begun.

---

## Planned Implementation Progression

After triage, the current reference roadmap is:

1. **Dataset Contract and Visual Data Inspection**
2. **Custom CNN and Shape-Safe Forward Path**
3. **Reproducible Training Engine**
4. **Custom CNN Baseline Experiment**
5. **Augmentation and Generalization Controls**
6. **Transfer Learning and Fine-Tuning**
7. **Evaluation Harness and Calibration**
8. **Robustness and OOD Evaluation**
9. **Failure Analysis and Interpretability**
10. **Inference Surface and Core Stabilization**
11. **Applied-Domain Feasibility and Selection**
12. **Applied Data Pipeline and Real Evaluation Set**
13. **Domain-Gap Baseline and Diagnosis**
14. **Data-Centric Intervention and Re-Evaluation**
15. **Final Integration, Portfolio Polish, and Closure Review**

The roadmap is a controlled starting point, not an immutable schedule. A phase may be divided into `A/B/C` subphases when data access, artifact stabilization, training approval, visual inspection, compute, or verification creates a natural boundary.

Approved changes should be driven by learning, tests, data feasibility, training evidence, and phase checks.

---

## Maturity and Closure Boundaries

### Learning foundation

Expected after the custom CNN baseline:

- trustworthy development dataset;
- custom CNN;
- reliable trainer;
- baseline run;
- basic evaluation artifacts.

### Model-engineering MVP

Expected after transfer learning:

- custom and pretrained models;
- frozen and fine-tuned transfer-learning paths;
- reproducible comparison foundation.

### Intended core: strong MVP+

Expected after core stabilization:

- evaluation beyond accuracy;
- calibration;
- degradation and OOD analysis;
- failure analysis;
- interpretability diagnostics;
- bounded inference;
- stable, independently demonstrable core.

### Full target: applied capstone

Expected after the domain-transfer phases:

- evidence-based domain selection;
- controlled or synthetic training source;
- independent real evaluation;
- measured domain gap;
- one intervention;
- re-evaluation;
- final technical and repository closeout.

If no applied domain responsibly passes the feasibility gate, closure at the strong MVP+ boundary requires an explicit decision explaining why the capstone was not attempted.

---

## Planned Technical Direction

The initial implementation is expected to use:

- Python;
- PyTorch and Torchvision;
- NumPy;
- Pillow and/or OpenCV where justified;
- Scikit-learn for selected evaluation metrics;
- Matplotlib and possibly Seaborn for plots;
- Pytest for the deterministic test suite;
- Google Colab or a similar single-GPU environment for bounded training.

Exact dependencies, versions, model backbone, development dataset, experiment format, and commands should be chosen through implementation evidence rather than treated as settled by this planning README.

The project should prefer focused project code over a generic framework. Notebooks may support exploration and explanation but should not become the sole implementation of core behavior.

---

## Setup

The T0 package structure and smoke command are intentionally minimal. They verify repository wiring only and do not validate PyTorch, computer-vision dependencies, datasets, models, or training.

Verified T0 smoke command:

```powershell
$env:PYTHONPATH = "src"
python -m visionlab.smoke
```

Expected future setup information:

```text
# create and activate environment
# install dependencies
# verify PyTorch and device
# inspect or register development data
# run CPU smoke path
# run approved training configuration
# run evaluation
# run inference demo
```

Do not infer PyTorch or training commands from the T0 smoke path.

---

## Running Tests

The deterministic local test suite currently covers repository wiring, T1 foundation and environment-probe helpers, Phase 1 data contracts and splits, Phase 2 custom-CNN forward-contract behavior, and Phase 3 CPU training-engine behavior.

Verified deterministic local test command:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\test.ps1
```

`scripts/test.ps1` sets `PYTHONPATH=src` explicitly, uses `.\.venv\Scripts\python.exe` when present, and falls back to `python` before invoking `unittest`. Raw `python -m unittest discover -s tests` does not read the `pyproject.toml` pytest configuration and may use an interpreter without PyTorch, so it should not be treated as the canonical local command.

The intended default suite should be deterministic, CPU-compatible, and independent of full dataset downloads or GPU access.

Expected future command:

```text
pytest
```

Planned coverage includes:

- dataset manifests and split validation;
- preprocessing shapes and ranges;
- custom-model forward shapes;
- loss and metric correctness;
- frozen-parameter behavior;
- checkpoint compatibility and restoration;
- calibration logic;
- degradation reproducibility;
- prediction and evaluation artifacts;
- corrupt-image and invalid-input handling;
- inference preprocessing parity;
- tiny-dataset training and intentional overfit;
- applied baseline/intervention comparison when that track begins.

Full-dataset, GPU, external-download, and optional live checks should be marked separately and document their last verified environment.

---

## Data, Checkpoints, and Generated Artifacts

The repository should not casually commit:

- downloaded datasets;
- private or restricted images;
- model checkpoints;
- training runs;
- caches;
- generated evaluation galleries;
- local notebooks with secrets or personal paths;
- environment files containing credentials.

Large or external artifacts should have documented acquisition or regeneration instructions. Curated tiny fixtures and selected portfolio images may be committed when licensing, privacy, and repository-size boundaries permit.

Final policies should be reflected in `.gitignore`, repository documentation, and artifact tooling.

---

## Environment Variables

A `.env.example` file may be introduced if implementation requires environment configuration.

Possible future concerns include:

```text
VISIONLAB_DATA_DIR=
VISIONLAB_OUTPUT_DIR=
VISIONLAB_DEVICE=
VISIONLAB_NUM_WORKERS=
VISIONLAB_SEED=
```

The actual names and defaults should follow implemented configuration. Secrets should not be introduced unless a selected external service genuinely requires them.

---

## Planned Non-Goals

The first complete version should not expand by default into:

- mandatory object detection;
- semantic or instance segmentation;
- video analysis or tracking;
- live camera streaming;
- multiple unrelated application domains;
- vision-language models;
- image-generation products;
- automated hyperparameter search;
- large model ensembles;
- adversarial-robustness research;
- distributed or multi-GPU training;
- production model serving;
- polished web dashboard;
- edge-device deployment;
- conference-style research claims.

Frequency-domain analysis, detection, Blender simulation, edge optimization, and a lightweight UI remain possible later additions only if approved evidence shows that they strengthen the project without undermining closure.

---

## Final Portfolio Direction

When complete, VisionLab should demonstrate:

- computer-vision fundamentals;
- custom CNN implementation;
- reproducible PyTorch training;
- transfer learning and fine-tuning;
- trustworthy dataset and split design;
- evaluation beyond accuracy;
- calibration and confidence analysis;
- robustness and OOD evaluation;
- systematic failure analysis;
- cautious interpretability diagnostics;
- data-centric experimentation;
- controlled-to-real domain-transfer analysis;
- bounded applied inference;
- disciplined AI-native tutoring, implementation, verification, and closeout.

The intended final framing is:

> VisionLab is an AI-native computer-vision laboratory that builds vision models from foundational components, evaluates how architecture and data distribution shape their failures, and applies that methodology to a controlled-to-real domain-transfer study.
