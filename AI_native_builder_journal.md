# AI-Native Builder Journal

VisionLab is being built through an incremental AI-native learning and development workflow.

The purpose of this file is to preserve a visible record of how the project, its requirements, its experiments, and the builder’s understanding of computer vision evolve over time.

The README explains what VisionLab is intended to become and reports its current public status.

This journal explains how VisionLab is being built by an AI-native builder: a development partnership in which the human builder retains learning, product, experiment, and architecture ownership; Codex serves as an integrated tutor, implementation partner, debugger, and reviewer; and ChatGPT supports specification, conceptual discussion, and broader review where useful.

---

## Current Journal State

Status: **T0 complete / ready for T1 planning**

The T0 bootstrap closeout has been accepted. No material training runs exist yet.

At this point, the main project artifacts are planning documents, governance records, a minimal package skeleton, and a local smoke path. The project identity, fundamentals-to-applied progression, AI-native workflow, requirements, phase boundaries, evaluation principles, and closure tiers have been drafted.

The applied domain remains intentionally undecided. It will be selected later through the implementation-stage feasibility gate rather than assumed at project start.

This journal should not imply that planned datasets, models, commands, experiments, checkpoints, results, or application behavior have already been implemented or validated.

Update this section as the project advances. Preserve historical detail in dated entries and phase closeouts rather than accumulating an outdated narrative here.

---

## 2026-08-13 - T0 Bootstrap Implementation

### Context

The builder approved T0 - Project Bootstrap and Baseline Capture, including `git init` and correction of the project specification path to `docs/project_specs.md`.

### AI Contribution

Codex inspected the repository, identified that the folder was not yet a Git repository, found that documentation referenced `docs/project_specs.md` while the specification lived at the repository root, proposed a bounded T0 plan, and implemented the approved bootstrap changes.

### Builder Review and Decision

The builder approved the T0 plan and requested that PyTorch and Python-version decisions remain deferred to T1. The observed Python 3.14.5 environment is recorded as a compatibility risk rather than resolved in T0.

### Evidence

- Git was initialized locally.
- The project specification now resides at `docs/project_specs.md`.
- T0 governance files exist at `docs/risk_register.md`, `docs/requirement_change_log.md`, and `docs/phase_catalog.md`.
- A minimal import smoke path exists under `src/visionlab/`.
- The T0 closeout draft is at `docs/phase_closeouts/T0_project_bootstrap_and_baseline_capture.md`.

### Project Impact

T0 establishes repository wiring and governance, not computer-vision behavior. Dataset selection, applied-domain selection, PyTorch installation, model implementation, training, evaluation, diagnostics, and inference remain out of scope until later approved phases.

### Next Boundary

T0 was accepted by the builder on 2026-08-13. T1 has not started and requires a separate concept briefing and approval boundary.

---

## Project Context

VisionLab is an AI-native computer-vision engineering laboratory.

The intended project will:

- establish trustworthy dataset and split identities;
- review and apply image and convolution fundamentals;
- implement and train a custom CNN using PyTorch primitives;
- build a reproducible training and checkpointing path;
- compare the custom model with one pretrained vision backbone;
- evaluate models beyond aggregate accuracy;
- measure calibration, confidence, degradation robustness, and domain shift;
- inspect representative and high-confidence failures;
- generate model-appropriate diagnostics with explicit limitations;
- support bounded image inference;
- select one applied domain through a feasibility decision;
- measure a controlled/synthetic-to-real or equivalent domain gap;
- test one evidence-supported intervention and re-evaluate it.

The project is intentionally bounded.

It is not intended to become a broad survey of classification, detection, segmentation, video, vision-language models, simulation, and edge deployment. It is also not intended to become a production medical, safety, inspection, or forensic authority.

The intended maturity progression is:

```text
Learning Foundation
  → Model-Engineering MVP
  → Evaluation-Centered MVP+
  → Applied Domain-Transfer Capstone
```

The strong MVP+ core should be independently closable before the applied capstone begins.

---

## Why This Is an AI-Native Builder Project

VisionLab is meaningful not only because of the intended software and experimental outputs, but also because of the learning and evaluation challenge.

The builder begins with general AI, Python, and PyTorch experience, but does not treat computer-vision engineering judgment as already complete. The project therefore requires progress in four connected areas:

- improving understanding of vision and model-training concepts;
- building a credible implementation path;
- learning to design and control experiments;
- improving the judgment needed to evaluate data, models, failures, and claims.

The core working idea is:

> The builder develops the vision system, while the system-building process develops the builder’s capacity to understand, challenge, and improve it.

This is different from using AI only as a code generator.

Codex should contribute through bounded roles:

- tutor before unfamiliar implementation;
- planner after repository inspection;
- implementer after approval;
- test and fixture author;
- debugging partner;
- experiment-design assistant;
- reviewer of data, configuration, and result integrity;
- documentation and closeout partner.

The human builder remains responsible for:

- understanding the important concepts;
- reviewing and approving substantive plans;
- controlling scope and phase boundaries;
- approving material training runs;
- examining actual images, plots, errors, and artifacts;
- separating observations from plausible explanations;
- selecting the applied domain;
- approving requirement changes;
- deciding whether phases and maturity boundaries are complete;
- determining the public portfolio narrative.

---

## Core Technical Question

VisionLab is organized around one unifying question:

> How do model architecture and training-data distribution affect generalization, robustness, confidence, and failure behavior when a vision model moves from controlled training conditions toward less-controlled real-world images?

This question connects the project’s major stages:

- the custom CNN establishes foundational architectural ownership;
- transfer learning introduces pretrained representations;
- controlled comparison examines architectural and training differences;
- calibration examines whether confidence matches correctness;
- degradation testing examines sensitivity to altered inputs;
- OOD testing examines cross-source generalization;
- failure analysis examines recurring model weaknesses;
- applied domain transfer examines the gap between controlled training data and real images;
- one intervention tests whether data or adaptation changes the observed gap.

The project does not assume that the custom model, pretrained model, stronger augmentation, synthetic data, domain randomization, or another intervention will produce the preferred result.

---

## Starting Assumptions

The initial direction is based on the following working assumptions:

- a custom CNN should precede transfer learning so the project develops real vision-model fundamentals;
- a low-friction development dataset can support the early learning and engineering phases;
- data identity and split integrity should be established before material training;
- test, OOD, and real-world data should remain outside routine model selection;
- training success must be judged through artifacts and evaluation, not process completion alone;
- aggregate accuracy is insufficient for a serious model assessment;
- calibration, degradation robustness, OOD behavior, and failure analysis can reveal weaknesses hidden by clean test accuracy;
- Grad-CAM and related methods are diagnostics, not proof of model understanding;
- negative or mixed experimental results are valid project outcomes;
- the applied domain should be selected through feasibility evidence during implementation;
- classification is the default applied-task boundary;
- controlled or synthetic data should be treated as a data-engineering hypothesis, not automatically realistic training data;
- one diagnosed intervention is more informative than several uncontrolled changes;
- the strong MVP+ core should be protected from applied-capstone risk;
- requirements and phase divisions may evolve when implementation evidence justifies change.

These are not conclusions. Triage, implementation, tests, visual inspection, and training evidence may confirm, refine, or reject them.

---

## Current Architectural Direction

The current reference laboratory workflow is:

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

The current applied-capstone loop is:

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

The central architectural principle is **data and experiment identity before claims**.

The system should keep clear boundaries among:

- dataset source and split identity;
- preprocessing and augmentation;
- model configuration and weights;
- training configuration and run state;
- prediction-level records;
- aggregate evaluation;
- visual observations;
- explanatory hypotheses;
- interventions;
- public conclusions.

The exact repository structure may evolve. The project should preserve focused responsibility boundaries without constructing empty architecture merely to match a proposed directory tree.

---

## Incremental AI-Native Workflow

The normal phase cadence is:

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

The concept briefing should function as tutoring. It should explain the concepts the builder needs to understand and evaluate the phase without becoming a substitute project review.

The phase check should determine whether the work has the intended conceptual and technical shape and should synchronize the builder and Codex before the next phase.

Broad phases may be divided into `A/B/C` subphases when:

- data feasibility should precede implementation;
- a schema or artifact contract should be stabilized first;
- a smoke path should precede material training;
- training and post-training interpretation require separate approvals;
- visual inspection creates a human-review boundary;
- compute or external access creates a natural pause;
- verification exposes a bounded repair;
- the phase is too broad to review safely as one change.

Phase splitting is a workflow refinement, not a failure.

---

## Material Training Workflow

Training is an experimental action, not merely a command.

The expected material-run sequence is:

```text
Pipeline and Tiny-Data Smoke Verification
  → Training Hypothesis and Configuration
  → Dataset / Split / Model Identity Review
  → Compute and Artifact Plan
  → Builder Approval
  → Material Training Run
  → Checkpoint and Metric Inspection
  → Evaluation and Failure Review
  → Interpretation Approval
```

Before a material run, the journal or associated phase artifact should identify:

- experiment purpose and hypothesis;
- dataset and split version;
- model and pretrained-weight identity, if any;
- preprocessing and augmentation profile;
- seed or seed set;
- optimizer, scheduler, and training budget;
- checkpoint and early-stop rules;
- metrics and prediction artifacts;
- expected environment and runtime;
- stop and failure conditions;
- outputs to preserve.

After the run, record:

- actual environment and duration;
- terminal status;
- selected checkpoint;
- learning-curve observations;
- evaluation sample coverage;
- warnings or anomalies;
- facts directly shown by artifacts;
- interpretations or hypotheses;
- decisions and follow-up.

Do not treat a falling training loss, successful process exit, or attractive graph as proof that an experiment succeeded.

---

## Initial Triage Plan

### T0 — Project Bootstrap and Baseline Capture

Purpose:

- establish the repository and operating documents;
- preserve a truthful starting state;
- define version-control boundaries for data, checkpoints, runs, outputs, private images, and secrets;
- establish initial risks and requirement governance;
- create a minimal environment or smoke path.

Expected artifacts may include:

- repository skeleton;
- `AGENTS.md`;
- `README.md`;
- `docs/project_specs.md`;
- this builder journal;
- `.gitignore` and environment guidance;
- initial risk register;
- requirement change log;
- initial phase catalog.

Status: **Complete**

### T1 — Vision Foundations and Feasibility Triage

Purpose:

- review image tensors, channels, normalization, convolution, pooling, receptive fields, and feature maps;
- verify the local and GPU execution path;
- complete small tensor and image-loading exercises;
- inspect one convolution or feature-map visualization;
- compare low-friction development-dataset candidates;
- establish provisional data and compute decisions.

Expected artifacts may include:

- vision glossary or concept notes;
- environment and device probe;
- tiny image-loading spike;
- feature-map visualization;
- candidate dataset comparison;
- provisional development-dataset decision;
- compute feasibility note;
- updated risks or requirements.

Status: **Not started**

---

## Reference Implementation Roadmap

The current project specification proposes these phases:

1. Dataset Contract and Visual Data Inspection
2. Custom CNN and Shape-Safe Forward Path
3. Reproducible Training Engine
4. Custom CNN Baseline Experiment
5. Augmentation and Generalization Controls
6. Transfer Learning and Fine-Tuning
7. Evaluation Harness and Calibration
8. Robustness and OOD Evaluation
9. Failure Analysis and Interpretability
10. Inference Surface and Core Stabilization
11. Applied-Domain Feasibility and Selection
12. Applied Data Pipeline and Real Evaluation Set
13. Domain-Gap Baseline and Diagnosis
14. Data-Centric Intervention and Re-Evaluation
15. Final Integration, Portfolio Polish, and Closure Review

This roadmap is a starting hypothesis.

It may change when learning, data access, tests, visual inspection, compute limits, training behavior, or phase checks expose better boundaries or sequencing.

The project should preserve its maturity progression even when phase numbering changes.

---

## Maturity Trail

### Learning Foundation

Expected evidence:

- trustworthy development-dataset contract;
- custom CNN;
- reliable trainer;
- genuine baseline experiment;
- basic evaluation artifacts;
- builder explanation of the data-to-logits path.

Status: **Not reached**

### Model-Engineering MVP

Expected evidence:

- custom and pretrained model paths;
- frozen-feature and controlled fine-tuning runs;
- compatible experiment artifacts;
- preliminary model comparison.

Status: **Not reached**

### Evaluation-Centered MVP+

Expected evidence:

- correct class-wise evaluation;
- calibration and confidence analysis;
- degradation robustness;
- OOD or cross-source testing;
- systematic failure analysis;
- model-appropriate diagnostics;
- bounded inference;
- core stabilization review.

Status: **Not reached**

### Applied Domain-Transfer Capstone

Expected evidence:

- approved domain decision;
- controlled or synthetic training source;
- independent real evaluation source;
- measured domain gap;
- diagnosed failure patterns;
- one approved intervention;
- before/after re-evaluation;
- honest capstone report.

Status: **Not reached**

---

## Requirement Evolution

Requirement changes are expected, but they should be explicit, evidence-based, and reviewable.

Potential reasons include:

- inaccessible or poorly licensed data;
- class or label ambiguity;
- group-level leakage risk;
- corrupt or insufficient samples;
- compute limits;
- training instability;
- metric or calibration flaws;
- comparison incompatibility;
- diagnostics that do not support the intended claim;
- an applied domain that fails feasibility review;
- a simpler implementation that better preserves the learning objective;
- phase boundaries that prove too broad.

Examples of healthy future changes might include:

- replacing a development dataset after a documented feasibility problem;
- splitting training infrastructure from the first material run;
- narrowing augmentation after visual inspection shows label distortion;
- replacing a pretrained backbone due to compute or preprocessing constraints;
- choosing controlled photography instead of Blender;
- keeping classification rather than promoting detection;
- replacing physical collection with a valid cross-source public evaluation when collection is unsafe or infeasible;
- closing at the strong MVP+ boundary if no applied domain responsibly passes the gate.

Approved substantive changes should be recorded in:

- `docs/requirement_change_log.md`

Major architecture or applied-domain decisions should also use an ADR or dedicated decision report where appropriate.

---

## What the Project Should Preserve

Even if features, datasets, and sequencing change, the project should preserve these principles unless deliberately reconsidered:

- custom CNN before transfer learning;
- data and split identity before material training;
- separation of training, validation, test, OOD, and real-world evaluation roles;
- group-aware leakage prevention;
- reproducible configurations and material-run artifacts;
- controlled model and intervention comparisons;
- evaluation beyond aggregate accuracy;
- calibration and confidence visibility;
- degradation and domain-shift measurement;
- representative failure analysis;
- cautious interpretation of diagnostics;
- valid negative and mixed results;
- training and domain-selection approval gates;
- an independently closable strong MVP+ core;
- human approval over substantive project changes;
- honest documentation of current state.

---

## Journal Entry Convention

The journal should remain useful rather than becoming a raw activity log.

Add entries for material moments such as:

- phase entry and closeout;
- important concept review;
- dataset or compute feasibility decision;
- material training approval and result;
- unexpected model or data behavior;
- requirement or architecture change;
- accepted, modified, or rejected AI recommendation;
- meaningful debugging or verification finding;
- applied-domain selection;
- maturity-boundary or project-closure decision.

A recommended entry format is:

```markdown
## YYYY-MM-DD — Phase/Decision Title

### Context
What question or phase boundary was active?

### Concept or Hypothesis
What did the builder need to understand or test?

### AI Contribution
What did Codex or ChatGPT propose, explain, implement, or review?

### Builder Review and Decision
What was accepted, modified, rejected, or left unresolved?

### Evidence
What code, tests, images, plots, metrics, artifacts, or manual checks support the result?

### Learning
What changed in the builder's understanding?

### Project Impact
What changed in requirements, architecture, sequencing, risk, or readiness?

### Next Boundary
What is approved next, and what remains out of scope?
```

Short entries may omit sections that add no value. Do not fabricate disagreement, learning, or AI error to make the journal look more dramatic.

---

## Experiment Entry Convention

For a material training or intervention run, record or link to:

- experiment/run ID;
- hypothesis;
- dataset and split identity;
- model and weight identity;
- configuration and seed;
- compute environment;
- approval point;
- expected and actual runtime;
- checkpoint rule and selected checkpoint;
- primary metrics;
- calibration, robustness, or OOD result when relevant;
- important failures or warnings;
- preserved artifact locations;
- factual observations;
- interpretation and uncertainty;
- builder decision.

The journal may summarize these items and link to a detailed experiment artifact rather than duplicating it.

Do not enter performance numbers that cannot be traced to a preserved artifact.

---

## Phase Closeout Trail

Phase closeout files should become the primary evidence of the project’s development history.

Each closeout should record, at a useful level:

- phase learning and engineering objectives;
- what was learned;
- what was built, trained, evaluated, or investigated;
- important files, configurations, runs, and artifacts;
- tests and manual or visual verification;
- observations and interpretations;
- assumptions confirmed, narrowed, or rejected;
- requirement or architecture changes;
- known limitations;
- readiness for the next phase.

Expected closeout trail:

- T0 — [Project Bootstrap and Baseline Capture](docs/phase_closeouts/T0_project_bootstrap_and_baseline_capture.md)
- T1 — to be added
- Phase 1 — to be added
- Phase 2 — to be added
- Phase 3 — to be added
- Phase 4 — to be added
- Phase 5 — to be added
- Phase 6 — to be added
- Phase 7 — to be added
- Phase 8 — to be added
- Phase 9 — to be added
- Phase 10 — to be added
- Phase 11 — to be added
- Phase 12 — to be added
- Phase 13 — to be added
- Phase 14 — to be added
- Phase 15 — to be added

If a phase is split, list and link each closeout explicitly. Do not hide meaningful subphase history behind one retroactive summary.

---

## AI Recommendation Trail

The journal should make material AI involvement reviewable without becoming a transcript archive.

For important recommendations, record whether the builder:

- accepted the recommendation;
- modified it;
- rejected it;
- deferred it pending evidence.

Useful examples include:

- dataset or backbone selection;
- proposed phase split;
- augmentation change;
- suspected leakage;
- explanation of a training anomaly;
- failure-cluster hypothesis;
- proposed intervention;
- applied-domain selection;
- closure or expansion recommendation.

The journal should not claim human verification merely because AI produced code or an explanation. Verification should identify tests, artifacts, visual review, or reasoning performed by the builder.

---

## Initial Evaluation Questions

As the project develops, the builder should improve the ability to evaluate questions such as:

- Are image shapes, channels, ranges, and normalization correct?
- Do augmentations preserve label meaning?
- Are train, validation, test, OOD, and real-world boundaries clean?
- Are correlated subjects, objects, videos, generators, or capture sessions kept together?
- Can the custom CNN’s feature path and parameter scale be explained?
- Does the training curve indicate underfitting, overfitting, instability, or a pipeline error?
- Is checkpoint selection based on the intended validation metric?
- Are model comparisons controlled and compatible?
- Which classes fail, and how do those failures differ between models?
- Does confidence track correctness?
- How quickly does performance degrade under blur, compression, rescaling, noise, or lighting shift?
- Does OOD performance collapse, and is the model confidently wrong?
- Are failure galleries representative rather than cherry-picked?
- Do Grad-CAM or other diagnostics reveal stable patterns, and what can they not establish?
- Is a suspected failure caused by model architecture, data distribution, labels, background shortcuts, or source artifacts?
- Does the selected applied domain have sufficient independent real examples?
- Does controlled or synthetic data resemble the relevant aspects of reality?
- Is the intervention supported by diagnosed evidence?
- Does the before/after result exceed likely training variation?
- Are public claims proportionate to the experiment’s strength?
- When should the project abstain from a conclusion, descope, or close at the strong MVP+ boundary?

The quality of these questions—and the evidence used to answer them—should improve throughout the project.

---

## How to Read the Project Trail

For a quick project view and current status:

- [README.md](README.md)

For current scope, requirements, phase definitions, and closure boundaries:

- [Project Specification](docs/project_specs.md)

For durable Codex working rules:

- [AGENTS.md](AGENTS.md)

For requirement changes:

- [Requirement Change Log](docs/requirement_change_log.md) once created

For the development history:

- read this journal;
- follow phase closeouts in order;
- inspect linked experiment artifacts for material results;
- consult ADRs for major architecture and domain decisions.

---

## Why This Matters

This repository is intended to preserve more than final code and attractive plots.

It should show evidence of a disciplined human-AI learning and development process in which:

- the vision problem is refined rather than assumed complete;
- concepts are reviewed before unfamiliar implementation;
- Codex expands implementation and analysis capacity;
- AI-generated work remains inspectable and contestable;
- data and split integrity precede performance claims;
- material compute is planned and approved;
- automated tests are paired with visual verification;
- experiments preserve configurations and artifacts;
- observations remain distinct from explanations;
- negative results remain visible;
- requirements evolve through evidence rather than uncontrolled drift;
- the applied domain is selected responsibly;
- the human builder retains ownership of learning, direction, quality, and final responsibility.

VisionLab will be successful not merely if it achieves strong accuracy or presents an impressive demo, but if the repository makes clear:

- how the system became credible;
- what experiments actually established;
- where the models failed;
- what remains uncertain;
- why important decisions were made;
- how the builder learned to evaluate and improve the work.

> The final model is one project artifact. The builder’s improved capacity to reason about vision systems is another.
