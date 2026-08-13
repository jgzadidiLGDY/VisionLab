# VisionLab — Project Specification

## 1. Project Overview

### 1.1 Project Name

**VisionLab**

### 1.2 Project Classification

VisionLab is a **strong MVP+ to bounded-substantial, portfolio-grade computer-vision and AI-native learning project**.

It intentionally grows through four maturity levels:

> learning foundation → model-engineering MVP → evaluation-centered MVP+ → applied domain-transfer capstone

VisionLab is broader than a standard image-classification exercise because it combines:

- vision and convolution fundamentals;
- a custom CNN built with PyTorch primitives;
- reproducible model training and experiment tracking;
- transfer learning and controlled model comparison;
- calibration, robustness, interpretability, and failure analysis;
- out-of-distribution evaluation;
- data-centric intervention and re-evaluation;
- an applied synthetic-to-real or equivalent domain-transfer study;
- a bounded inference interface;
- disciplined AI-native tutoring, implementation, verification, and closeout.

It is not intended to be a production inspection system, forensic authority, medical device, autonomous perception stack, or novel computer-vision research framework.

### 1.3 Product Definition

VisionLab is a local computer-vision engineering laboratory for building, training, comparing, diagnosing, and applying image models.

The system should allow the builder to move from a small custom CNN to one pretrained vision backbone, evaluate both under consistent conditions, inspect their errors and confidence, test degradation and domain shift, and then apply the resulting methodology to one focused real-world visual domain.

The applied domain is intentionally **not selected in this specification**. It will be selected at an implementation-stage decision gate after the core training and evaluation capabilities exist and candidate domains can be assessed using evidence rather than speculation.

### 1.4 Core Project Thesis

VisionLab investigates one unifying question:

> How do model architecture and training-data distribution affect generalization, robustness, confidence, and failure behavior when a vision model moves from controlled training conditions toward less-controlled real-world images?

The project is also a demonstration of disciplined AI-native development. The builder uses Codex as tutor, implementation partner, reviewer, debugger, experiment-design assistant, and documentation partner while retaining authority over scope, architecture, experiments, interpretations, and project closure.

### 1.5 Project Lineage

VisionLab deliberately combines lessons from prior projects or project ideas:

- **TinyGPT:** learn important AI foundations by implementing a meaningful model pipeline rather than only calling a hosted model;
- **RustFromC:** pair concept review with bounded implementation, verification, phase checks, and closeouts;
- **InterAgents:** treat contracts, evaluation scenarios, failure behavior, and stabilization as first-class engineering work;
- **deepfake reference:** make OOD generalization, degradation robustness, and failure analysis central experimental concerns;
- **synthetic-to-real reference:** treat data generation and domain design as part of model engineering, followed by diagnosis, intervention, and re-evaluation.

---

## 2. AI-Native Development Model

### 2.1 Development Principle

AI-native development does not transfer project ownership or experimental judgment to Codex.

The builder remains responsible for:

- approving phase boundaries and implementation plans;
- learning enough vision and ML concepts to evaluate substantive decisions;
- selecting datasets and the eventual applied domain;
- approving training runs that consume material compute;
- examining actual data, predictions, errors, and plots;
- distinguishing observed results from plausible explanations;
- deciding whether proposed interventions are justified;
- approving requirement or architecture changes;
- deciding whether each maturity boundary has genuinely been reached.

Codex provides leverage by helping to:

- brief concepts before implementation;
- decompose broad phases into bounded subphases;
- inspect data and implementation risks;
- propose experiment controls;
- implement approved code;
- write tests and fixtures;
- analyze failures without fabricating conclusions;
- maintain documentation and phase artifacts;
- perform phase checks, stabilization reviews, and closeout reviews.

### 2.2 Paired Tutor/Build Workflow

Each substantive phase should normally follow:

```text
Concept briefing
    → builder questions and boundary approval
    → implementation plan
    → approval
    → bounded implementation
    → automated and visual verification
    → result interpretation
    → phase check
    → phase closeout
```

The concept briefing should be concise enough to function as tutoring, not a project review. It should explain the concepts necessary to understand and evaluate the phase.

### 2.3 Training-Run Approval Boundary

Training is an experimental action, not merely a code execution step.

Before any material GPU run, the phase should establish:

- the dataset version and split identity;
- the model and configuration;
- the seed or seed set;
- the metrics to record;
- the checkpoint policy;
- the estimated runtime and compute environment;
- the success, failure, and stop conditions;
- the artifacts that must be preserved.

The builder approves the run plan before execution. After training, the builder and Codex inspect actual artifacts before making claims or changing direction.

### 2.4 Phase Checks and Context Synchronization

Before moving to a new phase, a phase check should:

1. determine whether the completed work has the intended conceptual and technical shape;
2. compare implementation against the approved phase boundary;
3. inspect tests, outputs, visual artifacts, and known limitations;
4. identify requirement or architecture drift;
5. synchronize the builder and Codex on the project’s current state;
6. recommend close, repair, split, or limited follow-up.

### 2.5 Flexible Subphases

The numbered phases are planning boundaries, not rigid promises that each objective fits in one implementation turn.

A phase should be split into `A/B/C` subphases when:

- data feasibility and implementation should be separated;
- training and post-training evaluation require separate approvals;
- a new artifact contract must be stabilized before downstream use;
- verification reveals a bounded repair;
- compute or external access creates a natural pause;
- the remaining work is too broad to review safely as one change.

Splitting a phase is not project failure. Silent scope expansion is.

### 2.6 Human Authority

Codex may recommend changes, but it may not autonomously redefine:

- the fundamentals-to-applied progression;
- the custom-CNN learning requirement;
- the controlled-comparison requirement;
- the dataset and split integrity rules;
- the distinction between evaluation evidence and interpretation;
- the applied-domain selection gate;
- the intended core and capstone closure boundaries;
- major success criteria or non-goals.

---

## 3. Problem Statement

Image classifiers can appear successful while relying on narrow datasets, source artifacts, background correlations, or overconfident predictions. Aggregate test accuracy often hides class-specific failures, calibration problems, degradation sensitivity, and collapse under a different data source.

At the same time, many learning projects begin directly with a pretrained model and therefore do not develop a clear understanding of image tensors, convolution, receptive fields, training dynamics, or the relationship between model behavior and data design.

VisionLab addresses both problems by creating a governed learning and engineering progression that:

- begins with model and vision fundamentals;
- constructs a custom CNN and reliable trainer;
- introduces transfer learning only after the baseline is understood;
- compares models under controlled conditions;
- evaluates confidence, robustness, and domain shift;
- inspects qualitative and quantitative failures;
- tests a data-centric intervention;
- culminates in a focused applied domain-transfer study.

The project does not assume that a more complex model, stronger augmentation, frequency features, or synthetic data will improve results. Negative and mixed results are valid when the experiment is sound and the limitations are reported honestly.

---

## 4. Project Goals

### 4.1 Learning Goals

The builder should develop a working understanding of:

- image tensors, channels, color spaces, resizing, and normalization;
- convolution, kernels, stride, padding, pooling, and receptive fields;
- feature maps and hierarchical representation learning;
- classification loss and optimization;
- overfitting, regularization, augmentation, and generalization;
- train, validation, test, OOD, and real-world evaluation boundaries;
- transfer learning and fine-tuning;
- class-wise metrics, calibration, robustness, and error analysis;
- model-appropriate interpretability and diagnostic limits;
- synthetic data, domain randomization, and domain gaps;
- experiment design and evidence-based iteration.

### 4.2 Product Goals

VisionLab should:

1. train and evaluate a custom CNN;
2. train and evaluate one pretrained comparison model;
3. preserve reproducible experiment artifacts;
4. compare models under controlled conditions;
5. evaluate at least one explicit domain shift;
6. run a repeatable degradation robustness suite;
7. expose calibration and confidence behavior;
8. generate error galleries and model diagnostics;
9. support single-image and bounded batch inference;
10. complete one applied domain-transfer study;
11. test one evidence-supported intervention and re-evaluate;
12. report actual outcomes without precommitted improvement claims.

### 4.3 AI-Native Workflow Goals

The repository should visibly demonstrate:

- concept tutoring paired with implementation;
- approval of bounded plans;
- experiment-run approval;
- automated and visual verification;
- requirement evolution based on evidence;
- phase checks and context resynchronization;
- distinction between AI suggestions and verified findings;
- stabilization and final closeout review.

### 4.4 Portfolio Goals

The completed project should demonstrate:

- PyTorch vision-model development;
- data-pipeline engineering;
- reproducible experimentation;
- transfer learning;
- evaluation beyond accuracy;
- OOD and robustness analysis;
- interpretability with appropriate caveats;
- data-centric improvement;
- physical or independently collected data evaluation;
- disciplined AI-native execution.

---

## 5. Non-Goals

The intended VisionLab version is not:

- a broad survey of every computer-vision task;
- a production defect-inspection system;
- a medical diagnostic system;
- a forensic authority for detecting manipulated media;
- an autonomous driving or robotics perception stack;
- a real-time video analytics platform;
- a vision-language-model project;
- an image-generation product;
- a novel neural-architecture research project;
- a distributed or multi-GPU training framework;
- a benchmark-chasing exercise;
- a polished SaaS product;
- proof that a model “understands” an image;
- proof that synthetic data eliminates the real-data requirement.

Object detection, segmentation, video, mobile deployment, edge hardware, and frequency-domain fusion are deferred unless explicitly promoted through requirement governance.

---

## 6. Supported Use Cases

### 6.1 Core Laboratory Use Case

The builder selects a registered experiment and runs a controlled training or evaluation workflow.

The system:

1. validates configuration and dataset identity;
2. loads a registered split;
3. records environment and seed metadata;
4. constructs the selected model;
5. trains or restores a checkpoint;
6. records metrics and checkpoints;
7. evaluates on configured datasets;
8. produces structured metrics and visual artifacts;
9. writes a complete experiment artifact;
10. supports comparison with another compatible run.

### 6.2 Diagnostic Use Case

Given a completed model run, VisionLab can:

- compute aggregate and per-class metrics;
- render a confusion matrix;
- assess calibration;
- identify high-confidence errors;
- generate a representative error gallery;
- run degradation sweeps;
- run a registered OOD evaluation;
- generate supported spatial diagnostics;
- compare behavior with another model.

### 6.3 Applied Inference Use Case

Given one image or a bounded image batch, VisionLab can:

- validate input type and dimensions;
- apply the exact registered inference preprocessing;
- select a compatible checkpoint;
- return top prediction, class probabilities, and confidence;
- display uncertainty or unsupported-input warnings where configured;
- generate available diagnostics;
- preserve inference metadata;
- export a compact result artifact.

### 6.4 Applied Domain-Transfer Use Case

After the applied-domain gate is approved, VisionLab should support a study with:

- one narrow visual task;
- a controlled or synthetic training source;
- an independently collected or clearly distinct real evaluation source;
- a documented domain gap;
- a failure analysis;
- one approved data or adaptation intervention;
- before/after re-evaluation.

Classification is the default task boundary. Detection may replace it only if feasibility evidence shows that localization adds material value without endangering closure.

---

## 7. Applied-Domain Decision Gate

### 7.1 Deferred Decision

The applied domain will not be selected during specification drafting.

It will be selected after the core model-engineering and evaluation subsystems are usable. This prevents the project from committing early to a dataset that is inaccessible, poorly licensed, too expensive, too small, unsafe to collect, or mismatched with the intended learning objectives.

### 7.2 Candidate-Domain Requirements

An eligible domain should satisfy most of the following:

- visually coherent and narrowly defined classes;
- affordable and lawful data access;
- sufficient training examples or feasible synthetic generation;
- independently collectable or clearly distinct real evaluation images;
- consistent labels;
- manageable privacy and safety concerns;
- compatibility with available compute;
- visually inspectable failures;
- meaningful domain-shift or robustness questions;
- plausible completion within the remaining project budget.

### 7.3 Candidate Examples

Examples are illustrative, not preselected:

- clean versus visibly damaged material surfaces;
- intact versus scratched or corroded household components;
- waste-material classification;
- produce category or visible condition classification;
- controlled versus real-world object-state classification;
- a carefully bounded media-forensics classification study.

### 7.4 Required Decision Record

The selected domain should be approved through an ADR or domain decision report containing:

- candidate comparison;
- dataset access and license evidence;
- real-data collection feasibility;
- sample visual inspection;
- label definition;
- leakage risks;
- compute estimate;
- minimum viable task;
- fallback domain;
- builder decision.

### 7.5 Domain Independence Before the Gate

Core modules should avoid assumptions about a specific domain. Dataset registration, class mappings, model configuration, evaluation, degradation transforms, artifacts, and inference should remain reusable across compatible classification datasets.

Domain independence does not justify building a generic framework. Abstractions should be introduced only where the core and applied tracks demonstrably share behavior.

---

## 8. Primary Outputs and Artifacts

### 8.1 Experiment Artifact

Every material training run should preserve:

- experiment ID and name;
- timestamp and status;
- code or commit identity when available;
- environment summary;
- dataset and split identity;
- preprocessing and augmentation configuration;
- model configuration and parameter count;
- seed;
- optimizer and scheduler configuration;
- training budget;
- epoch or step metrics;
- selected checkpoint and selection rule;
- final evaluation metrics;
- warnings, failures, and early-stop reason;
- artifact paths and hashes where practical.

### 8.2 Evaluation Report

A completed evaluation should contain:

- dataset and split coverage;
- accuracy and balanced accuracy where appropriate;
- precision, recall, and F1;
- per-class metrics;
- confusion matrix;
- ROC-AUC or PR-AUC where meaningful;
- calibration metric and reliability diagram;
- confidence distribution;
- representative correct predictions;
- false positives and false negatives;
- highest-confidence errors;
- degradation results;
- OOD results;
- limitations and data-quality notes.

### 8.3 Model Comparison Report

The custom CNN and pretrained model should be compared using:

- the same registered data partitions;
- comparable preprocessing where technically appropriate;
- declared differences in input resolution or augmentation;
- model size;
- training cost and duration;
- in-distribution metrics;
- calibration;
- robustness curves;
- OOD performance;
- failure-pattern differences;
- inference latency on at least one common environment.

### 8.4 Applied Capstone Report

The applied study should preserve:

- domain and task definition;
- data-source inventory;
- real-data capture or acquisition protocol;
- synthetic or controlled data-generation description;
- source-aware split design;
- baseline results;
- measured domain gap;
- failure clusters;
- intervention rationale;
- before/after comparison;
- null or mixed findings;
- ethical, safety, and operational limitations.

### 8.5 Human-Readable Portfolio Artifacts

The final repository should provide:

- a concise public README;
- selected learning and architecture documentation;
- actual result tables and plots;
- representative failure examples;
- a short technical report;
- an AI-native builder journal;
- phase closeouts and requirement-change history;
- a reproducible demonstration path.

---

## 9. Core Architectural Principles

### 9.1 Data Identity Before Training

No material result should exist without a known dataset version and split identity.

The project should record enough metadata to determine what images, subjects or objects, sources, labels, and transformations contributed to a run.

### 9.2 Split Integrity

Train, validation, test, OOD, and real-world boundaries must be explicit.

Where images are correlated by subject, video, physical object, capture session, generator, or source, splitting should occur at the correlated-group level rather than at the individual-image level.

### 9.3 Training and Evaluation Separation

The test, OOD, and real-world evaluation sets must not drive ordinary epoch-by-epoch model selection.

Any use of real data for few-shot adaptation must preserve a separate untouched real evaluation subset.

### 9.4 Controlled Comparisons

Each experiment should identify:

- the hypothesis;
- the changed variable;
- the controlled variables;
- the metric and dataset;
- the expected diagnostic result;
- the possible null result.

Do not change model, split, augmentation, optimizer, input resolution, and training budget simultaneously and then attribute the outcome to one factor.

### 9.5 Evaluation Beyond Accuracy

Aggregate accuracy is necessary but insufficient. VisionLab should examine class-wise behavior, calibration, confidence, robustness, domain shift, latency, and qualitative failures.

### 9.6 Diagnostics Are Not Explanations of Understanding

Grad-CAM, saliency, feature maps, embeddings, and spectra are diagnostic evidence with known limitations. They must not be described as proof of causal reasoning or human-like understanding.

### 9.7 Negative Results Are Valid

A correct experiment may show that:

- transfer learning offers only modest improvement;
- strong augmentation hurts clean accuracy;
- calibration remains poor;
- frequency features learn source artifacts;
- domain randomization does not close the gap;
- a synthetic-to-real intervention has mixed effects.

The project succeeds when it measures and explains the result honestly.

### 9.8 Artifacts Before Claims

README results, technical conclusions, and resume bullets must be derived from preserved artifacts. No numeric improvement claim may be written before the experiment.

### 9.9 Working Over Premature Generality

Prefer a clear implementation for the selected experiments over a generic computer-vision framework. Reuse should follow observed repetition.

---

## 10. Logical Workflow

```text
START
  |
  v
ValidateConfigAndDataset
  |
  v
BuildModelAndRecordIdentity
  |
  +----------------------+
  |                      |
  v                      v
TrainOrRestore      DataQualityChecks
  |                      |
  +----------+-----------+
             |
             v
EvaluateInDistribution
             |
      +------+-------+
      |              |
      v              v
RobustnessSweep   OODEvaluation
      |              |
      +------+-------+
             |
             v
CalibrationAndFailureAnalysis
             |
             v
Diagnostics
             |
             v
WriteExperimentArtifact
             |
             v
CompareModelsOrApproveIntervention
             |
             v
            END
```

Applied capstone loop:

```text
SelectDomain
  → BuildOrAcquireControlledTrainingData
  → CollectIndependentRealEvaluationData
  → TrainBaseline
  → MeasureDomainGap
  → DiagnoseFailures
  → ApproveOneIntervention
  → Retrain
  → Re-evaluate
  → CloseOrRejectClaims
```

---

## 11. Core Domain Models

The exact implementation may use dataclasses, typed dictionaries, Pydantic models, or another lightweight typed mechanism. Schema complexity should grow only when downstream use justifies it.

### 11.1 `DatasetSpec`

Minimum fields:

- dataset ID and version;
- name and source;
- license or usage notes;
- task type;
- class mapping;
- root or resolver;
- group identity field where applicable;
- train, validation, test, OOD, and real split references;
- preprocessing profile;
- manifest hash;
- known limitations.

### 11.2 `SampleRecord`

Minimum fields where available:

- sample ID;
- source ID;
- relative path or storage reference;
- label;
- group, subject, object, video, or capture-session ID;
- synthetic/real designation;
- generation or capture metadata;
- split;
- checksum;
- quality flags.

### 11.3 `ModelSpec`

Minimum fields:

- model ID;
- model family;
- architecture configuration;
- input size;
- number of classes;
- pretrained-weight identity if used;
- frozen and trainable components;
- parameter count;
- normalization profile;
- output semantics.

### 11.4 `TrainingConfig`

Minimum fields:

- experiment ID;
- dataset ID;
- model ID;
- seed;
- batch size;
- optimizer;
- learning rate;
- scheduler;
- epoch or step budget;
- augmentation profile;
- regularization;
- checkpoint rule;
- early-stop rule;
- device and mixed-precision settings.

### 11.5 `TrainingRun`

Minimum fields:

- run status;
- configuration snapshot;
- start and end timestamps;
- environment metadata;
- per-epoch or per-step metrics;
- best checkpoint;
- terminal checkpoint;
- stop reason;
- warnings and failures;
- artifact inventory.

### 11.6 `PredictionRecord`

Minimum fields:

- sample ID;
- model/run ID;
- true label when known;
- predicted label;
- class probabilities or scores;
- confidence;
- correctness;
- inference timestamp;
- preprocessing profile;
- diagnostic references.

### 11.7 `EvaluationResult`

Minimum fields:

- run and dataset identity;
- evaluation split;
- sample count;
- aggregate metrics;
- per-class metrics;
- confusion matrix;
- calibration metrics;
- latency summary;
- excluded-sample count and reasons;
- warnings;
- artifact references.

### 11.8 `RobustnessResult`

Minimum fields:

- corruption or degradation type;
- severity level;
- deterministic transform configuration;
- baseline metric;
- degraded metric;
- delta;
- sample coverage;
- seed where stochastic;
- plot/table reference.

### 11.9 `FailureCase`

Minimum fields:

- sample and run identity;
- error type;
- true and predicted class;
- confidence;
- source/domain metadata;
- observed visual factors;
- hypothesis tags;
- diagnostic references;
- human review notes.

### 11.10 `DomainStudy`

Minimum fields:

- domain decision reference;
- task definition;
- controlled/synthetic source identity;
- real-source identity;
- baseline run;
- measured gap;
- failure clusters;
- intervention hypothesis;
- intervention configuration;
- comparison run;
- outcome;
- limitations.

---

## 12. Functional Requirements

### 12.1 Configuration and Run Control

- Validate required configuration before loading large data or training.
- Assign an experiment/run ID before material work begins.
- Support CPU smoke execution and GPU training where available.
- Make device selection explicit.
- Support deterministic seeds to the practical degree documented by PyTorch and the selected environment.
- Fail clearly on incompatible model, checkpoint, class map, or preprocessing profiles.

### 12.2 Data Intake and Inspection

- Register datasets through a manifest or equivalent identity mechanism.
- Validate readable images, labels, class counts, and split membership.
- Detect duplicate paths and, where feasible, duplicate or near-duplicate content.
- Validate group-level split boundaries where metadata permits.
- Produce class-distribution summaries.
- Produce sample grids for every important split.
- Record excluded or corrupt samples.

### 12.3 Preprocessing and Augmentation

- Separate training augmentation from deterministic evaluation preprocessing.
- Preserve the exact inference preprocessing profile.
- Support resizing, cropping, tensor conversion, and normalization.
- Introduce augmentation incrementally and inspect transformed samples visually.
- Allow seeded or recorded transformations for diagnostic reproduction where practical.

### 12.4 Custom CNN

- Implement a compact CNN using PyTorch layers rather than a pretrained vision model.
- Include convolution, activation, downsampling, and classification components.
- Expose intermediate shapes for tests or diagnostics.
- Compute and report parameter count.
- Support configurable class count and input profile within the bounded project needs.
- Avoid architecture complexity not justified by the learning objective.

### 12.5 Training Engine

- Support training and validation loops.
- Record loss, learning rate, and selected metrics.
- Support checkpoint save and restore.
- Support best-checkpoint selection using a validation metric.
- Support early stopping or bounded epoch budgets.
- Support gradient safety checks and non-finite loss handling.
- Support mixed precision when available and verified.
- Preserve failed-run artifacts where practical.

### 12.6 Transfer Learning

- Use one selected pretrained backbone.
- Support head-only training with frozen features.
- Support one controlled fine-tuning stage.
- Record pretrained-weight identity.
- Make trainable/frozen parameters inspectable.
- Compare against the custom CNN without implying that identical hyperparameters are always optimal for both.

### 12.7 Evaluation

- Compute appropriate aggregate and per-class metrics.
- Preserve prediction-level records for evaluation datasets.
- Produce confusion matrices.
- Support binary or multiclass classification without overgeneralizing the design.
- Identify class imbalance and use balanced metrics when needed.
- Avoid using test metrics for routine model selection.

### 12.8 Calibration and Uncertainty

- Compute at least one calibration metric such as expected calibration error or Brier score.
- Produce a reliability diagram.
- Compare confidence distributions for correct and incorrect predictions.
- Support a provisional low-confidence warning threshold.
- Clearly distinguish low confidence from reliable OOD detection.

### 12.9 Robustness Evaluation

- Provide deterministic or reproducibly seeded degradation transforms.
- Include at least JPEG compression, blur, rescaling, brightness/contrast shift, or noise, with a bounded final selection.
- Evaluate multiple severity levels.
- Plot model performance against severity.
- Preserve clean-data baselines on the same comparison.
- Apply identical degradation conditions to compared models.

### 12.10 OOD and Domain-Shift Evaluation

- Register OOD or cross-source data separately from the ordinary test set.
- Report performance deltas rather than only absolute results.
- Preserve source and group identities where available.
- Compare confidence and error patterns across domains.
- Avoid claiming general OOD detection unless specifically implemented and evaluated.

### 12.11 Failure Analysis

- Generate a reproducible error table.
- Select representative errors using declared criteria.
- Include high-confidence mistakes.
- Support human tagging of observed factors.
- Separate observations from hypotheses.
- Identify likely label errors or data-quality concerns without silently relabeling data.
- Preserve unresolved cases.

### 12.12 Interpretability and Diagnostics

- Provide Grad-CAM or another supported spatial diagnostic for compatible CNN layers.
- Validate output size, range, and overlay alignment.
- Generate diagnostics for correct and incorrect predictions.
- State method limitations in generated reports.
- If frequency-domain features are later promoted, use frequency-specific diagnostics and ablations rather than relying on Grad-CAM alone.
- Treat UMAP/t-SNE as exploratory visualization, not a quantitative domain-distance metric.

### 12.13 Model Comparison

- Compare the custom CNN and pretrained model using registered runs.
- Prevent comparison of incompatible class maps or evaluation splits.
- Include model size, training cost, latency, calibration, robustness, OOD, and failure behavior.
- Explain experimental differences that cannot be held constant.
- Report uncertainty due to training variance where multiple seeds are not feasible.

### 12.14 Inference Interface

- Support a CLI as the required interface.
- Support a lightweight local UI only after core closure if approved.
- Load a registered model/checkpoint/preprocessing bundle.
- Validate invalid, corrupt, oversized, and unsupported inputs.
- Support one image and a bounded batch.
- Return predictions, scores, warnings, and model identity.
- Avoid presenting confidence as guaranteed correctness.

### 12.15 Synthetic or Controlled Data Generation

After the applied-domain gate:

- generate or construct labeled controlled training examples;
- record generation parameters and source identity;
- include both narrow and randomized generation conditions where feasible;
- validate labels through sample overlays or equivalent checks;
- prevent generation metadata from leaking directly into model inputs;
- preserve reproducibility for a bounded sample.

Blender is a candidate tool, not a preselected requirement. Simpler controlled photography, compositing, procedural 2D generation, or another justified approach may better fit the chosen domain.

### 12.16 Real-World Evaluation Data

- Define a capture or acquisition protocol.
- Record physical object, subject, source, or capture-session identity when relevant.
- Keep correlated samples together during splitting.
- Use comparable clean/defective or class capture conditions to reduce shortcuts.
- Preserve an untouched real evaluation subset if real images are used for adaptation.
- Document the effective number of independent objects or subjects, not only image count.

### 12.17 Intervention and Re-Evaluation

- Diagnose the baseline domain gap before choosing an intervention.
- Approve one primary intervention.
- State the hypothesis and expected effect.
- Preserve the original baseline.
- Retrain under a controlled configuration.
- Re-run the same real-world evaluation.
- Report actual improvement, regression, or mixed result.
- Avoid causal claims unsupported by the experimental design.

---

## 13. Non-Functional Requirements

### 13.1 Reproducibility

- Configurations must be serializable.
- Dataset and split identity must be preserved.
- Random seeds must be recorded.
- Checkpoint and preprocessing compatibility must be validated.
- Metrics should be regenerable from preserved prediction records where practical.
- Large datasets and checkpoints may remain external, but retrieval or regeneration instructions must be documented.

### 13.2 Reliability

- Corrupt images must be classified and reported.
- Non-finite losses must stop or safely fail the run.
- Interrupted training should be resumable from approved checkpoints where feasible.
- Every material run should terminate with a status and artifact inventory.
- Evaluation should not silently skip samples.

### 13.3 Maintainability

- Separate data, models, training, evaluation, diagnostics, inference, and artifacts.
- Keep model code independent from the UI.
- Keep dataset-specific code behind narrow adapters or profiles.
- Avoid a plugin architecture unless multiple real implementations justify it.
- Version important schemas and configuration profiles when compatibility matters.

### 13.4 Testability

- Core logic must run on tiny synthetic fixtures.
- Tests must not require downloading full datasets.
- CPU smoke paths are required.
- Live or large-data tests should be explicitly marked.
- Visual outputs require programmatic checks plus selected human inspection.

### 13.5 Compute and Performance

Initial targets are provisional:

- CPU unit and smoke suite: practical for local development;
- full custom-CNN training: feasible on a single Colab/Kaggle-class GPU;
- transfer-learning experiments: feasible within bounded free-tier sessions;
- checkpoint frequency: sufficient for recovery without excessive storage;
- inference: responsive for one image on a common laptop CPU or available GPU;
- no uncontrolled hyperparameter search.

### 13.6 Privacy, Safety, and Ethical Boundaries

- Do not commit private or sensitive images without informed permission.
- Avoid biometric identity claims.
- Do not scrape restricted datasets or evade access controls.
- Record dataset licenses and attribution requirements.
- Do not describe research classifiers as reliable safety, medical, or forensic authorities.
- Avoid generating or distributing harmful manipulated-media examples beyond what a bounded study requires.
- Remove secrets and personal paths from artifacts.

---

## 14. Requirement Evolution Governance

### 14.1 Principle

Requirements are evolving hypotheses constrained by the project’s identity.

Evidence from data access, training behavior, visual inspection, tests, compute limits, and domain selection may justify change. Changes must be recorded at the appropriate level.

### 14.2 Change Levels

#### Level 1 — Clarification

Examples:

- metric definition;
- field naming;
- acceptance-test precision;
- documentation correction;
- fixture refinement.

Codex may incorporate these within an approved phase and record them in the closeout.

#### Level 2 — Phase-Level Adjustment

Examples:

- splitting a phase;
- changing augmentation details;
- revising a schema;
- replacing a library;
- adding a bounded validation;
- changing training budget after a smoke run.

Requires builder approval before substantive implementation.

#### Level 3 — Project-Level Change

Examples:

- removing the custom CNN;
- making detection mandatory;
- adding segmentation or video;
- changing the applied capstone away from domain transfer;
- promoting edge deployment into core scope;
- changing the intended closure boundary;
- adding a second unrelated application domain.

Requires a separate proposal and explicit builder approval.

### 14.3 Requirement Change Log

Maintain:

`docs/requirement_change_log.md`

Each material entry should contain:

- change ID and date;
- phase;
- original requirement;
- proposed revision;
- technical or experimental evidence;
- learning impact;
- scope, architecture, test, and compute impact;
- builder decision.

---

## 15. Project Phases

The phase catalog is intentionally detailed, but each phase remains eligible for approved subdivision.

## T0 — Project Bootstrap and Baseline Capture

### Purpose

Create the repository, operating documents, and a visible starting point before substantive implementation.

### Concept Briefing

- project identity and maturity progression;
- AI-native tutor/build workflow;
- experiment versus implementation work;
- repository and artifact boundaries;
- large-file and secret handling.

### Build

- repository skeleton;
- `AGENTS.md`;
- project specification;
- builder journal;
- phase-check and phase-briefing workflow references;
- environment baseline;
- initial README;
- `.gitignore`;
- risk register;
- requirement change log;
- initial smoke command.

### Exit Criteria

- repository and operating documents exist;
- intended scope and non-goals are visible;
- no dataset or applied domain is accidentally implied as final;
- secrets, datasets, checkpoints, runs, and local workflow files have explicit version-control policies;
- Phase T1 is approved.

## T1 — Vision Foundations and Feasibility Triage

### Purpose

Develop the minimum conceptual foundation needed to evaluate the first model and confirm the available compute/tooling path.

### Concept Briefing

- image representation;
- convolution, pooling, receptive fields, and feature maps;
- classification loss;
- train/validation/test roles;
- overfitting and generalization;
- GPU versus CPU workflow.

### Build

- small tensor/image exercises;
- environment and PyTorch device probe;
- tiny image-loading spike;
- one convolution/feature-map visualization;
- candidate development-dataset comparison;
- provisional core dataset decision;
- compute feasibility note.

### Exit Criteria

- builder can explain the basic CNN data path;
- PyTorch and intended compute route are verified;
- at least one low-friction development dataset is feasible;
- no material training has begun;
- Phase 1 boundary is approved.

## Phase 1 — Dataset Contract and Visual Data Inspection

### Purpose

Establish a trustworthy, inspectable dataset foundation before model training.

### Concept Briefing

- datasets and dataloaders;
- preprocessing versus augmentation;
- leakage and correlated samples;
- class imbalance;
- normalization and image inspection.

### Build

- dataset registry/spec;
- class mappings;
- train/validation/test manifests;
- dataset validation;
- class-distribution report;
- sample grids;
- corrupt-image handling;
- tiny fixture dataset for tests;
- initial data tests.

### Exit Criteria

- every core split is identifiable and inspectable;
- data shapes and labels are correct;
- sample grids have been visually reviewed;
- known leakage limits are documented;
- evaluation preprocessing is deterministic;
- Phase 2 training input is approved.

## Phase 2 — Custom CNN and Shape-Safe Forward Path

### Purpose

Implement the foundational vision model before building a full trainer.

### Concept Briefing

- convolutional blocks;
- activations and downsampling;
- flattening/global pooling;
- logits and softmax;
- parameter initialization;
- tensor-shape reasoning.

### Build

- custom CNN configuration;
- convolutional feature extractor;
- classification head;
- forward pass;
- parameter counting;
- intermediate-shape inspection;
- forward/loss smoke path;
- model shape and invalid-input tests.

### Exit Criteria

- the custom model runs on CPU;
- logits and loss have correct semantics;
- parameter count is documented;
- shape failures are clear;
- the builder can explain the model path;
- no pretrained backbone has replaced the learning objective.

### Milestone

**Learning model skeleton**

## Phase 3 — Reproducible Training Engine

### Purpose

Build and verify the reusable training infrastructure without yet optimizing headline performance.

### Concept Briefing

- optimization and gradient descent;
- batching and epochs;
- validation and checkpoint selection;
- learning-rate scheduling;
- early stopping;
- reproducibility limits.

### Build

- training loop;
- validation loop;
- optimizer and scheduler configuration;
- metric logger;
- checkpoint save/restore;
- non-finite loss handling;
- early-stop or bounded-run control;
- experiment directory and metadata;
- tiny-batch overfit test;
- interrupted/resumed run test where feasible.

### Exit Criteria

- a tiny sample can be deliberately overfit;
- checkpoints restore compatible state;
- metrics and learning rates are preserved;
- failure leaves an inspectable status;
- a full baseline run plan can be reviewed before execution.

## Phase 4 — Custom CNN Baseline Experiment

### Purpose

Train, evaluate, and interpret the first genuine baseline.

### Concept Briefing

- learning curves;
- underfitting and overfitting;
- generalization gap;
- baseline discipline;
- limits of single-run conclusions.

### Build

- approved training configuration;
- one bounded baseline run;
- best-checkpoint selection;
- train/validation curves;
- test evaluation;
- prediction records;
- baseline report;
- initial error sample review.

### Exit Criteria

- the run is reproducible from preserved configuration;
- actual metrics and curves exist;
- test data was not used for checkpoint selection;
- obvious pipeline failures are ruled out;
- strengths and limitations are documented without inflated claims.

### Milestone

**Learning foundation complete**

## Phase 5 — Augmentation and Generalization Controls

### Purpose

Improve the baseline through controlled data and regularization experiments.

### Concept Briefing

- augmentation invariances;
- regularization;
- data leakage through transformations;
- ablation design;
- training variance.

### Build

- visually verified augmentation profiles;
- minimal-augmentation control;
- one standard augmentation experiment;
- optional regularization comparison if justified;
- controlled result comparison;
- updated baseline decision.

### Exit Criteria

- transformations preserve label meaning;
- only declared variables changed;
- clean and augmented results are compared honestly;
- selected baseline configuration has a documented rationale.

## Phase 6 — Transfer Learning and Fine-Tuning

### Purpose

Introduce pretrained representations after the custom CNN foundation is understood.

### Concept Briefing

- pretrained features;
- frozen-backbone training;
- fine-tuning;
- normalization and weight compatibility;
- catastrophic forgetting and learning-rate differences.

### Build

- one approved pretrained backbone;
- head replacement;
- frozen-feature run;
- one controlled fine-tuning run;
- trainable-parameter inspection;
- compatible experiment artifacts;
- preliminary comparison with the custom CNN.

### Exit Criteria

- pretrained-weight identity is recorded;
- frozen and trainable parameters are correct;
- both training modes are reproducible;
- comparison limitations are explicit;
- one pretrained candidate is selected for downstream evaluation.

### Milestone

**Model-engineering MVP**

## Phase 7 — Evaluation Harness and Calibration

### Purpose

Replace headline accuracy with a structured, reusable evaluation system.

### Concept Briefing

- precision, recall, F1, ROC-AUC, and PR-AUC;
- macro, micro, weighted, and balanced metrics;
- calibration and reliability;
- threshold selection;
- confidence versus correctness.

### Build

- metric engine;
- per-class evaluation;
- confusion matrices;
- calibration metric;
- reliability diagrams;
- confidence distributions;
- prediction-record export;
- compatible model-comparison report;
- metric unit tests.

### Exit Criteria

- metrics are tested against small known examples;
- class-wise failures are visible;
- confidence behavior is inspectable;
- compared models use identical registered evaluation samples;
- model preference is supported by more than aggregate accuracy.

## Phase 8 — Robustness and OOD Evaluation

### Purpose

Measure how model behavior changes outside clean in-distribution conditions.

### Concept Briefing

- corruption robustness;
- distribution shift;
- OOD evaluation versus OOD detection;
- source shortcuts;
- performance deltas and robustness curves.

### Build

- degradation transform registry;
- bounded severity sweeps;
- clean-versus-degraded comparison;
- OOD or cross-source dataset registration;
- OOD performance and confidence report;
- custom-versus-pretrained robustness comparison;
- robustness tests and artifact checks.

### Exit Criteria

- degradation transforms have been visually reviewed;
- both models receive equivalent degraded inputs;
- OOD data is isolated from model selection;
- performance drops and confidence shifts are reported;
- no unsupported general OOD-detection claim is made.

## Phase 9 — Failure Analysis and Interpretability

### Purpose

Turn aggregate results into inspectable hypotheses about model behavior.

### Concept Briefing

- false-positive and false-negative analysis;
- high-confidence errors;
- saliency and Grad-CAM;
- feature-space visualization;
- correlation, attribution, and causal limits.

### Build

- error tables;
- representative and high-confidence error galleries;
- human review tags;
- Grad-CAM or compatible spatial diagnostics;
- correct-versus-incorrect diagnostic matrix;
- optional embedding exploration;
- failure hypothesis report;
- label/data-quality issue inventory.

### Exit Criteria

- selection criteria for examples are declared;
- observations and hypotheses are separated;
- diagnostic limitations are visible;
- at least several recurring or unresolved failure patterns are documented;
- no model change is prescribed without supporting evidence.

### Milestone

**Evaluation-centered MVP+**

## Phase 10 — Inference Surface and Core Stabilization

### Purpose

Package the core model/evaluation work behind a bounded user-facing path and determine whether the core project is stable enough for applied expansion.

### Concept Briefing

- training/inference preprocessing parity;
- model bundle compatibility;
- invalid-input handling;
- latency measurement;
- confidence communication.

### Build

- model bundle or registry;
- single-image CLI inference;
- bounded batch inference;
- prediction and warning output;
- optional diagnostic generation;
- CPU latency measurement;
- end-to-end smoke tests;
- core architecture and code review;
- core closeout report.

### Exit Criteria

- inference uses registered preprocessing;
- invalid inputs fail clearly;
- one reproducible demo path exists;
- core tests are green;
- major architecture duplication is addressed or documented;
- core is independently closable before applied-domain work begins.

## Phase 11 — Applied-Domain Feasibility and Selection

### Purpose

Select one applied domain using actual feasibility evidence.

### Concept Briefing

- domain gaps;
- real-world collection design;
- synthetic/controlled data choices;
- group-level leakage;
- task scoping and label reliability.

### Build

- candidate-domain comparison;
- access and license checks;
- small sample acquisitions or generation spikes;
- label and task definitions;
- real-data collection feasibility;
- compute/storage estimate;
- classification-versus-detection decision;
- fallback decision;
- approved domain ADR.

### Exit Criteria

- one domain and fallback are approved;
- sample data has been visually inspected;
- task and label semantics are stable enough to implement;
- effective independent-sample constraints are understood;
- minimum complete capstone is bounded.

## Phase 12 — Applied Data Pipeline and Real Evaluation Set

### Purpose

Build the controlled/synthetic training source and independent real evaluation boundary.

### Concept Briefing

- procedural or controlled data generation;
- automatic annotation and validation;
- domain randomization;
- capture protocols;
- shortcut risks.

### Build

- controlled or synthetic generator/acquisition path;
- generation manifests and parameters;
- narrow and randomized conditions where feasible;
- label validation overlays or sample checks;
- real-world capture/acquisition protocol;
- group-aware real manifests;
- untouched real evaluation split;
- source-comparison grids;
- data-quality and leakage audit.

### Exit Criteria

- controlled/synthetic examples are reproducible at bounded scale;
- real examples are independently sourced and identifiable;
- labels have been manually spot-checked;
- correlated groups do not cross protected boundaries;
- obvious source shortcuts are tested or documented;
- baseline training plan is approved.

## Phase 13 — Domain-Gap Baseline and Diagnosis

### Purpose

Measure how a controlled/synthetic-trained model behaves on real data before attempting adaptation.

### Concept Briefing

- sim-to-real or controlled-to-real transfer;
- source shift;
- effective sample size;
- representation diagnostics;
- uncertainty under small real datasets.

### Build

- synthetic/controlled-only baseline training;
- in-source evaluation;
- untouched real-world evaluation;
- measured performance and calibration gap;
- source-aware failure gallery;
- background/lighting/viewpoint checks where relevant;
- exploratory feature-space comparison;
- domain-gap diagnosis report.

### Exit Criteria

- the gap is measured on preserved samples;
- image count is not confused with independent-object count;
- failures are grouped by evidence rather than intuition alone;
- at least one intervention hypothesis is supported;
- no intervention has contaminated the untouched evaluation set.

## Phase 14 — Data-Centric Intervention and Re-Evaluation

### Purpose

Test whether one bounded intervention changes real-world performance.

### Concept Briefing

- domain randomization;
- few-shot adaptation;
- ablation and controlled re-evaluation;
- training variance;
- limits of causal interpretation.

### Build

- intervention proposal and approval;
- one primary intervention;
- controlled retraining;
- identical real evaluation rerun;
- before/after metrics and calibration;
- changed and persistent failure cases;
- null/mixed-result analysis;
- capstone technical report.

### Exit Criteria

- baseline and intervention runs are both preserved;
- changed variables are explicit;
- evaluation samples and metrics are compatible;
- results are reported regardless of direction;
- claims match the experimental strength.

### Milestone

**Applied domain-transfer capstone / bounded-substantial target**

## Phase 15 — Final Integration, Portfolio Polish, and Closure Review

### Purpose

Determine whether VisionLab is genuinely ready to close and present its technical and AI-native story accurately.

### Build

- full deterministic test suite;
- selected GPU/live checks documented;
- end-to-end demo verification;
- artifact and configuration audit;
- repository hygiene;
- concise README;
- architecture documentation;
- final model card or model report;
- short technical report;
- selected plots and failure examples;
- completed builder journal and phase catalog;
- comprehensive code and architecture review;
- closure decision and known limitations.

### Exit Criteria

- intended core requirements are satisfied;
- applied capstone is complete or explicitly closed at the approved fallback boundary;
- commands and documentation match actual code;
- no unsupported metrics or resume claims remain;
- large artifacts have reproducible retrieval/regeneration instructions;
- the repository is understandable from a fresh clone;
- the builder can explain the main model, data, experimental, and workflow decisions.

---

## 16. Closure Boundaries and Scope Fallbacks

### 16.1 Learning Foundation

Reached when Phases 1–4 are complete:

- trustworthy development dataset;
- custom CNN;
- reliable trainer;
- genuine baseline experiment;
- basic evaluation artifacts.

This is a completed learning project but not the intended final VisionLab target.

### 16.2 Model-Engineering MVP

Reached when Phase 6 is complete:

- custom CNN and pretrained model;
- frozen and fine-tuned transfer-learning paths;
- reproducible comparison foundation.

### 16.3 Intended Core: Strong MVP+

Reached when Phase 10 is complete:

- evaluation beyond accuracy;
- calibration;
- robustness and OOD analysis;
- failure analysis;
- interpretability;
- bounded inference surface;
- stabilized and demonstrable core.

This boundary is independently portfolio-worthy and should not be endangered by the applied expansion.

### 16.4 Full Target: Applied Capstone

Reached when Phases 11–15 are complete:

- evidence-based domain selection;
- controlled or synthetic data source;
- independent real evaluation;
- measured domain gap;
- one intervention;
- re-evaluation and honest report;
- final repository closure.

### 16.5 Approved Fallbacks

If implementation evidence requires descoping:

1. classification replaces detection;
2. controlled 2D generation or photography replaces Blender;
3. a cross-source public dataset replaces physical collection if collection is unsafe or infeasible;
4. a smaller but group-valid real evaluation set replaces a larger leaky set;
5. core MVP+ closes independently if no applied domain passes the feasibility gate;
6. a null intervention result remains a complete capstone result;
7. local CLI replaces a web UI;
8. CPU/GPU latency measurement replaces dedicated edge-hardware deployment.

Fallbacks preserve the generalization and data-centric thesis. They may not silently remove it.

---

## 17. Deferred Features

The following are deferred unless implementation evidence supports a separately approved promotion:

- object detection as a mandatory task;
- semantic or instance segmentation;
- video classification or tracking;
- real-time camera streaming;
- multi-object or multi-domain support;
- vision transformers as a second major comparison family;
- frequency-domain DCT fusion;
- self-supervised pretraining;
- active learning;
- automated hyperparameter optimization;
- ensemble models;
- sophisticated OOD detectors;
- adversarial robustness;
- generative augmentation using diffusion models;
- full Blender/Unity simulation if a simpler generator suffices;
- Jetson, mobile, or embedded deployment;
- INT8/TensorRT optimization;
- cloud serving, authentication, and multi-user support;
- polished dashboard;
- conference-style paper claims.

---

## 18. Testing and Verification Strategy

### 18.1 Unit Tests

Required for:

- class mappings and manifests;
- split validation;
- preprocessing shapes and ranges;
- model forward shapes;
- loss computation;
- parameter freezing;
- metric correctness;
- calibration bins;
- degradation transforms;
- artifact serialization;
- checkpoint compatibility.

### 18.2 Contract Tests

Required for:

- dataset specs;
- model and training configs;
- checkpoint/model/preprocessing bundles;
- prediction records;
- evaluation artifacts;
- comparison compatibility;
- domain-study artifacts.

### 18.3 Integration Tests

Required for:

- tiny dataset to training artifact;
- checkpoint restoration to evaluation;
- prediction records to metrics;
- robustness sweep;
- inference preprocessing parity;
- invalid-input handling;
- baseline/intervention comparison.

### 18.4 Golden and Scenario Paths

Maintain at least:

1. tiny-batch intentional overfit;
2. successful custom-CNN smoke training;
3. successful pretrained-model smoke path;
4. checkpoint incompatibility rejection;
5. corrupt-image handling;
6. invalid split/leakage fixture rejection;
7. clean and degraded evaluation comparison;
8. high-confidence error extraction;
9. single-image inference;
10. applied baseline and intervention comparison when that track begins.

### 18.5 Visual Verification

Human inspection is required for selected:

- raw sample grids;
- augmented sample grids;
- degradation grids;
- confusion matrices and reliability diagrams;
- high-confidence errors;
- Grad-CAM overlays;
- synthetic/controlled versus real examples;
- annotation overlays;
- before/after failure cases.

Visual inspection should use declared samples or reproducible selection criteria, not only favorable examples.

### 18.6 Training Verification

For material runs, verify:

- configuration snapshot;
- dataset/split identity;
- loss behavior;
- checkpoint selection;
- evaluation sample count;
- metric and prediction artifact presence;
- warnings and early-stop reason;
- absence of obvious leakage or preprocessing mismatch.

### 18.7 Optional Live/Heavy Tests

Tests requiring full datasets, GPU, external downloads, or optional services should be marked and excluded from the default deterministic suite. Their last verified environment and result should be documented.

---

## 19. Evaluation Rubric

### 19.1 Learning Quality

- builder can explain concepts implemented in the phase;
- concept briefing is accurate and appropriately scoped;
- implementation decisions reflect understanding rather than blind copying;
- learning gaps are recorded honestly.

### 19.2 Data Quality

- provenance and license awareness;
- label clarity;
- split integrity;
- group-aware leakage prevention;
- visual inspection;
- corrupt and excluded-sample accounting;
- realistic limitations.

### 19.3 Model Quality

- correct architecture and outputs;
- appropriate training behavior;
- reproducible configuration;
- parameter and checkpoint clarity;
- justified complexity.

### 19.4 Evaluation Quality

- metric correctness;
- class-wise analysis;
- calibration;
- robustness;
- OOD separation;
- representative failure analysis;
- uncertainty about training variance;
- honest negative results.

### 19.5 Product Quality

- understandable inference path;
- preprocessing parity;
- clear warnings;
- useful visual and structured outputs;
- bounded responsiveness;
- avoidance of false authority.

### 19.6 Engineering Quality

- maintainability;
- test coverage;
- failure handling;
- artifact completeness;
- configuration clarity;
- repository hygiene;
- fresh-clone reproducibility.

### 19.7 AI-Native Workflow Quality

- effective tutor/build pairing;
- plan and training-run approval;
- phase checks;
- documented requirement changes;
- verified versus rejected AI suggestions;
- context synchronization;
- final human evaluation authority.

---

## 20. Success Criteria

The intended VisionLab project is complete when:

1. a custom CNN has been implemented and trained;
2. a pretrained comparison model has been frozen-trained and fine-tuned;
3. datasets and splits have stable identities;
4. material experiments preserve configurations, metrics, and checkpoints;
5. test data is isolated from routine model selection;
6. aggregate and per-class metrics are correct and inspectable;
7. calibration and confidence behavior are evaluated;
8. at least one degradation robustness suite is complete;
9. at least one OOD or cross-source evaluation is complete;
10. representative and high-confidence failures are analyzed;
11. model-appropriate diagnostics are generated with stated limitations;
12. the custom and pretrained models are compared across accuracy, calibration, robustness, OOD behavior, cost, and latency;
13. a bounded single-image inference path works from a fresh setup;
14. the applied domain is selected through a documented feasibility gate;
15. controlled or synthetic training data and independent real evaluation data are established, or an explicitly approved equivalent domain-transfer design is used;
16. the baseline domain gap is measured;
17. one evidence-supported intervention is tested;
18. before/after results are reported regardless of outcome direction;
19. core tests are deterministic and heavy tests are clearly separated;
20. documentation matches the actual implementation;
21. no numeric claim predates its supporting artifact;
22. the repository documents the AI-native learning/build process;
23. a final review determines that the project is ready to close or records the remaining bounded work;
24. the builder can explain and evaluate the main model, data, experimental, and workflow choices.

If the applied-domain feasibility gate rejects every candidate, successful completion at the strong MVP+ core boundary is permitted only through an explicit closure decision explaining why the capstone was not responsibly attempted.

---

## 21. Recommended Repository Structure

```text
VisionLab/
├── AGENTS.md
├── AI_native_builder_journal.md
├── README.md
├── pyproject.toml
├── .gitignore
├── .env.example
├── configs/
│   ├── datasets/
│   ├── models/
│   ├── training/
│   └── evaluation/
├── docs/
│   ├── project_specs.md
│   ├── architecture.md
│   ├── vision_glossary.md
│   ├── evaluation_rubric.md
│   ├── risk_register.md
│   ├── requirement_change_log.md
│   ├── domain_decision.md
│   ├── phase_catalog.md
│   ├── adr/
│   ├── phase_closeouts/
│   └── reports/
├── src/
│   └── visionlab/
│       ├── data/
│       │   ├── manifests.py
│       │   ├── datasets.py
│       │   ├── validation.py
│       │   ├── transforms.py
│       │   └── inspection.py
│       ├── models/
│       │   ├── custom_cnn.py
│       │   ├── transfer.py
│       │   └── registry.py
│       ├── training/
│       │   ├── engine.py
│       │   ├── checkpoints.py
│       │   ├── schedules.py
│       │   └── reproducibility.py
│       ├── evaluation/
│       │   ├── metrics.py
│       │   ├── calibration.py
│       │   ├── robustness.py
│       │   ├── ood.py
│       │   ├── failures.py
│       │   └── comparison.py
│       ├── diagnostics/
│       │   ├── gradcam.py
│       │   ├── features.py
│       │   └── galleries.py
│       ├── domain/
│       │   ├── generation.py
│       │   ├── capture.py
│       │   └── study.py
│       ├── artifacts/
│       │   ├── schemas.py
│       │   ├── writer.py
│       │   └── reader.py
│       ├── inference/
│       │   ├── predictor.py
│       │   └── bundle.py
│       └── cli.py
├── tests/
│   ├── unit/
│   ├── contract/
│   ├── integration/
│   ├── golden/
│   └── fixtures/
├── notebooks/
│   └── exploration/
├── demo/
├── scripts/
└── outputs/                 # ignored generated artifacts
```

The final structure may be simplified during implementation. Empty architecture should not be created merely to match this tree.

---

## 22. Initial Risk Register

### 22.1 Scope Risk

**Risk:** VisionLab becomes a survey of classification, detection, segmentation, deepfakes, simulation, and edge deployment.

**Control:** classification-first core, one applied domain, explicit deferred list, project-level approval for major additions.

### 22.2 Phase Calibration Risk

**Risk:** broad phases hide multiple approvals, training boundaries, or artifact contracts.

**Control:** approved `A/B/C` subdivision and phase-check discipline.

### 22.3 Dataset Leakage Risk

**Risk:** correlated images cross splits and inflate performance.

**Control:** group-aware manifests, duplicate checks, source-aware review, protected real evaluation.

### 22.4 Shortcut Learning Risk

**Risk:** the model learns background, compression, source, or rendering artifacts rather than the intended class.

**Control:** source controls, comparable capture conditions, failure analysis, saliency diagnostics, cross-source evaluation, and ablations.

### 22.5 Compute Risk

**Risk:** training ambitions exceed free-tier GPU sessions.

**Control:** low-friction development dataset, smoke runs, bounded models, checkpointing, no uncontrolled search, approved budgets.

### 22.6 Experimental Integrity Risk

**Risk:** project success becomes tied to a desired improvement.

**Control:** engineering success separated from experimental direction; null results accepted; claims generated after artifacts.

### 22.7 Real-Data Sample Risk

**Risk:** many photographs of few physical objects are treated as many independent examples.

**Control:** record object/session identities and report effective independent sample count.

### 22.8 Interpretability Overclaim Risk

**Risk:** heatmaps are presented as proof of model reasoning.

**Control:** diagnostics terminology, method limitations, multiple evidence types, no causal claim without intervention.

### 22.9 AI-Native Workflow Risk

**Risk:** Codex implementation speed outpaces builder understanding and review.

**Control:** concept briefings, approval boundaries, phase checks, builder explanations, and documented accepted/modified/rejected recommendations.

### 22.10 Portfolio-Presentation Risk

**Risk:** the repository emphasizes impressive language rather than reproducible evidence.

**Control:** concise README, artifact-backed results, honest limitations, and separation of public summary from detailed process documentation.

---

## 23. Final Project Principle

VisionLab should not be judged only by whether it achieves high classification accuracy or presents attractive heatmaps.

It should be judged by whether it demonstrates a credible, inspectable progression in which:

- vision fundamentals are learned and implemented;
- data identity and split integrity precede training;
- models are compared under controlled conditions;
- confidence and failure behavior are visible;
- robustness and domain shift are measured;
- diagnostics are interpreted cautiously;
- data-generation and real-world gaps are treated as engineering problems;
- intervention follows evidence;
- negative results remain reportable;
- the builder’s ability to evaluate the system improves throughout the build.

> VisionLab develops a vision system while the AI-native workflow develops the builder’s capacity to understand, challenge, and improve it.
