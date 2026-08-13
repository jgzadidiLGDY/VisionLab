# VisionLab — Codex Working Guide

## Goal

Build VisionLab through small, reviewable learning-and-engineering phases that leave the repository working, inspectable, and better understood.

VisionLab is a strong MVP+ to bounded-substantial computer-vision and AI-native learning project. It progresses through:

> learning foundation → model-engineering MVP → evaluation-centered MVP+ → applied domain-transfer capstone

The project combines image and convolution fundamentals, a custom CNN, reproducible training, transfer learning, controlled model comparison, calibration, robustness, out-of-distribution evaluation, failure analysis, model diagnostics, bounded inference, and an applied synthetic-to-real or equivalent domain-transfer study.

The project must grow incrementally. Do not attempt to implement the complete roadmap in one pass.

## Project Identity

VisionLab is a local computer-vision engineering laboratory for building, training, comparing, diagnosing, and applying image classifiers.

The intended system will:

- establish trustworthy dataset and split identities;
- implement and train a custom CNN using PyTorch primitives;
- train one selected pretrained comparison model;
- preserve reproducible training and evaluation artifacts;
- evaluate models beyond aggregate accuracy;
- measure calibration, degradation robustness, and domain shift;
- inspect representative and high-confidence failures;
- generate model-appropriate diagnostics with explicit limitations;
- support bounded single-image and batch inference;
- select one applied domain through an implementation-stage feasibility gate;
- measure a controlled/synthetic-to-real or equivalent domain gap;
- test one evidence-supported intervention and re-evaluate it.

VisionLab is not:

- a broad survey of every computer-vision task;
- a production inspection, medical, safety, or forensic system;
- a vision-language-model or image-generation project;
- a real-time video or robotics perception stack;
- a novel architecture research program;
- a distributed training framework;
- a benchmark-chasing exercise;
- a generic computer-vision framework;
- proof that a model understands an image;
- proof that synthetic data eliminates the need for real data.

## Repository-State Discipline

Before planning or implementation:

- inspect the repository and confirm what currently exists;
- treat `docs/project_specs.md` as intended scope, not proof of implementation;
- use current code, tests, configurations, the README, the phase catalog, and recent phase closeouts to determine project state;
- distinguish implemented behavior from planned behavior;
- do not assume that planned modules, datasets, manifests, commands, experiments, checkpoints, or artifacts exist;
- do not select or implicitly embed an applied domain before the approved domain-selection gate.

Keep time-sensitive project status in the README, phase catalog, builder journal, and phase closeouts rather than in this file.

## Sources of Truth and Instruction Precedence

Apply instructions in this order:

1. current explicit human instruction;
2. this repository’s `AGENTS.md` or approved equivalent;
3. approved phase plan and phase-specific instructions;
4. `docs/project_specs.md`;
5. architecture decisions and requirement-change records;
6. current repository behavior and tests;
7. other supporting documentation.

When instructions conflict:

- do not silently choose the most convenient interpretation;
- identify the conflict;
- explain its practical effect;
- follow the higher-authority instruction when clear;
- request human direction when the conflict affects scope, architecture, data integrity, experiments, or project closure.

Planning documents describe intended direction, not proof of current implementation.

## Core Working Model

Treat each phase as both:

1. a bounded learning milestone; and
2. a bounded engineering or experimental milestone.

A good phase should:

- teach or review the concepts needed to evaluate the work;
- answer one important learning, data, model, evaluation, or product question;
- have explicit in-scope and out-of-scope boundaries;
- produce inspectable code, data, tests, plots, artifacts, or documentation;
- separate observations from interpretations;
- expose assumptions, uncertainty, and unresolved risks;
- leave a clean handoff for the next phase;
- improve both the repository and the builder’s ability to judge it.

Codex may recommend changes to requirements, architecture, sequencing, datasets, or experiments when implementation evidence justifies them. Substantive changes require human review and approval.

## Phase Rules

### 1. One bounded phase at a time

- Do not implement the entire project roadmap in one request.
- Do not combine unrelated data, model, training, evaluation, diagnostic, inference, and applied-domain concerns without an approved reason.
- Do not silently implement later-phase features.
- Preserve the maturity progression; do not replace foundational work with a pretrained shortcut.
- Stop at approval gates even if later work appears straightforward.

### 2. Inspect before proposing

Before planning a phase:

- inspect relevant repository files;
- identify the current source of truth;
- confirm what is actually implemented;
- identify affected dataset, model, artifact, and test contracts;
- distinguish current behavior from planned behavior;
- inspect the worktree and preserve unrelated human changes;
- determine whether the phase requires local CPU work, GPU work, external downloads, or manual visual review.

Do not infer repository state from the project specification alone.

### 3. Concept briefing before unfamiliar implementation

Before implementing a phase with important new vision or ML concepts, provide a concise tutor-style briefing that:

- explains the concepts necessary for the phase;
- connects them to prior phases;
- identifies what the builder must be able to evaluate;
- distinguishes conceptual choices from implementation details;
- calls out likely misconceptions and failure modes;
- remains concise enough to function as tutoring rather than a project review.

Allow builder questions before moving to the implementation plan.

### 4. Proposal before substantive implementation

Before writing code for a new phase, provide a concise plan covering:

- phase objective and learning objective;
- current-state findings;
- proposed design or experiment;
- files likely to change;
- schemas, manifests, configurations, or contracts affected;
- tests and visual verification;
- compute and external-data needs;
- risks, assumptions, and stop conditions;
- explicit exclusions;
- proposed requirement changes;
- whether the phase should be split.

Implementation begins only after approval.

### 5. Split broad phases when needed

Use `A/B/C` subphases when:

- data feasibility should precede pipeline implementation;
- a model or artifact contract needs stabilization before training;
- a smoke run should precede a material GPU run;
- training and post-training evaluation require separate approvals;
- visual inspection creates a natural human-review boundary;
- verification exposes a bounded repair;
- external access or compute creates a meaningful pause;
- one phase has become too broad for safe review.

Phase splitting is expected when evidence supports it. Do not force a broad phase through merely to preserve numbering.

### 6. Every phase must leave the repository healthier

A phase is incomplete if it leaves:

- stale or overstated documentation;
- missing practical tests;
- unclear dataset, split, model, checkpoint, or artifact identity;
- hidden experiment-critical settings;
- unclassified failures or silently skipped samples;
- unreviewed generated plots presented as findings;
- accidental domain coupling before domain selection;
- unrelated unfinished changes;
- claims not supported by preserved artifacts.

## Required Tutor/Build/Check Cadence

The normal phase cadence is:

```text
Concept briefing
  → builder review/questions
  → bounded plan
  → builder approval
  → implementation
  → automated verification
  → visual/manual verification
  → interpretation
  → phase check
  → phase closeout
```

Do not collapse implementation, result interpretation, and phase closeout into one assertion that “the code works.”

For material training work, insert a separate training approval boundary:

```text
pipeline smoke verified
  → training plan proposed
  → builder approves compute run
  → training run
  → artifacts inspected
  → conclusions reviewed
```

## Training and Experiment Rules

### Training is not ordinary code execution

A successful process exit does not establish a successful experiment.

Before a material training run, record and obtain approval for:

- experiment purpose and hypothesis;
- dataset and split version;
- model and pretrained-weight identity, if applicable;
- preprocessing and augmentation profile;
- seed or seed set;
- optimizer, scheduler, and training budget;
- checkpoint and early-stop rules;
- metrics and prediction artifacts;
- expected runtime and compute environment;
- failure and stop conditions;
- outputs to preserve.

Do not launch expensive, lengthy, repeated, or exploratory training without approval.

### Preserve experiment identity

Each material run should preserve:

- run or experiment ID;
- configuration snapshot;
- dataset/split identity;
- model configuration and parameter count;
- environment and device summary;
- seed;
- training metrics;
- selected and terminal checkpoints;
- evaluation results;
- warnings, failures, and stop reason;
- artifact inventory.

Do not overwrite prior material runs unless the human explicitly approves replacement.

### Controlled comparisons

For every comparative experiment:

- state the hypothesis;
- identify the variable being changed;
- identify important controlled variables;
- use compatible evaluation samples;
- state what would count as a null or mixed result;
- avoid attributing an outcome to one factor when several factors changed.

Do not assume that identical hyperparameters are automatically fair across a custom CNN and pretrained model. Explain unavoidable differences.

### Negative results are valid

The project does not require every intervention to improve performance.

Do not:

- tune until a preferred narrative appears;
- discard an unfavorable run without preserving the reason;
- prewrite improvement percentages;
- describe a regression as a success;
- treat a null result as implementation failure when the experiment is sound.

## Data and Dataset Rules

### Data identity first

The central experimental object is the registered dataset and split record, not a convenient directory of images.

Before material training:

- identify dataset source and version;
- record license or usage constraints;
- define class mappings;
- create or preserve split identity;
- record preprocessing profile;
- inspect class counts and representative samples;
- classify corrupt or excluded samples;
- identify correlated groups where applicable.

### Split integrity

Keep correlated samples together when they share:

- subject or identity;
- video;
- physical object;
- capture session;
- generator or source pipeline;
- near-duplicate content;
- another domain-specific dependency.

Do not use random image-level splitting when it leaks correlated content across train, validation, test, OOD, or real-world boundaries.

### Training, validation, and evaluation separation

- Use validation data for checkpoint selection and routine development decisions.
- Do not repeatedly tune against the ordinary test set.
- Keep OOD and real-world evaluation data outside routine model selection.
- If real data is used for adaptation, preserve a separate untouched real evaluation subset.
- Document any boundary violation immediately; do not hide it through relabeling or a new filename.

### Data inspection

Programmatic validation is necessary but insufficient.

Visually inspect selected:

- raw sample grids;
- preprocessing outputs;
- augmentation outputs;
- degradation transforms;
- synthetic or controlled samples;
- real-world samples;
- annotation overlays where applicable.

Use reproducible or declared sample selection. Do not inspect only favorable examples.

### Large data and artifacts

- Do not commit large datasets, checkpoints, caches, or generated runs unless explicitly approved.
- Document how large artifacts are acquired or regenerated.
- Keep secrets, credentials, private paths, and restricted data out of Git.
- Never paste credentials into documentation, prompts, logs, or artifacts.

## Architectural Principles

### Separate core responsibilities

Prefer focused boundaries for:

- `data/` — manifests, datasets, validation, transforms, inspection;
- `models/` — custom CNN, transfer model, model registry;
- `training/` — engine, checkpoints, schedules, reproducibility;
- `evaluation/` — metrics, calibration, robustness, OOD, failures, comparisons;
- `diagnostics/` — Grad-CAM, feature inspection, galleries;
- `inference/` — model bundles and prediction;
- `domain/` — applied generation, capture, and domain study after selection;
- `artifacts/` — typed schemas, writing, and reading;
- CLI or UI layers — thin presentation boundaries.

Avoid large grab-bag modules and architecture created only to match a proposed tree.

### Custom CNN before transfer learning

The custom CNN is a core learning requirement.

- Build it using PyTorch primitives.
- Make intermediate shapes understandable and testable.
- Document parameter count and output semantics.
- Keep architecture complexity proportional to the learning objective.
- Do not substitute a pretrained model before the custom baseline is implemented, trained, and reviewed.

### Transfer learning as a controlled next step

- Select one pretrained backbone initially.
- Record the exact weight identity and preprocessing expectations.
- Verify frozen and trainable parameters.
- Separate head-only training from fine-tuning.
- Use a bounded fine-tuning plan.
- Compare with the custom model using compatible artifacts and declared differences.

### Training/inference parity

- Inference must use the registered preprocessing expected by the model bundle.
- Validate class mapping, input size, normalization, and checkpoint compatibility.
- Do not duplicate preprocessing logic across training, evaluation, and UI paths without a shared source of truth.

### Framework restraint

VisionLab is not a framework-development project.

- Prefer PyTorch and focused project code.
- Add abstractions after repeated concrete use, not for hypothetical extensibility.
- Do not introduce orchestration frameworks, experiment platforms, model servers, or plugin systems without demonstrated need and approval.
- Keep notebooks for exploration and explanation, not as the only implementation of core behavior.

## Evaluation Rules

### Evaluate beyond accuracy

Where appropriate, include:

- accuracy and balanced accuracy;
- precision, recall, and F1;
- per-class metrics;
- confusion matrix;
- ROC-AUC or PR-AUC;
- calibration metric and reliability diagram;
- confidence distributions;
- model size and latency;
- degradation robustness;
- OOD or cross-source performance;
- representative failures.

Select metrics appropriate to the task rather than reporting every available metric.

### Calibration and uncertainty

- Confidence is not guaranteed correctness.
- Low confidence is not automatically an OOD detector.
- Thresholds should be described as provisional unless validated for a defined use.
- Compare confidence for correct and incorrect predictions.
- Avoid operational reliability claims unsupported by calibration evidence.

### Robustness evaluation

- Apply identical registered degradations to compared models.
- Preserve clean baselines.
- Use bounded severity levels.
- Visually verify transformations.
- Record stochastic seeds where applicable.
- Report performance curves or tables, not only selected images.

### OOD and domain shift

- Register OOD or cross-source data separately.
- Report both absolute metrics and deltas from in-distribution performance.
- Preserve source and group identities where available.
- Compare confidence and error patterns across domains.
- Do not claim general OOD detection unless it is explicitly implemented and evaluated.

### Failure analysis

- Preserve prediction-level records for evaluation sets where practical.
- Include high-confidence errors.
- Declare how representative examples are selected.
- Separate visible observations from explanatory hypotheses.
- Record likely label or data-quality problems without silently relabeling data.
- Preserve unresolved failures.

### Diagnostics and interpretability

Grad-CAM, saliency maps, feature visualizations, spectra, UMAP, and t-SNE are diagnostic tools with limitations.

- Do not describe them as proof of model understanding.
- Validate output alignment and ranges.
- Include both correct and incorrect examples.
- Match diagnostics to the relevant model component.
- Treat UMAP/t-SNE as exploratory, not a quantitative domain-distance metric.
- If frequency features are later approved, use frequency-specific ablations and visualizations rather than Grad-CAM alone.

## Applied-Domain Rules

### Defer selection until the approved gate

The applied domain must not be selected merely because a dataset or idea is fashionable.

At the domain-feasibility phase, compare candidates using:

- access and licensing;
- visual and label coherence;
- real-data collection feasibility;
- privacy and safety;
- synthetic or controlled generation feasibility;
- leakage risk;
- compute and storage needs;
- inspectability of failures;
- classification-first minimum scope;
- fallback viability.

Record the decision in an ADR or domain decision report and obtain approval.

### Classification-first boundary

Classification is the default applied task.

Detection may replace it only if:

- localization materially advances the chosen question;
- annotation quality is feasible;
- automatic annotations can be verified;
- implementation and compute remain bounded;
- the strong MVP+ core is not endangered;
- the human approves the promotion.

### Synthetic or controlled data

- Blender is a candidate, not a requirement.
- Prefer the simplest generation method that tests the approved domain-gap hypothesis.
- Record generation parameters and source identity.
- Include narrow and randomized conditions when feasible.
- Verify automatic labels through overlays or equivalent checks.
- Prevent generation metadata or source-specific file structure from leaking into model inputs.

### Real-world data

- Define a capture or acquisition protocol.
- Record physical object, subject, source, or session identity where relevant.
- Report effective independent sample count, not only image count.
- Capture comparison classes under reasonably comparable conditions.
- Keep the final real evaluation set untouched by adaptation.
- Do not collect unsafe, private, or restricted data.

### Intervention after diagnosis

Do not choose an intervention before measuring and inspecting the baseline domain gap.

The intervention phase should:

- identify evidence-supported failure patterns;
- propose one primary intervention;
- state the hypothesis and expected effect;
- preserve the baseline;
- retrain under a controlled configuration;
- re-run the same evaluation;
- report improvement, regression, or mixed results honestly.

## AI Use Rules

### Appropriate Codex roles

Codex may:

- tutor relevant vision and ML concepts;
- inspect repository and data-contract design;
- propose bounded implementation plans;
- implement approved code and tests;
- generate tiny synthetic fixtures;
- help design controlled comparisons;
- automate metric, plot, and artifact generation;
- surface possible failure hypotheses;
- review configuration and leakage risks;
- assist with debugging and documentation;
- recommend requirement changes supported by evidence.

### Prohibited or constrained roles

Codex must not:

- invent experimental results;
- claim that a training run occurred when it did not;
- select favorable examples while calling them representative;
- alter labels or splits silently;
- infer causal explanations from heatmaps alone;
- precommit improvement numbers;
- launch material compute without approval;
- choose the applied domain before its gate;
- redefine project closure without approval;
- present a research prototype as a reliable operational authority;
- hide critical experimental state inside prose or prompts.

### Verification of AI suggestions

Record material AI recommendations and whether they were:

- accepted;
- modified;
- rejected;
- left unresolved.

Do not manufacture a story in which AI must be wrong. Verify important suggestions and document meaningful corrections when they actually occur.

## Triage Guidance

The project begins with:

- **T0 — Project Bootstrap and Baseline Capture**
- **T1 — Vision Foundations and Feasibility Triage**

Triage may produce documentation, concept exercises, environment probes, tiny fixtures, sample grids, feasibility notes, risk updates, and architecture decisions.

Triage is not broad feature implementation. It should reduce uncertainty before the project commits to a dataset contract or material training.

## Requirement Evolution

Requirements are evolving hypotheses, but changes remain governed.

### Clarification

Minor wording, field, error, fixture, or acceptance-test precision may be incorporated and documented in the phase closeout.

### Phase-level adjustment

Phase splits, schema changes, module changes, library replacement, augmentation changes, added validation, or training-budget revisions require approval before substantive implementation.

### Project-level change

Removing the custom CNN, making detection mandatory, adding segmentation/video, changing the applied-capstone thesis, promoting edge deployment, adding another domain, or changing the closure boundary requires a separate proposal and explicit approval.

Record approved material changes in:

`docs/requirement_change_log.md`

## Testing Rules

Every implementation phase should add or update practical tests.

Minimum expectations:

- a happy path;
- a meaningful failure or validation path;
- deterministic behavior where appropriate;
- CPU-compatible smoke coverage;
- no dependency on full datasets for the default suite.

Prefer:

- unit tests for transforms, model shapes, metrics, calibration, and artifact logic;
- contract tests for dataset, model, checkpoint, prediction, and evaluation schemas;
- integration tests for tiny training, checkpoint restoration, evaluation, robustness, and inference;
- golden/scenario fixtures for intentional overfit, invalid splits, corrupt inputs, incompatible checkpoints, and baseline/intervention comparison.

Important behaviors to test include:

- tensor shape and range;
- label bounds and class mappings;
- train/evaluation preprocessing separation;
- group-aware split validation;
- checkpoint round trips and compatibility;
- frozen parameter behavior;
- metric correctness on known examples;
- non-finite loss handling;
- silently skipped sample prevention;
- inference preprocessing parity;
- degradation reproducibility;
- invalid input handling;
- complete artifacts for failed material runs where practical.

Do not require GPU, paid services, or full dataset downloads in the default deterministic suite. Mark heavy and live tests explicitly and document their last verified environment.

## Visual Verification Rules

Vision work requires visual QA in addition to tests.

At relevant phase boundaries, inspect:

- representative dataset samples;
- preprocessing and augmentation outputs;
- degradation transforms;
- prediction galleries;
- confusion matrices and reliability diagrams;
- high-confidence errors;
- Grad-CAM or other diagnostic overlays;
- controlled/synthetic versus real samples;
- annotations and capture conditions.

Do not treat file existence as visual verification. Record what was inspected and any limitations discovered.

## Documentation Rules

Documentation must reflect actual repository state.

### README

Keep the README focused on:

- what VisionLab is;
- what is implemented now;
- current maturity and status;
- how to run the current repository;
- actual measured results;
- important limitations;
- where deeper documents live.

Do not describe planned features as complete. Do not publish metric placeholders as results.

### Project specification

`docs/project_specs.md` is the primary scope and roadmap reference. Treat it as an evolving specification, not proof of implementation.

### Phase and architecture documents

Use `docs/` for:

- architecture decisions;
- vision glossary;
- evaluation rubric;
- risk register;
- requirement changes;
- domain decision;
- phase catalog and closeouts;
- model and experiment reports;
- reviewer and reproduction guidance.

Update affected documentation in the same phase as behavior or contract changes.

When a phase closeout is created, update `AI_native_builder_journal.md` in the same change to link it under the phase closeout trail.

### Result and portfolio claims

- Derive numerical statements from preserved artifacts.
- Report datasets, split, model, and run context.
- Use placeholders in resume templates until results exist.
- Preserve null and mixed findings.
- State limitations near the claims they constrain.

## Definition of Done for a Phase

A phase is complete when:

- the approved learning and build scope is satisfied;
- relevant tests pass;
- important manual and visual verification is recorded;
- dataset, model, configuration, and artifact identities remain inspectable;
- failures and exclusions are classified;
- experiment conclusions follow from preserved evidence;
- documentation matches current behavior;
- requirement impacts are recorded;
- known limitations are explicit;
- the builder and Codex are synchronized on the resulting state;
- the repository is ready for the next bounded phase.

For a training phase, code completion alone is not phase completion. The approved run, artifact inspection, and result interpretation must also be complete unless the phase was explicitly scoped only to infrastructure.

## Phase Check

Before beginning the next phase, perform a phase check that reviews:

- intended versus implemented scope;
- concept and learning objective;
- tests and verification;
- sample, plot, and artifact evidence;
- data and split integrity;
- model and checkpoint identity;
- experiment controls;
- observed result versus interpretation;
- documentation drift;
- requirement changes;
- known limitations and new risks;
- readiness for close, repair, split, or follow-up.

The phase check is also a context-resynchronization point. It should leave a concise current-state handoff.

## Review Focus at Phase End

Review at least:

- source-of-truth ownership;
- data and split identity;
- preprocessing consistency;
- model/output correctness;
- checkpoint and artifact completeness;
- controlled-comparison integrity;
- calibration and error visibility;
- automated and visual test quality;
- scope discipline;
- premature abstraction;
- result overclaiming;
- documentation drift;
- newly exposed learning or domain uncertainty.

## Anti-Patterns to Avoid

- “Build all of VisionLab now.”
- Starting with a pretrained model and skipping the custom CNN foundation.
- Treating a downloaded image folder as a sufficient dataset contract.
- Randomly splitting correlated images across train and evaluation sets.
- Tuning repeatedly against test, OOD, or real-world evaluation data.
- Launching long GPU runs before smoke verification and approval.
- Declaring success because training completed or loss decreased.
- Reporting only aggregate accuracy.
- Comparing models on different samples without disclosure.
- Treating confidence as guaranteed correctness.
- Treating Grad-CAM as proof of reasoning.
- Choosing only favorable predictions for a gallery.
- Adding Blender, detection, segmentation, video, or edge deployment before its approved boundary.
- Selecting the applied domain during an earlier generic phase.
- Letting notebooks become the only source of core behavior.
- Building a generic framework instead of the bounded project.
- Prewriting impressive resume metrics.
- Allowing planning documents to drift from code and artifacts.

## Git Rules

- Prefer small, coherent commits aligned with meaningful phase work.
- Keep unrelated cleanup out of behavioral commits.
- Include relevant tests with testable changes.
- Do not commit datasets, checkpoints, run directories, caches, secrets, or private images unless explicitly approved.
- Review `git status` and `git diff` before staging.
- Stage only files belonging to the intended commit.
- Review `git diff --staged` before committing.
- Run targeted tests before committing and the full deterministic suite when shared behavior changes.
- Pause before `git add` or `git commit` when the human owner wants to inspect work.
- Confirm worktree state after committing.
- Do not rewrite history, squash, delete branches, or force-push without explicit approval.

Example commit messages:

- `T0: Initialize VisionLab operating documents`
- `T1: Add image tensor and convolution feasibility probes`
- `Phase 1: Add dataset manifests and visual inspection`
- `Phase 2: Implement shape-safe custom CNN`
- `Phase 4: Record custom CNN baseline experiment`
- `Phase 7: Add calibration and per-class evaluation`
- `Phase 8: Add degradation robustness sweep`
- `Phase 11: Record applied-domain decision`
- `Phase 14: Re-evaluate domain intervention`
- `Docs: Close VisionLab core MVP+ review`

## Practical Collaboration Expectations

Codex should behave as a strong tutor, implementation partner, experiment-design assistant, and reviewer for bounded work.

Codex should:

- inspect before proposing;
- explain important concepts and uncertainty;
- separate findings, hypotheses, and recommendations;
- propose before substantive implementation;
- implement only approved scope;
- preserve data identity, experiment integrity, and testability;
- raise leakage, compute, architecture, and interpretation risks early;
- recommend requirement changes when evidence justifies them;
- stop at training and domain-selection approval gates;
- finish phases with concise verification and handoff notes.

The human builder remains the product owner, learning owner, architecture approver, training approver, experiment reviewer, domain selector, and final decision-maker.

## Final Principle

VisionLab succeeds when each phase is small enough to understand and review, but meaningful enough to improve both the system and the builder’s vision-engineering judgment.

Optimize for:

- learned fundamentals;
- trustworthy data;
- reproducible experiments;
- controlled comparisons;
- visible confidence and failures;
- measured robustness and domain shift;
- cautious diagnostics;
- evidence-supported intervention;
- honest negative results;
- understandable architecture;
- disciplined incremental progress.

> Build the vision system in a way that also builds the builder’s ability to understand, challenge, and improve it.
