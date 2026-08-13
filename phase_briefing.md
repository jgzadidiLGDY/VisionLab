# Skill: VisionLab Phase Briefing

## Purpose

Before preparing an implementation plan for a VisionLab phase, provide a concise review of the concepts, terms, technologies, and experimental ideas needed to understand that phase.

The objective is to give the builder and Codex a shared conceptual foundation before implementation choices are proposed.

Because VisionLab combines computer-vision fundamentals, model engineering, controlled experimentation, evaluation, and an applied capstone, the briefing should explain only the knowledge needed to review the named phase responsibly.

This briefing is:

- phase-bounded;
- concept-focused;
- concise;
- connected to VisionLab where useful;
- preparation for later plan review.

It is not:

- an implementation plan;
- a code or repository review;
- a broad computer-vision lesson;
- a feasibility study unless the phase calls for one;
- permission to modify files or run costly training.

---

## Phase Input

**Phase name:**  
`[PHASE_NAME]`

---

## Instructions

Inspect only enough of the repository to understand how the phase relates to VisionLab's current boundary, architecture, prior results, and roadmap.

Use the current project documents as the source of truth, especially where relevant:

- `docs/project_specs.md`
- the approved phase plan and prior phase closeouts
- experiment, dataset, and model documentation
- requirement or decision records
- `README.md`
- `AGENTS.md`
- `AI_native_builder_journal.md`

If the repository uses different paths, identify the current equivalents. Do not invent structures that have not been approved or implemented.

Then provide the following briefing.

## 1. Phase Focus

In one short paragraph, explain:

- what the phase is intended to establish or improve;
- why it matters to VisionLab's learning or product progression;
- which data, model, training, evaluation, diagnostic, or applied-domain boundary it primarily affects.

## 2. Key Concepts and Terms

Select approximately **three to five** concepts most important to understanding the phase.

For each, briefly explain:

- what it means;
- why it matters in this phase;
- how it relates to VisionLab;
- any nearby concept that should not be confused with it.

Possible concepts include:

- tensor shape, channel order, and receptive field;
- logits, probabilities, loss, and class prediction;
- preprocessing versus augmentation;
- train, validation, test, and out-of-distribution data;
- overfitting versus underfitting;
- optimization behavior versus generalization;
- checkpoint versus reproducible experiment artifact;
- transfer learning versus fine-tuning;
- accuracy versus class-sensitive metrics;
- discrimination versus calibration;
- confidence versus correctness;
- corruption robustness versus distribution shift;
- failure analysis versus interpretability;
- saliency evidence versus causal explanation;
- controlled comparison versus informal model comparison;
- synthetic data utility versus synthetic-to-real transfer;
- diagnosis versus intervention;
- technical feasibility versus applied-domain suitability.

Choose only concepts that materially affect the named phase.

## 3. VisionLab Mapping

Briefly identify where the concepts currently appear or are expected to appear in VisionLab.

Mention only the most relevant:

- specifications or decision records;
- dataset, sample, model, training, prediction, or evaluation contracts;
- preprocessing and augmentation pipelines;
- model or training components;
- configurations and experiment artifacts;
- evaluation or diagnostic outputs;
- fixtures and tests;
- phase findings and approval gates.

If the phase is not yet implemented, map to approved requirements and earlier evidence rather than assuming code structure.

## 4. Why These Concepts Affect the Plan

Briefly explain how misunderstanding the selected concepts could weaken the implementation plan.

Focus on practical consequences such as:

- data leakage or invalid splits;
- incompatible preprocessing between training and inference;
- shape or label-contract errors;
- uncontrolled comparisons;
- treating a training run as reproducible without preserving its identity;
- optimizing only for accuracy;
- confusing confidence with reliability;
- overstating robustness or interpretability;
- using synthetic results as proof of real-world validity;
- selecting the applied domain prematurely;
- hiding unresolved assumptions in code or configuration;
- requesting expensive training before the smoke path is verified.

## 5. What the Builder Should Watch For

Provide a checklist of approximately three to five items the builder should understand or verify when reviewing the later plan.

Focus on:

- the authoritative dataset and split contract;
- tensor, label, preprocessing, and inference invariants;
- the hypothesis and controlled comparison being proposed;
- metrics, baselines, and artifacts needed to evaluate the phase;
- reproducibility and training-run approval boundaries;
- expected failure, edge, or negative-result paths;
- whether the phase stays inside the approved scope;
- decisions that remain provisional or deferred.

---

## VisionLab Principles

Apply these principles where relevant:

### Data Identity Before Training

Dataset origin, version, classes, splits, transformations, and limitations must be explicit before results are trusted.

### Custom CNN Before Transfer Learning

The custom model establishes the learning foundation. Transfer learning is a controlled next step, not a substitute for understanding the baseline.

### Controlled Experiments

Each meaningful comparison should identify the hypothesis, changed variable, fixed conditions, seed policy, metrics, and preserved artifacts.

### Evaluation Beyond Accuracy

Class behavior, calibration, robustness, OOD performance, and failure cases matter where the phase reaches those concerns.

### Diagnostics Without Overclaiming

Visualizations and attribution methods can reveal model behavior, but they do not prove human-like understanding or causal reasoning.

### Negative Results Are Evidence

A failed intervention or weak transfer result remains useful when the setup, outcome, and interpretation are preserved honestly.

### Training Is an Approval Boundary

Do not imply permission for long, costly, or externally dependent runs. The plan should separate cheap verification from approved training.

### Applied Domain Remains Deferred

Until the designated selection phase, keep foundations domain-independent and treat candidate domains as provisional examples only.

---

## Guardrails

- Keep the briefing concise and phase-specific.
- Use approximately three to five key concepts.
- Do not prepare the implementation plan or propose exact code changes.
- Do not modify files, install dependencies, or start training.
- Do not perform broad architecture, dataset, or domain review.
- Do not introduce terminology merely because it is generally relevant to vision AI.
- Distinguish measured evidence, assumptions, interpretations, and unresolved uncertainty.
- Do not present provisional requirements or candidate domains as settled.
- Use repository references for orientation, not exhaustive analysis.

---

## Expected Tone and Length

Prefer:

- one short phase-focus paragraph;
- three to five concise concept explanations;
- a brief VisionLab mapping;
- a short planning-impact explanation;
- a three-to-five-item review checklist.

The goal is not comprehensive computer-vision mastery. It is to give the builder enough conceptual footing to review the phase plan, recognize hidden assumptions, and evaluate the later implementation responsibly.
