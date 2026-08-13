# Skill: VisionLab Phase Check

## Purpose

Run a lightweight end-of-phase review before VisionLab moves to the next phase.

Use this skill to:

1. confirm that the completed phase is in the intended shape; and
2. give the builder and Codex another chance to synchronize their understanding of its results, limitations, and implications.

This review is high-level and reusable across all VisionLab phases.

It is not:

- a broad repository audit;
- a full code review;
- a new implementation phase;
- a requirement-redesign exercise;
- permission to modify files or run new training;
- a substitute for builder review of important visual and experimental outputs.

---

## What to Inspect

Inspect only what is necessary to understand the completed phase and its relationship to the current roadmap.

Relevant materials may include:

- the phase briefing and approved plan;
- changed source and configuration files;
- dataset, sample, model, training, prediction, and evaluation contracts;
- preprocessing and augmentation logic;
- training and inference paths;
- tests and tiny fixtures;
- experiment metadata, checkpoints, metrics, and reports;
- confusion matrices, learning curves, failure galleries, or attribution outputs;
- decision and requirement-change records;
- phase documentation and closeout material;
- `README.md`, `AGENTS.md`, and `AI_native_builder_journal.md`;
- current working-tree status.

Do not expand the review into unrelated phases or rerun expensive experiments without approval.

---

## Review Goals

### 1. Phase Intent and Scope

Verify that:

- the implementation matches the approved objective and exit criteria;
- the phase did not silently absorb later work;
- exclusions and approval gates remain respected;
- material scope changes were recorded and approved;
- the repository is left coherent and usable.

Flag both under-delivery and unnecessary overreach.

### 2. Vision and ML Correctness

Check whether the phase applies its key concepts consistently.

Depending on the phase, this may include:

- image, tensor, label, and batch-shape contracts;
- data provenance and split integrity;
- preprocessing, augmentation, and inference parity;
- loss, logits, probabilities, predictions, and metrics;
- custom-CNN and transfer-learning boundaries;
- optimization versus generalization claims;
- accuracy, class behavior, calibration, robustness, and OOD semantics;
- synthetic-to-real limitations;
- failure analysis and interpretability boundaries.

The goal is to identify conceptual drift, not to reteach the phase.

### 3. Architecture and Responsibility Boundaries

Verify that responsibilities remain clear and appropriately simple.

Relevant questions include:

- Are data intake, transforms, models, training, evaluation, and inference separated where needed?
- Is preprocessing shared or made equivalent across training and inference?
- Are experiment identity and configuration preserved outside ad hoc notebook state?
- Are framework mechanics subordinate to VisionLab's contracts and semantics?
- Have temporary feasibility scripts leaked into core architecture?
- Are applied-domain assumptions isolated until the approved decision gate?

Prefer explicit working boundaries over premature generality.

### 4. Experimental Evidence and Claims

Check whether claims are supported by preserved evidence.

Verify where relevant:

- the dataset, splits, configuration, seed policy, environment, and checkpoint are identifiable;
- comparisons change only the intended variable or disclose confounders;
- metrics are calculated on the correct partition;
- visual outputs correspond to the claimed run and samples;
- weak, failed, or negative results are recorded honestly;
- diagnostic outputs are not overstated as causal explanations;
- synthetic or controlled results are not presented as real-world validation;
- result claims distinguish observation, interpretation, and limitation.

### 5. Tests and Verification

Check that verification is proportional to the phase.

Look for:

- CPU smoke coverage where feasible;
- tensor-shape, label, transform, and configuration tests;
- deterministic unit tests for metrics or data handling;
- integration tests when training, evaluation, artifact, or inference flow changed;
- representative happy, edge, and failure paths;
- tiny offline fixtures instead of dependence on large private data;
- explicit builder review of meaningful visual outputs;
- approved evidence for any long or costly training run.

List only meaningful missing checks.

### 6. Documentation and Context Alignment

Verify that the repository communicates its current state accurately.

Check whether:

- phase documentation matches what the repository demonstrates;
- experiment and result claims link to appropriate artifacts;
- requirements, terminology, setup, and artifact locations are current;
- approved changes appear in decision or change records;
- `README.md` reflects current maturity and limitations;
- `AGENTS.md` remains durable rather than becoming a status log;
- the builder journal and closeout trail capture important learning;
- stale or superseded decisions are clearly labeled.

Identify any material difference between what Codex reports as complete and what the repository actually supports.

### 7. Readiness for the Next Phase

Determine whether the phase provides a stable foundation for the next bounded step.

Check that:

- unresolved issues and limitations are explicit;
- blockers are distinguished from acceptable follow-ups;
- deferred decisions retain a named approval gate;
- no hidden data, compute, credential, or environment dependency blocks entry;
- the next phase follows logically from current evidence;
- the applied-domain choice remains deferred until its designated phase unless the specification is formally changed.

---

## Output Format

# Phase-Check Report

## 1. Overall Status

Choose one:

- **Ready**
- **Ready with small follow-ups**
- **Not ready**

Include a short evidence-based explanation.

## 2. Intended-Shape Assessment

State whether the phase:

- achieved its approved objective and exit criteria;
- stayed within scope;
- left the repository in a coherent state.

## 3. Key Findings

List only the most important findings. Separate them where useful into:

- vision or ML correctness;
- architecture;
- experimental evidence;
- tests and verification;
- documentation and context.

## 4. Builder–Codex Context Check

Summarize:

- what the phase now establishes;
- what remains provisional;
- what is explicitly deferred;
- any mismatch in assumptions, terminology, or claimed results.

## 5. Required Follow-Ups

List only items that should be addressed before the next phase, distinguishing:

- **Blocking**
- **Non-blocking**

If none, say so.

## 6. Next-Phase Readiness

State:

- whether the next phase may begin;
- any entry checkpoint, approval, or builder decision required first;
- the most important context that should carry forward.

## 7. Proposed Phase Closeout Note

Write a short closeout note capturing:

- the core outcome and evidence;
- major boundaries preserved;
- important limitations or deferred decisions;
- the recommended next bounded step.

---

## Constraints

- Do not implement fixes unless explicitly asked.
- Do not prepare the next implementation plan.
- Do not start new training or external-data acquisition.
- Do not conduct a broad redesign or reopen settled decisions without new evidence.
- Keep feedback proportional to the phase.
- Distinguish blocking issues from useful improvements.
- Prefer correctness, reproducibility, traceability, and conceptual clarity over style observations.
- Treat visual inspection as evidence that must be reviewed, not inferred from file existence.
- Do not require the applied-domain decision before its approved gate.

---

## Typical Invocation

Use this skill when asked:

- “Run a phase check.”
- “Is this phase in the intended shape?”
- “Check whether VisionLab is ready for the next phase.”
- “Review the phase and synchronize context.”
- “Do an end-of-phase check before closeout.”

---

## Final Principle

A VisionLab phase is ready when it is not merely implemented, but understood and supported by reviewable evidence.

The phase check should confirm that:

- the repository matches the approved phase;
- the builder and Codex share the same current context;
- experiment identity, assumptions, results, and limitations are visible;
- the next step rests on a stable and reproducible foundation.
