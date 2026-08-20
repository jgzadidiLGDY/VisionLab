# Phase 5B - Material Augmentation Comparison

Status: Accepted and closed as Phase 5B.

Builder decision: Phase 5B implementation, material run, and phase-check review were accepted. The phase is closed with the preserved Phase 4B baseline retained as the reference baseline. The single approved augmented run is recorded as controlled comparison evidence only. No rerun, tuning pass, broader augmentation search, or later-phase work is included here.

## Purpose

Phase 5B tested one bounded augmentation hypothesis against the accepted Phase 4B baseline:

- keep the Phase 4B CIFAR-10 split, preprocessing, model, optimizer, seed, DataLoader policy, epoch budget, and checkpoint-selection rule fixed;
- change only the train-time augmentation profile to the Phase 5A candidate;
- restore the best checkpoint selected by minimum validation loss;
- evaluate the official test split once after checkpoint selection;
- preserve enough run evidence to compare the result directly against `phase4b-cifar10-custom-cnn-baseline-001`.

## Approved Configuration

- Run ID: `phase5b-cifar10-custom-cnn-augmentation-candidate-001`
- Comparison baseline: `phase4b-cifar10-custom-cnn-baseline-001`
- Dataset: Phase 1B registered CIFAR-10
- Split policy: `45,000` train / `5,000` validation / `10,000` official test
- Model: `CustomCNNConfig(input_channels=3, image_size=(32, 32), num_classes=10, feature_channels=(32, 64, 128), dropout=0.0)`
- Control augmentation profile: `phase4-control-no-augmentation` version `1.0`
- Candidate augmentation profile: `phase5a-candidate-horizontal-flip-random-crop` version `1.0`
- Validation/test preprocessing: deterministic Phase 4 preprocessing only
- Optimizer: Adam
- Learning rate: `0.001`
- Weight decay: `0.0`
- Scheduler: none
- Batch size: `128`
- Epoch budget: `10`
- Seed: `20260818`
- DataLoader policy: seeded train shuffle, unshuffled validation/test, `num_workers=0`, `drop_last=false`
- Checkpoint selection: minimum validation loss
- Official test usage: evaluate once only after restoring the selected best checkpoint

The approved material configuration was executed without changing the seed, hyperparameters, checkpoint rule, evaluation sequence, or any other experimental variable.

## Controlled Comparison Boundary

Changed experimental variable:

- train-time augmentation profile only: `phase5a-candidate-horizontal-flip-random-crop` version `1.0`

Controlled variables preserved from Phase 4B:

- CIFAR-10 dataset identity, split counts, and class order;
- deterministic validation/test preprocessing;
- custom CNN architecture and parameterization;
- Adam optimizer, learning rate `0.001`, weight decay `0.0`, and no scheduler;
- batch size `128`, epoch budget `10`, seed `20260818`, and DataLoader policy;
- checkpoint selection by minimum validation loss;
- restore-best-before-test evaluation sequence;
- official test evaluation count of exactly one.

## Dataset and Evaluation Sequence Verification

Material preflight passed before training began and is preserved under the Phase 5B run directory.

The run followed the approved sequence:

```text
train with candidate train-time augmentation
  -> select best checkpoint by minimum validation loss
  -> restore selected best checkpoint
  -> generate final validation artifacts
  -> evaluate official test split once
  -> generate official test artifacts
```

Best-checkpoint evidence:

- selected epoch: `10`
- selected checkpoint: `outputs/phase5b-cifar10-custom-cnn-augmentation-candidate-001/checkpoints/best.pt`
- terminal checkpoint: `outputs/phase5b-cifar10-custom-cnn-augmentation-candidate-001/checkpoints/terminal.pt`

For this run, the selected best checkpoint and terminal checkpoint both correspond to epoch 10 because validation loss improved through the final epoch.

## Results

Restored-best validation:

- validation loss: `1.055734`
- validation accuracy: `0.620600`

Official test evaluation:

- test loss: `1.056135`
- test accuracy: `0.630800`

Phase 4B reference result:

- reference run ID: `phase4b-cifar10-custom-cnn-baseline-001`
- reference test loss: `1.024515`
- reference test accuracy: `0.635900`

Observed deltas versus the fixed Phase 4B baseline:

- test loss delta: `+0.031620`
- test accuracy delta: `-0.005100`

## Observations

- The approved single augmented run finished successfully and preserved the full planned artifact set.
- The run contract and metadata confirm that the candidate augmentation profile was the only changed experimental variable.
- Relative to the preserved Phase 4B no-augmentation baseline, this single run produced higher test loss and lower test accuracy.
- Validation and official test preprocessing remained deterministic and augmentation-free.
- The official test split was evaluated once, after best-checkpoint restoration, and not used for model selection.

## Interpretation and Decision

- The observed regression in this single controlled run is sufficient to reject adoption of `phase5a-candidate-horizontal-flip-random-crop` version `1.0` as the new baseline at this time.
- The preserved Phase 4B no-augmentation run remains the reference baseline.
- This closeout does not claim that data augmentation generally hurts CIFAR-10 performance, or that all crop/flip policies would regress under other seeds, budgets, or models.
- The result is single-run evidence only. It is not a variance estimate, robustness result, or broader generalization claim.

## Artifact Inventory

Phase 5B material-run artifacts are preserved under ignored `outputs/phase5b-cifar10-custom-cnn-augmentation-candidate-001/`.

Required artifacts:

- `outputs/phase5b-cifar10-custom-cnn-augmentation-candidate-001/preflight_report.json`
- `outputs/phase5b-cifar10-custom-cnn-augmentation-candidate-001/run_contract.json`
- `outputs/phase5b-cifar10-custom-cnn-augmentation-candidate-001/metadata.json`
- `outputs/phase5b-cifar10-custom-cnn-augmentation-candidate-001/artifacts/augmentation_profile_registry.json`
- `outputs/phase5b-cifar10-custom-cnn-augmentation-candidate-001/artifacts/history.json`
- `outputs/phase5b-cifar10-custom-cnn-augmentation-candidate-001/artifacts/curve_data.csv`
- `outputs/phase5b-cifar10-custom-cnn-augmentation-candidate-001/checkpoints/best.pt`
- `outputs/phase5b-cifar10-custom-cnn-augmentation-candidate-001/checkpoints/terminal.pt`
- `outputs/phase5b-cifar10-custom-cnn-augmentation-candidate-001/artifacts/val_summary.json`
- `outputs/phase5b-cifar10-custom-cnn-augmentation-candidate-001/artifacts/val_predictions.csv`
- `outputs/phase5b-cifar10-custom-cnn-augmentation-candidate-001/artifacts/test_summary.json`
- `outputs/phase5b-cifar10-custom-cnn-augmentation-candidate-001/artifacts/test_predictions.csv`
- `outputs/phase5b-cifar10-custom-cnn-augmentation-candidate-001/phase5b_comparison_report.md`
- `outputs/phase5b-cifar10-custom-cnn-augmentation-candidate-001/phase5b_result.json`

Reference comparison artifacts remain preserved under ignored `outputs/phase4b-cifar10-custom-cnn-baseline-001/`, including:

- `outputs/phase4b-cifar10-custom-cnn-baseline-001/run_contract.json`
- `outputs/phase4b-cifar10-custom-cnn-baseline-001/metadata.json`
- `outputs/phase4b-cifar10-custom-cnn-baseline-001/artifacts/val_summary.json`
- `outputs/phase4b-cifar10-custom-cnn-baseline-001/artifacts/val_predictions.csv`
- `outputs/phase4b-cifar10-custom-cnn-baseline-001/artifacts/test_summary.json`
- `outputs/phase4b-cifar10-custom-cnn-baseline-001/artifacts/test_predictions.csv`
- `outputs/phase4b-cifar10-custom-cnn-baseline-001/baseline_report.md`

Phase 5A inspection artifacts remain supporting evidence only:

- `outputs/phase5a_augmentation_inspection/augmentation_profile_registry.json`
- `outputs/phase5a_augmentation_inspection/phase5a_candidate_augmentation_grid.png`
- `outputs/phase5a_augmentation_inspection/phase5a_visual_inspection_note.md`

## Verification

Phase 5B implementation/plumbing verification completed before the material run with the canonical deterministic suite passing at `54` tests.

Closeout verification after documentation updates:

```text
powershell -ExecutionPolicy Bypass -File scripts\test.ps1
```

Result: passed after closeout updates.

## Worktree Note

`git status` continues to show deleted root files:

- `phase_briefing.md`
- `phase_check.md`

These were pre-existing worktree changes unrelated to Phase 5B implementation, experiment execution, evaluation, or closeout. They remain untouched and excluded from the Phase 5B closeout scope.

## Conclusions

Phase 5B is closed and accepted.

The preserved outcome is:

- one approved single-run augmentation comparison was completed exactly as planned;
- the candidate augmentation profile was the only changed experimental variable;
- checkpoint selection used minimum validation loss;
- the selected best checkpoint was restored before official test evaluation;
- the official test split was evaluated once after checkpoint selection;
- the restored-best augmented run underperformed the preserved Phase 4B no-augmentation baseline on both test loss and test accuracy;
- `phase5a-candidate-horizontal-flip-random-crop` version `1.0` is not adopted as the new baseline based on the current single-run evidence;
- the Phase 4B baseline remains the reference baseline for later comparisons.

Phase 6 has not begun. Any future augmentation revisit would require a separate approved plan and should not reuse this closeout as proof of a broader augmentation conclusion.
