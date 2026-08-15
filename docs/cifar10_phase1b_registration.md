# CIFAR-10 Phase 1B Registration and Inspection

Status: Phase 1B accepted.

## Scope

Phase 1B registers CIFAR-10 as the provisional core development dataset for VisionLab. It does not implement a model, training loop, augmentation experiment, or Phase 2 work.

## Source and Version Record

Primary source: University of Toronto CIFAR-10 page by Alex Krizhevsky.

Evidence recorded on 2026-08-14 from the University of Toronto CIFAR-10 page:

- CIFAR-10 and CIFAR-100 are labeled subsets of the 80 million tiny images dataset created by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton.
- CIFAR-10 has 60,000 32x32 color images in 10 mutually exclusive classes.
- The upstream dataset has 50,000 training images and 10,000 test images.
- The upstream test batch contains 1,000 randomly selected images per class.
- The upstream training batches contain the remaining 5,000 images per class across five 10,000-image batches.
- The Python archive MD5 listed by the source page is `c58f30108f718f92721af3b95e74349a`.
- The source page asks users to cite Alex Krizhevsky, "Learning Multiple Layers of Features from Tiny Images", 2009.

License and usage note:

- The original University of Toronto CIFAR-10 page requests citation but does not visibly state a license.
- UCI's current CIFAR-10 repository metadata lists license `CC BY 4.0`.
- VisionLab records this as a provenance limitation: license metadata differs by source route, and downstream reuse should cite both the original source page and any repository metadata used for licensing claims.

Torchvision route:

- Phase 1B uses `torchvision.datasets.CIFAR10` to acquire the dataset into ignored local `data/`.
- Torchvision documents the dataset root, train/test selector, transform hooks, and download option.

References:

- University of Toronto CIFAR-10 page: https://cave.cs.toronto.edu/kriz/cifar.html
- Torchvision CIFAR10 documentation: https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html
- UCI CIFAR-10 metadata: https://archive.ics.uci.edu/dataset/691/cifar+10
- PyTorch CIFAR-10 tutorial normalization convention: https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

## Class Mapping

Ordered class names:

| index | class |
| --- | --- |
| 0 | airplane |
| 1 | automobile |
| 2 | bird |
| 3 | cat |
| 4 | deer |
| 5 | dog |
| 6 | frog |
| 7 | horse |
| 8 | ship |
| 9 | truck |

The class order is part of the model-facing contract.

## Split Policy

Upstream partitions:

- `train`: 50,000 images, 5,000 per class.
- `test`: 10,000 images, 1,000 per class.

VisionLab derived splits:

- `train`: upstream train samples not selected for validation.
- `val`: deterministic stratified subset selected from upstream train only.
- `test`: upstream test partition, untouched by validation construction.

Validation policy:

- validation size: 5,000 images total;
- validation per class: 500;
- stratification: exact per-class selection from upstream train labels;
- seed: `20260814`;
- selection unit: upstream train source index;
- routine: `visionlab.data.splits.stratified_validation_indices`;
- stable IDs do not change when VisionLab split assignment changes.

## Stable Sample IDs

Sample IDs are derived from immutable upstream source partition and original source index:

```text
cifar10-{upstream_partition}-{upstream_index:05d}
```

Examples:

- `cifar10-train-00000`
- `cifar10-train-49999`
- `cifar10-test-00000`

The VisionLab split is recorded separately as `split` and may be `train`, `val`, or `test`.

## Deterministic Preprocessing Profile

Inspection preprocessing:

- image size: 32x32;
- color mode: RGB;
- input value range: `[0.0, 1.0]`;
- normalization mean: `(0.5, 0.5, 0.5)`;
- normalization std: `(0.5, 0.5, 0.5)`;
- augmentation: none;
- deterministic: true.

Rationale:

- The Phase 1B inspection profile uses channel-wise `0.5` mean/std to map `[0, 1]` inputs to approximately `[-1, 1]`.
- This mirrors the simple PyTorch CIFAR-10 tutorial convention.
- These values are not computed from test-set statistics.
- Future training phases may approve a separate train-only empirical normalization profile, but that would be a training/configuration decision, not a Phase 1B visual-inspection requirement.

Important limitation:

- The generated preprocessed PNG grids are display artifacts. They show the normalized tensor values mapped back to viewable RGB for visual sanity checking; they are not persisted model inputs.

## Generated Outputs

The Phase 1B script writes ignored generated outputs under:

`outputs/phase1b_cifar10_inspection/`

Expected files:

- `cifar10_phase1b_manifest_summary.json`
- `cifar10_phase1b_class_counts.json`
- `cifar10_phase1b_manifest_records.jsonl`
- `raw_train_grid.png`
- `raw_val_grid.png`
- `raw_test_grid.png`
- `preprocessed_train_grid.png`
- `preprocessed_val_grid.png`
- `preprocessed_test_grid.png`

These files are generated artifacts and are not committed by default.

Observed generated-output summary:

- full manifest records: 60,000 JSONL records;
- VisionLab train split: 45,000 records, 4,500 per class;
- VisionLab validation split: 5,000 records, 500 per class;
- VisionLab test split: 10,000 records, 1,000 per class;
- generated grids: six PNG files covering raw and preprocessed train/val/test samples.

## Visual Inspection Findings

Status: Codex and builder visual inspection complete. Builder accepted Phase 1B on 2026-08-14.

Codex inspected:

- `raw_train_grid.png`;
- `raw_val_grid.png`;
- `raw_test_grid.png`;
- `preprocessed_train_grid.png`;
- `preprocessed_val_grid.png`;
- `preprocessed_test_grid.png`.

Findings:

- Raw train, validation, and test grids are readable at the expected CIFAR-10 32x32 scale.
- All ten classes appear in each split grid.
- The grid labels align with the visible class themes at spot-check level.
- Several examples are visually ambiguous or small, especially animal classes; this is expected for CIFAR-10 and should inform later failure analysis.
- Preprocessed grids appear visually equivalent to raw grids because the inspection renderer maps the normalized `[-1, 1]` values back to display RGB.
- No corrupt or blank samples were observed in the inspected grids.

Accepted limitations:

- Visual review covered deterministic grids, not all 60,000 samples.
- Spot-check visual agreement is not proof of perfect label quality.
- The documented visual limitations are acceptable for the provisional development dataset.

## Known Group and Leakage Limitations

- CIFAR-10 does not expose subject, photographer, web source, capture-session, duplicate, or near-duplicate group identifiers through the standard torchvision interface.
- Phase 1B can preserve upstream partition and source index, but it cannot prove correlated samples are absent across splits.
- The validation split is selected only from upstream train, so the upstream test partition remains untouched.
- Group-aware split validation is therefore limited and should not be overstated in later results.
- CIFAR-10 is a web-image-derived benchmark; the broader Tiny Images lineage has known ethical and provenance concerns. VisionLab uses CIFAR-10 only as a provisional development dataset, not as an applied-domain selection.

## Commands

Phase 1B live registration command:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python scripts\register_cifar10_phase1b.py
```

Observed result:

```text
record_count: 60000
split_counts: train 45000, val 5000, test 10000
class_count_min_max_by_split: train 4500/4500, val 500/500, test 1000/1000
```

Deterministic default verification:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\test.ps1
```

Observed result:

```text
Ran 22 tests in 0.075s
OK
```
