# VisionLab Dataset Contract

Status: Phase 1A accepted; CIFAR-10 is not registered yet.

The dataset contract records the minimum identity, split, label, validation, and preprocessing information VisionLab needs before model training. It is intentionally not a generic dataset framework.

## Contract Shape

The reusable contract is implemented in `src/visionlab/data/manifests.py`.

### Dataset identity

- `dataset_id`
- `version`
- `source`
- `license_or_usage`
- optional `description`

### Class mapping

- ordered class names;
- derived `class_to_idx` mapping;
- non-empty, unique class names.

The class order is part of the model-facing contract. Later checkpoints and predictions should preserve the same ordering.

### Split names

The manifest declares valid split names explicitly. Phase 1A fixtures use:

- `train`
- `val`
- `test`

Future manifests may add other declared splits when the phase scope requires them, such as OOD or real-world evaluation partitions.

### Preprocessing metadata

The preprocessing spec records deterministic input assumptions:

- `image_size` as width and height;
- `color_mode` as `RGB` or `L`;
- `value_range`;
- `normalization_mean`;
- `normalization_std`;
- `deterministic`.

Phase 1A does not implement augmentation. The contract preserves the distinction by recording deterministic preprocessing only.

### Sample records

Each sample records:

- `sample_id`
- `split`
- `label`
- `relative_path`
- optional `source_id`
- optional `group_id`
- optional `checksum`

`relative_path` must stay under the dataset root. `group_id` is optional because not every public dataset exposes correlated-group identity. If group identity is unavailable, that absence should be documented as a dataset limitation rather than silently ignored.

## Validation Behavior

Validation is implemented in `src/visionlab/data/validation.py`.

Phase 1A validation checks:

- duplicate sample IDs;
- undeclared splits;
- undeclared labels;
- unsafe relative paths;
- missing sample files;
- invalid tiny fixture image files;
- image size mismatch;
- color-mode mismatch;
- empty split warnings;
- class counts by split.

The committed tiny fixtures are ASCII Netpbm images (`P3` RGB). This keeps the default deterministic suite dependency-free. Phase 1B may use Pillow or torchvision for CIFAR-10 acquisition and image handling, but the reusable manifest shape should remain the same.

## Phase 1A Tiny Fixture Dataset

Fixture root:

`data/fixtures/phase1_tiny_rgb`

Samples:

| sample_id | split | label | relative_path |
| --- | --- | --- | --- |
| `train-red-000` | `train` | `red` | `train/red_block.ppm` |
| `train-green-000` | `train` | `green` | `train/green_block.ppm` |
| `val-red-000` | `val` | `red` | `val/red_block_val.ppm` |
| `test-green-000` | `test` | `green` | `test/green_block_test.ppm` |

The fixture is deliberately tiny. It validates contract behavior; it is not a training dataset, visual-data finding, or model benchmark.

## CIFAR-10 Readiness

The contract can represent CIFAR-10 in Phase 1B without embedding CIFAR-10-specific assumptions:

- CIFAR-10 source/version/license notes map to dataset identity.
- The ten CIFAR-10 labels map to ordered class names.
- The official train/test source partitions can be represented as declared splits.
- A deterministic validation split can be represented as `val` after Phase 1B approves the split policy.
- CIFAR-10 image size and RGB preprocessing metadata fit the preprocessing spec.
- CIFAR-10 sample records can use stable sample IDs and relative or source-local references.

Known limitation for Phase 1B planning: CIFAR-10 does not expose rich correlated-group metadata, so `group_id` may be unavailable and should be recorded as a leakage-limit note.
