# T1 Development Dataset Candidate Comparison

Status: T1 recommendation, not an applied-domain decision.

T1 compares low-friction datasets for early custom-CNN and data-contract work. The provisional development dataset should make Phase 1 feasible without selecting the eventual applied domain.

## Selection Criteria

- accessible through common tooling such as torchvision;
- clear classification labels;
- small enough for local CPU smoke checks and bounded GPU training later;
- visual enough to teach RGB/channel/preprocessing issues;
- documented source and split structure;
- not tied to the deferred applied-domain gate;
- low privacy and safety risk for a learning repository;
- supports visual inspection and failure analysis.

## Candidates

| Candidate | Evidence | Strengths | Risks or limits | T1 judgment |
| --- | --- | --- | --- | --- |
| Fashion-MNIST | 60,000 train and 10,000 test examples; 28x28 grayscale; 10 classes; MIT license. Source: https://github.com/zalandoresearch/fashion-mnist | Very low friction, small, clear labels, fast CPU checks. | Grayscale only; less useful for RGB/channel learning; too MNIST-like for later robustness/OOD work. | Useful fallback or tiny smoke fixture, not best core dataset. |
| CIFAR-10 | 60,000 32x32 color images; 10 classes; 50,000 train and 10,000 test; 6,000 images per class. Source: https://cave.cs.toronto.edu/kriz/cifar.html | Low friction, RGB, balanced, widely supported by torchvision, small enough for bounded training. | Tiny 32x32 images limit visual inspection detail; no official validation split; web-image origin has known broader dataset concerns. | Recommended provisional core development dataset. |
| STL-10 | 96x96 color images; 10 classes; 500 train and 800 test images per class; includes 100,000 unlabeled images. Source: https://cs.stanford.edu/~acoates/stl10/ | Higher resolution than CIFAR-10 and useful for representation-learning discussion. | Larger and more complex; unlabeled set is outside early supervised scope; fewer labeled training examples; ImageNet-derived source requires care. | Candidate for later comparison or fallback, not first core dataset. |
| EuroSAT RGB | 27,000 Sentinel-2 image patches; 10 land-use/land-cover classes; RGB version commonly used. Example source: https://opengeoai.org/examples/image_recognition/ | RGB, compact, visually coherent, useful for domain-shift discussion later. | Satellite imagery may prematurely bias the applied-domain imagination; source/licensing route needs more careful Phase 1 evidence; less natural for first custom-CNN fundamentals. | Defer until a later data or domain-feasibility phase. |

## Recommendation

Use **CIFAR-10** as the provisional core development dataset for Phase 1 planning.

Rationale:

- It is RGB, which better supports image-tensor, channel, normalization, augmentation, and custom-CNN learning than grayscale Fashion-MNIST.
- It is small and balanced enough for bounded CPU/GPU workflow.
- It has a widely understood train/test structure.
- It does not select or imply the applied capstone domain.

Phase 1 should still treat this as provisional. Before material training, VisionLab must register dataset identity, source, license/usage notes, class mapping, validation split policy, visual sample grids, and any known data-quality limitations.

## Explicit Non-Decision

This recommendation is not the applied-domain selection. The applied domain remains deferred until the approved domain-feasibility gate.
