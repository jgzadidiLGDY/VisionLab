import unittest
from dataclasses import replace
from pathlib import Path

from visionlab.data import (
    ClassMapping,
    DatasetIdentity,
    DatasetManifest,
    PreprocessingSpec,
    SampleRecord,
    validate_manifest,
)


FIXTURE_ROOT = Path("data/fixtures/phase1_tiny_rgb")


def tiny_manifest(samples=None):
    return DatasetManifest(
        identity=DatasetIdentity(
            dataset_id="phase1_tiny_rgb",
            version="1.0",
            source="committed VisionLab tiny fixtures",
            license_or_usage="project test fixture",
            description="Four tiny RGB Netpbm images for dataset-contract tests.",
        ),
        classes=ClassMapping(("red", "green")),
        split_names=("train", "val", "test"),
        preprocessing=PreprocessingSpec(
            image_size=(4, 4),
            color_mode="RGB",
            value_range=(0.0, 1.0),
            normalization_mean=(0.5, 0.5, 0.5),
            normalization_std=(0.5, 0.5, 0.5),
        ),
        samples=tuple(
            samples
            if samples is not None
            else (
                SampleRecord(
                    sample_id="train-red-000",
                    split="train",
                    label="red",
                    relative_path="train/red_block.ppm",
                    source_id="phase1-fixture",
                    group_id="red-block",
                ),
                SampleRecord(
                    sample_id="train-green-000",
                    split="train",
                    label="green",
                    relative_path="train/green_block.ppm",
                    source_id="phase1-fixture",
                    group_id="green-block",
                ),
                SampleRecord(
                    sample_id="val-red-000",
                    split="val",
                    label="red",
                    relative_path="val/red_block_val.ppm",
                    source_id="phase1-fixture",
                    group_id="red-block-val",
                ),
                SampleRecord(
                    sample_id="test-green-000",
                    split="test",
                    label="green",
                    relative_path="test/green_block_test.ppm",
                    source_id="phase1-fixture",
                    group_id="green-block-test",
                ),
            )
        ),
    )


class DatasetContractTest(unittest.TestCase):
    def test_tiny_manifest_validates_and_counts_classes_by_split(self):
        report = validate_manifest(tiny_manifest(), FIXTURE_ROOT)

        self.assertTrue(report.is_valid)
        self.assertEqual(report.errors, ())
        self.assertEqual(
            report.class_counts_by_split,
            {
                "train": {"red": 1, "green": 1},
                "val": {"red": 1, "green": 0},
                "test": {"red": 0, "green": 1},
            },
        )

    def test_manifest_rejects_duplicate_sample_ids(self):
        base = tiny_manifest()
        duplicate = replace(base.samples[1], sample_id=base.samples[0].sample_id)
        report = validate_manifest(
            replace(base, samples=(base.samples[0], duplicate)),
            FIXTURE_ROOT,
        )

        self.assertIn("duplicate_sample_id", {error.code for error in report.errors})

    def test_manifest_rejects_unknown_label(self):
        base = tiny_manifest()
        bad_sample = replace(base.samples[0], label="blue")
        report = validate_manifest(replace(base, samples=(bad_sample,)), FIXTURE_ROOT)

        self.assertIn("unknown_label", {error.code for error in report.errors})

    def test_manifest_rejects_unknown_split(self):
        base = tiny_manifest()
        bad_sample = replace(base.samples[0], split="holdout")
        report = validate_manifest(replace(base, samples=(bad_sample,)), FIXTURE_ROOT)

        self.assertIn("unknown_split", {error.code for error in report.errors})

    def test_manifest_rejects_missing_files(self):
        base = tiny_manifest()
        bad_sample = replace(base.samples[0], relative_path="train/missing.ppm")
        report = validate_manifest(replace(base, samples=(bad_sample,)), FIXTURE_ROOT)

        self.assertIn("missing_file", {error.code for error in report.errors})

    def test_manifest_rejects_paths_outside_dataset_root(self):
        base = tiny_manifest()
        bad_sample = replace(base.samples[0], relative_path="../README.md")
        report = validate_manifest(replace(base, samples=(bad_sample,)), FIXTURE_ROOT)

        self.assertIn("unsafe_path", {error.code for error in report.errors})

    def test_manifest_rejects_image_size_mismatch(self):
        base = tiny_manifest()
        bad_preprocessing = replace(base.preprocessing, image_size=(8, 8))
        report = validate_manifest(
            replace(base, preprocessing=bad_preprocessing),
            FIXTURE_ROOT,
        )

        self.assertIn("image_size_mismatch", {error.code for error in report.errors})

    def test_manifest_rejects_color_mode_mismatch(self):
        base = tiny_manifest()
        bad_preprocessing = PreprocessingSpec(
            image_size=(4, 4),
            color_mode="L",
            value_range=(0.0, 1.0),
            normalization_mean=(0.5,),
            normalization_std=(0.5,),
        )
        report = validate_manifest(
            replace(base, preprocessing=bad_preprocessing),
            FIXTURE_ROOT,
        )

        self.assertIn("color_mode_mismatch", {error.code for error in report.errors})

    def test_preprocessing_spec_checks_channel_metadata(self):
        with self.assertRaisesRegex(ValueError, "normalization_mean"):
            PreprocessingSpec(
                image_size=(4, 4),
                color_mode="RGB",
                value_range=(0.0, 1.0),
                normalization_mean=(0.5,),
                normalization_std=(0.5, 0.5, 0.5),
            )

    def test_manifest_serializes_contract_shape(self):
        as_dict = tiny_manifest().to_dict()

        self.assertEqual(as_dict["identity"]["dataset_id"], "phase1_tiny_rgb")
        self.assertEqual(as_dict["classes"], ["red", "green"])
        self.assertEqual(as_dict["split_names"], ["train", "val", "test"])
        self.assertEqual(as_dict["preprocessing"]["color_mode"], "RGB")
        self.assertEqual(as_dict["samples"][0]["sample_id"], "train-red-000")


if __name__ == "__main__":
    unittest.main()
