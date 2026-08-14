import unittest

from visionlab.environment import environment_summary, package_status


class EnvironmentProbeTest(unittest.TestCase):
    def test_package_status_handles_missing_optional_dependency(self):
        status = package_status("definitely_not_a_real_visionlab_dependency")

        self.assertFalse(status.installed)
        self.assertIn("ModuleNotFoundError", status.import_error)

    def test_environment_summary_has_torch_device_section(self):
        summary = environment_summary()

        self.assertIn("python_version", summary)
        self.assertIn("packages", summary)
        self.assertIn("torch_device", summary)
        self.assertIn("recommended_local_device", summary["torch_device"])
