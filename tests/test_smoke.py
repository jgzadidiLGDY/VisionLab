import unittest

from visionlab.smoke import environment_summary


class SmokeTest(unittest.TestCase):
    def test_environment_summary_finds_project_spec(self):
        summary = environment_summary()

        self.assertEqual(summary["visionlab_version"], "0.0.0")
        self.assertEqual(summary["project_spec_exists"], "True")
