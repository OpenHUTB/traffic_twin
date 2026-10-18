import importlib.util
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "waypoint_control"
    / "multi_obj_track"
    / "split_data.py"
)


def load_split_data_module():
    spec = importlib.util.spec_from_file_location("split_data", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SplitDataSecurityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_split_data_module()
        cls.junction_key = cls.module.JUNCTION_PATTERN.pattern.split(r"\s*")[0]

    def test_normal_junction_id_stays_inside_output_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "output"
            input_file = root / "v2x_latency.log"
            input_file.write_text(
                f"{self.junction_key} 1234, lane=1\n",
                encoding="utf-8",
            )

            self.module.split_data_by_junction(input_file, output_dir)

            self.assertTrue((output_dir / "1234" / input_file.name).is_file())

    def test_path_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "output"
            input_file = root / "v2x_latency.log"
            input_file.write_text(
                f"{self.junction_key} ../../outside, lane=1\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "relative directory name"):
                self.module.split_data_by_junction(input_file, output_dir)

            self.assertFalse((root.parent / "outside" / input_file.name).exists())

    def test_absolute_junction_path_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "relative directory name"):
            self.module.safe_junction_path("output", "/tmp/outside")

    def test_backslash_traversal_is_rejected_on_every_platform(self):
        with self.assertRaisesRegex(ValueError, "relative directory name"):
            self.module.safe_junction_path("output", r"..\outside")

    def test_windows_drive_relative_path_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "relative directory name"):
            self.module.safe_junction_path("output", "C:outside")


if __name__ == "__main__":
    unittest.main()
