import os
import tempfile
import unittest
from pathlib import Path

from doctor import PLACEHOLDERS, load_dotenv


class DoctorTests(unittest.TestCase):
    def test_load_dotenv_parses_comments_quotes_and_blank_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text(
                """
# comment
GEMINI_API_KEY="abc123"
DASHSCOPE_API_KEY='dashscope456'
WORKSPACE_PATHS=./workspace, ../docs
BROKEN_LINE
""",
                encoding="utf-8",
            )

            values = load_dotenv(env_file)

        self.assertEqual(values["GEMINI_API_KEY"], "abc123")
        self.assertEqual(values["DASHSCOPE_API_KEY"], "dashscope456")
        self.assertEqual(values["WORKSPACE_PATHS"], "./workspace, ../docs")
        self.assertNotIn("BROKEN_LINE", values)

    def test_placeholder_set_treats_empty_values_as_missing(self):
        self.assertIn("", PLACEHOLDERS)
        self.assertIn("your_gemini_api_key_here", PLACEHOLDERS)

    def test_load_dotenv_returns_empty_dict_for_missing_files(self):
        missing = Path(os.environ.get("LOCALAPPDATA", tempfile.gettempdir())) / "missing-local-brain.env"
        self.assertEqual(load_dotenv(missing), {})


if __name__ == "__main__":
    unittest.main()
