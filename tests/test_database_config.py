import ast
import unittest
from pathlib import Path

from intersection_navigation.database_config import database_config_from_env


NAVIGATION_PATH = (
    Path(__file__).resolve().parents[1]
    / "intersection_navigation"
    / "twin_navigation.py"
)


class DatabaseConfigTest(unittest.TestCase):
    def test_reads_connection_settings_from_environment(self):
        config = database_config_from_env(
            {
                "TRAFFIC_TWIN_DB_HOST": "db.internal",
                "TRAFFIC_TWIN_DB_PORT": "3307",
                "TRAFFIC_TWIN_DB_NAME": "traffic",
                "TRAFFIC_TWIN_DB_USER": "traffic_service",
                "TRAFFIC_TWIN_DB_PASSWORD": "from-environment",
            }
        )

        self.assertEqual(
            config,
            {
                "host": "db.internal",
                "port": 3307,
                "user": "traffic_service",
                "password": "from-environment",
                "database": "traffic",
            },
        )

    def test_uses_non_secret_defaults(self):
        config = database_config_from_env(
            {
                "TRAFFIC_TWIN_DB_USER": "traffic_service",
                "TRAFFIC_TWIN_DB_PASSWORD": "from-environment",
            }
        )

        self.assertEqual(config["host"], "localhost")
        self.assertEqual(config["port"], 3306)
        self.assertEqual(config["database"], "test")

    def test_rejects_missing_credentials(self):
        with self.assertRaisesRegex(ValueError, "TRAFFIC_TWIN_DB_USER"):
            database_config_from_env({})

    def test_rejects_invalid_port(self):
        with self.assertRaisesRegex(ValueError, "must be an integer"):
            database_config_from_env(
                {
                    "TRAFFIC_TWIN_DB_PORT": "not-a-port",
                    "TRAFFIC_TWIN_DB_USER": "traffic_service",
                    "TRAFFIC_TWIN_DB_PASSWORD": "from-environment",
                }
            )

    def test_navigation_does_not_pass_a_password_literal_to_connect(self):
        tree = ast.parse(NAVIGATION_PATH.read_text(encoding="utf-8"))
        connect_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "connect"
        ]

        self.assertTrue(connect_calls)
        for call in connect_calls:
            self.assertFalse(
                any(
                    keyword.arg == "password"
                    and isinstance(keyword.value, ast.Constant)
                    for keyword in call.keywords
                )
            )


if __name__ == "__main__":
    unittest.main()
