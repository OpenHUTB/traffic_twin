"""Load database connection settings without committing credentials."""

import os


def _required_value(environ, name):
    value = environ.get(name, "")
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def database_config_from_env(environ=None):
    """Return keyword arguments for ``mysql.connector.connect``."""
    environ = os.environ if environ is None else environ

    host = environ.get("TRAFFIC_TWIN_DB_HOST", "localhost").strip()
    database = environ.get("TRAFFIC_TWIN_DB_NAME", "test").strip()
    if not host or not database:
        raise ValueError("TRAFFIC_TWIN_DB_HOST and TRAFFIC_TWIN_DB_NAME cannot be empty")

    try:
        port = int(environ.get("TRAFFIC_TWIN_DB_PORT", "3306"))
    except ValueError as exc:
        raise ValueError("TRAFFIC_TWIN_DB_PORT must be an integer") from exc
    if not 1 <= port <= 65535:
        raise ValueError("TRAFFIC_TWIN_DB_PORT must be between 1 and 65535")

    return {
        "host": host,
        "port": port,
        "user": _required_value(environ, "TRAFFIC_TWIN_DB_USER"),
        "password": _required_value(environ, "TRAFFIC_TWIN_DB_PASSWORD"),
        "database": database,
    }
