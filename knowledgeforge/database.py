import sqlite3
from contextlib import closing
from pathlib import Path

from config import AUTH_DB_PATH


USERS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    email TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    created_at TEXT NOT NULL,
    is_active INTEGER NOT NULL DEFAULT 1 CHECK (is_active IN (0, 1))
)
"""


def get_connection(db_path=None):
    path = Path(db_path or AUTH_DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database(db_path=None):
    with closing(get_connection(db_path)) as connection:
        with connection:
            connection.execute(USERS_TABLE_SQL)
