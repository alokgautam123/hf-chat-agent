import sqlite3
import uuid
from contextlib import closing
from datetime import datetime, timezone

from pwdlib import PasswordHash

from database import get_connection, initialize_database


password_hasher = PasswordHash.recommended()


class DuplicateEmailError(ValueError):
    """Raised when an account already uses the normalized email address."""


def normalize_email(email):
    return email.strip().casefold()


def _row_to_user(row):
    if row is None:
        return None

    return {
        "id": row["id"],
        "email": row["email"],
        "password_hash": row["password_hash"],
        "created_at": row["created_at"],
        "is_active": bool(row["is_active"]),
    }


def create_user(email, password, db_path=None):
    normalized_email = normalize_email(email)

    if not normalized_email:
        raise ValueError("Email is required.")

    if not password:
        raise ValueError("Password is required.")

    initialize_database(db_path)
    user = {
        "id": str(uuid.uuid4()),
        "email": normalized_email,
        "password_hash": password_hasher.hash(password),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "is_active": True,
    }

    try:
        with closing(get_connection(db_path)) as connection:
            with connection:
                connection.execute(
                    """
                    INSERT INTO users (
                        id,
                        email,
                        password_hash,
                        created_at,
                        is_active
                    )
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        user["id"],
                        user["email"],
                        user["password_hash"],
                        user["created_at"],
                        int(user["is_active"]),
                    ),
                )
    except sqlite3.IntegrityError as error:
        raise DuplicateEmailError(
            "An account with that email already exists."
        ) from error

    return user


def find_user_by_email(email, db_path=None):
    initialize_database(db_path)

    with closing(get_connection(db_path)) as connection:
        row = connection.execute(
            "SELECT * FROM users WHERE email = ?",
            (normalize_email(email),),
        ).fetchone()

    return _row_to_user(row)


def find_user_by_id(user_id, db_path=None):
    initialize_database(db_path)

    with closing(get_connection(db_path)) as connection:
        row = connection.execute(
            "SELECT * FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()

    return _row_to_user(row)


def verify_user_password(user, password):
    if not user or not user["is_active"]:
        return False

    return password_hasher.verify(password, user["password_hash"])
