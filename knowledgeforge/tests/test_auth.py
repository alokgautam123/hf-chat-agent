import sqlite3
import tempfile
import unittest
import uuid
from pathlib import Path

from services.auth import (
    DuplicateEmailError,
    create_user,
    find_user_by_email,
    find_user_by_id,
    verify_user_password,
)


class AuthenticationServiceTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temporary_directory.name) / "auth-test.db"

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_create_user(self):
        user = create_user(
            "user@example.com",
            "correct horse battery staple",
            self.db_path,
        )

        self.assertEqual(user, find_user_by_id(user["id"], self.db_path))
        self.assertEqual(uuid.UUID(user["id"]).version, 4)
        self.assertTrue(user["is_active"])
        self.assertTrue(user["created_at"])

    def test_password_is_stored_hashed(self):
        password = "correct horse battery staple"
        user = create_user("user@example.com", password, self.db_path)

        self.assertNotEqual(user["password_hash"], password)
        self.assertNotIn(password, user["password_hash"])
        self.assertTrue(user["password_hash"].startswith("$argon2"))

    def test_email_is_normalized(self):
        user = create_user(
            "  User@Example.COM  ",
            "correct horse battery staple",
            self.db_path,
        )

        self.assertEqual(user["email"], "user@example.com")
        self.assertEqual(
            find_user_by_email(" USER@example.com ", self.db_path)["id"],
            user["id"],
        )

    def test_duplicate_normalized_email_is_rejected(self):
        create_user(
            "user@example.com",
            "correct horse battery staple",
            self.db_path,
        )

        with self.assertRaises(DuplicateEmailError):
            create_user(
                " User@Example.COM ",
                "another secure password",
                self.db_path,
            )

    def test_correct_password_is_verified(self):
        user = create_user(
            "user@example.com",
            "correct horse battery staple",
            self.db_path,
        )

        self.assertTrue(
            verify_user_password(user, "correct horse battery staple")
        )

    def test_incorrect_password_is_rejected(self):
        user = create_user(
            "user@example.com",
            "correct horse battery staple",
            self.db_path,
        )

        self.assertFalse(verify_user_password(user, "wrong password"))

    def test_users_table_structure(self):
        create_user(
            "user@example.com",
            "correct horse battery staple",
            self.db_path,
        )

        with sqlite3.connect(self.db_path) as connection:
            columns = connection.execute("PRAGMA table_info(users)").fetchall()

        self.assertEqual(
            [column[1] for column in columns],
            ["id", "email", "password_hash", "created_at", "is_active"],
        )


if __name__ == "__main__":
    unittest.main()
