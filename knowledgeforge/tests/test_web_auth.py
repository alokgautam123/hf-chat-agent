import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("SESSION_SECRET", "test-session-secret-with-sufficient-length")

import config

config.SESSION_SECRET = os.environ["SESSION_SECRET"]

import database
import web
from fastapi.testclient import TestClient
from services.auth import create_user


def csrf_token(response):
    match = re.search(r'name="csrf_token" value="([^"]+)"', response.text)

    if not match:
        raise AssertionError("CSRF token was not rendered.")

    return match.group(1)


class WebAuthenticationTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        database.AUTH_DB_PATH = str(self.root / "auth-test.db")
        web.DOCS_DIR = str(self.root / "docs")
        self.documents_patcher = patch(
            "web.list_indexed_documents",
            return_value=[],
        )
        self.documents_patcher.start()
        self.client = TestClient(web.app)

    def tearDown(self):
        self.documents_patcher.stop()
        self.temporary_directory.cleanup()

    def register_user(self, email="user@example.com"):
        page = self.client.get("/register")
        return self.client.post(
            "/register",
            data={
                "email": email,
                "password": "correct horse battery staple",
                "csrf_token": csrf_token(page),
            },
            follow_redirects=False,
        )

    def test_registration_creates_user_and_signs_in(self):
        response = self.register_user(" NewUser@Example.COM ")

        self.assertEqual(response.status_code, 303)
        self.assertEqual(response.headers["location"], "/")

        home = self.client.get("/")
        self.assertEqual(home.status_code, 200)
        self.assertIn("newuser@example.com", home.text)

    def test_login_establishes_session(self):
        create_user(
            "user@example.com",
            "correct horse battery staple",
        )
        page = self.client.get("/login")
        response = self.client.post(
            "/login",
            data={
                "email": " USER@EXAMPLE.COM ",
                "password": "correct horse battery staple",
                "csrf_token": csrf_token(page),
            },
            follow_redirects=False,
        )

        self.assertEqual(response.status_code, 303)
        self.assertEqual(self.client.get("/").status_code, 200)

    def test_logout_clears_session(self):
        self.register_user()
        home = self.client.get("/")
        response = self.client.post(
            "/logout",
            data={"csrf_token": csrf_token(home)},
            follow_redirects=False,
        )

        self.assertEqual(response.status_code, 303)
        self.assertEqual(response.headers["location"], "/login")
        protected = self.client.get("/", follow_redirects=False)
        self.assertEqual(protected.status_code, 303)

    def test_anonymous_users_are_redirected_from_protected_routes(self):
        requests = [
            ("get", "/", {}),
            ("post", "/ask", {"data": {"question": "test"}}),
            ("post", "/upload", {}),
            ("post", "/documents/delete", {"data": {"source": "test.txt"}}),
        ]

        for method, path, kwargs in requests:
            with self.subTest(path=path):
                response = getattr(self.client, method)(
                    path,
                    follow_redirects=False,
                    **kwargs,
                )
                self.assertEqual(response.status_code, 303)
                self.assertEqual(response.headers["location"], "/login")

    def test_invalid_login_is_rejected(self):
        create_user(
            "user@example.com",
            "correct horse battery staple",
        )
        page = self.client.get("/login")
        response = self.client.post(
            "/login",
            data={
                "email": "user@example.com",
                "password": "wrong password",
                "csrf_token": csrf_token(page),
            },
        )

        self.assertEqual(response.status_code, 401)
        self.assertIn("Invalid email or password.", response.text)

    def test_authenticated_posts_require_valid_csrf_token(self):
        self.register_user()

        with patch("web.answer_question") as ask:
            response = self.client.post(
                "/ask",
                data={"question": "What is TTL?"},
            )

        self.assertEqual(response.status_code, 400)
        ask.assert_not_called()

    def test_authenticated_product_routes_keep_using_existing_services(self):
        self.register_user()
        home = self.client.get("/")
        token = csrf_token(home)
        answer = {
            "answer": "TTL limits an IP datagram's lifetime.",
            "answered": True,
            "citations": [{"source": "RFC791.pdf", "page": 8}],
        }

        with (
            patch("web.ingest_document") as ingest,
            patch("web.answer_question", return_value=answer) as ask,
            patch(
                "web.delete_indexed_document",
                return_value={"deleted": True},
            ) as delete,
        ):
            upload = self.client.post(
                "/upload",
                data={"csrf_token": token},
                files={"file": ("route-test.txt", b"test", "text/plain")},
            )
            question = self.client.post(
                "/ask",
                data={"question": "What is TTL?", "csrf_token": token},
            )
            deletion = self.client.post(
                "/documents/delete",
                data={"source": "route-test.txt", "csrf_token": token},
            )

        self.assertEqual(upload.status_code, 200)
        uploaded_file = self.root / "docs" / ingest.call_args.args[1] / "route-test.txt"
        self.assertTrue(uploaded_file.is_file())
        ingest.assert_called_once_with(uploaded_file, ingest.call_args.args[1])
        self.assertEqual(question.status_code, 200)
        self.assertIn("TTL limits an IP datagram", question.text)
        ask.assert_called_once_with("What is TTL?", ingest.call_args.args[1])
        self.assertEqual(deletion.status_code, 200)
        delete.assert_called_once_with("route-test.txt", ingest.call_args.args[1])

    def test_two_users_can_upload_the_same_filename_to_separate_paths(self):
        self.register_user("first@example.com")
        first_token = csrf_token(self.client.get("/"))

        second_client = TestClient(web.app)
        register_page = second_client.get("/register")
        second_client.post(
            "/register",
            data={
                "email": "second@example.com",
                "password": "correct horse battery staple",
                "csrf_token": csrf_token(register_page),
            },
        )
        second_token = csrf_token(second_client.get("/"))

        with patch("web.ingest_document") as ingest:
            first_upload = self.client.post(
                "/upload",
                data={"csrf_token": first_token},
                files={"file": ("shared.txt", b"first", "text/plain")},
            )
            second_upload = second_client.post(
                "/upload",
                data={"csrf_token": second_token},
                files={"file": ("shared.txt", b"second", "text/plain")},
            )

        first_path, first_user_id = ingest.call_args_list[0].args
        second_path, second_user_id = ingest.call_args_list[1].args

        self.assertEqual(first_upload.status_code, 200)
        self.assertEqual(second_upload.status_code, 200)
        self.assertNotEqual(first_user_id, second_user_id)
        self.assertEqual(first_path, self.root / "docs" / first_user_id / "shared.txt")
        self.assertEqual(second_path, self.root / "docs" / second_user_id / "shared.txt")
        self.assertTrue(first_path.is_file())
        self.assertTrue(second_path.is_file())


if __name__ == "__main__":
    unittest.main()
