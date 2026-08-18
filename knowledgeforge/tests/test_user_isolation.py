import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import services.documents as documents_service
import vectordb
from services.documents import delete_indexed_document, list_indexed_documents
from services.qa import answer_question


class UserIsolationTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.docs_path = self.root / "docs"
        documents_service.DOCS_DIR = str(self.docs_path)

        vectordb.VECTOR_DB_DIR = str(self.root / "chroma")
        vectordb._client = None
        vectordb._collection = None
        self.collection = vectordb.get_collection()

    def tearDown(self):
        vectordb._client = None
        vectordb._collection = None
        self.temporary_directory.cleanup()

    def add_vector(self, vector_id, user_id, source, document, embedding):
        self.collection.add(
            ids=[vector_id],
            documents=[document],
            embeddings=[embedding],
            metadatas=[
                {
                    "user_id": user_id,
                    "source": source,
                    "type": "txt",
                    "page": 1,
                }
            ],
        )

    def test_each_user_lists_only_their_documents(self):
        self.add_vector("a:0", "user-a", "a.txt", "alpha", [1.0, 0.0])
        self.add_vector("b:0", "user-b", "b.txt", "beta", [0.0, 1.0])

        self.assertEqual(
            list_indexed_documents("user-a"),
            [{"source": "a.txt"}],
        )
        self.assertEqual(
            list_indexed_documents("user-b"),
            [{"source": "b.txt"}],
        )

    def test_qa_does_not_retrieve_another_users_vectors(self):
        self.add_vector(
            "b:secret",
            "user-b",
            "private.txt",
            "User B secret content",
            [1.0, 0.0],
        )

        with patch(
            "services.qa.create_embeddings",
            return_value=np.array([[1.0, 0.0]]),
        ):
            result = answer_question("What is the secret?", "user-a")

        self.assertFalse(result["answered"])
        self.assertEqual(result["citations"], [])

    def test_user_cannot_delete_another_users_document(self):
        user_b_directory = self.docs_path / "user-b"
        user_b_directory.mkdir(parents=True)
        user_b_file = user_b_directory / "private.txt"
        user_b_file.write_text("private", encoding="utf-8")
        self.add_vector(
            "b:private",
            "user-b",
            "private.txt",
            "private",
            [1.0, 0.0],
        )

        result = delete_indexed_document("private.txt", "user-a")

        self.assertFalse(result["deleted"])
        self.assertTrue(user_b_file.is_file())
        self.assertEqual(
            self.collection.get(
                where={"user_id": "user-b"},
                include=[],
            )["ids"],
            ["b:private"],
        )

    def test_same_filename_is_listed_and_deleted_per_user(self):
        for user_id, vector_id, content, embedding in [
            ("user-a", "a:shared", "alpha", [1.0, 0.0]),
            ("user-b", "b:shared", "beta", [0.0, 1.0]),
        ]:
            user_directory = self.docs_path / user_id
            user_directory.mkdir(parents=True)
            (user_directory / "shared.txt").write_text(content, encoding="utf-8")
            self.add_vector(
                vector_id,
                user_id,
                "shared.txt",
                content,
                embedding,
            )

        result = delete_indexed_document("shared.txt", "user-a")

        self.assertTrue(result["deleted"])
        self.assertFalse((self.docs_path / "user-a" / "shared.txt").exists())
        self.assertTrue((self.docs_path / "user-b" / "shared.txt").is_file())
        self.assertEqual(list_indexed_documents("user-a"), [])
        self.assertEqual(
            list_indexed_documents("user-b"),
            [{"source": "shared.txt"}],
        )


if __name__ == "__main__":
    unittest.main()
