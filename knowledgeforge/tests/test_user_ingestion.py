import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from services.ingestion import _create_chunk_ids, ingest_document


class UserIngestionTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_same_filename_has_different_stable_ids_for_each_user(self):
        chunks = [
            {
                "text": "shared document",
                "metadata": {"source": "shared.txt"},
            }
        ]

        first_ids = _create_chunk_ids(chunks, "user-one")
        second_ids = _create_chunk_ids(chunks, "user-two")

        self.assertNotEqual(first_ids, second_ids)
        self.assertEqual(first_ids, _create_chunk_ids(chunks, "user-one"))
        self.assertEqual(second_ids, _create_chunk_ids(chunks, "user-two"))

    def test_same_filename_is_indexed_with_each_users_ownership(self):
        first_file = self.root / "user-one" / "shared.txt"
        second_file = self.root / "user-two" / "shared.txt"
        first_file.parent.mkdir()
        second_file.parent.mkdir()
        first_file.write_text("first user's content", encoding="utf-8")
        second_file.write_text("second user's content", encoding="utf-8")

        indexed_batches = []

        def capture_index(ids, documents, embeddings, metadatas):
            indexed_batches.append(
                {
                    "ids": ids,
                    "documents": documents,
                    "metadatas": metadatas,
                }
            )

        with (
            patch(
                "services.ingestion.create_embeddings",
                side_effect=lambda chunks: np.zeros((len(chunks), 2)),
            ),
            patch("services.ingestion.index_documents", side_effect=capture_index),
            patch("services.ingestion.document_count", return_value=2),
        ):
            ingest_document(first_file, "user-one")
            ingest_document(second_file, "user-two")

        first_batch, second_batch = indexed_batches

        self.assertTrue(
            all(meta["user_id"] == "user-one" for meta in first_batch["metadatas"])
        )
        self.assertTrue(
            all(meta["user_id"] == "user-two" for meta in second_batch["metadatas"])
        )
        self.assertTrue(
            set(first_batch["ids"]).isdisjoint(second_batch["ids"])
        )


if __name__ == "__main__":
    unittest.main()
