from pathlib import Path

from config import DOCS_DIR
from vectordb import delete_by_source
from vectordb import get_collection


def list_indexed_documents(user_id):
    """Return the unique document sources stored in ChromaDB metadata."""
    results = get_collection().get(
        where={"user_id": user_id},
        include=["metadatas"],
    )
    sources = {
        metadata.get("source")
        for metadata in results["metadatas"]
        if metadata.get("source")
    }

    return [
        {"source": source}
        for source in sorted(sources)
    ]


def delete_indexed_document(source, user_id):
    """Delete a document's vectors and its stored source file."""
    if not source or Path(source).name != source:
        raise ValueError("Document source must be a filename.")

    vectors_deleted = delete_by_source(source, user_id)
    file_path = Path(DOCS_DIR) / user_id / source
    file_deleted = False

    if file_path.is_file():
        file_path.unlink()
        file_deleted = True

    return {
        "source": source,
        "vectors_deleted": vectors_deleted,
        "file_deleted": file_deleted,
        "deleted": vectors_deleted > 0 or file_deleted,
    }
