import hashlib

from document_loader import load_document, load_documents
from embeddings import create_embeddings
from text_splitter import create_chunks
from vectordb import document_count, index_documents


def _create_chunk_ids(chunks, user_id):
    chunk_indexes = {}
    ids = []

    for chunk in chunks:
        source = chunk["metadata"]["source"]
        document_key = f"{user_id}:{source}"
        document_id = hashlib.sha256(document_key.encode("utf-8")).hexdigest()
        chunk_index = chunk_indexes.get(document_id, 0)

        ids.append(f"{document_id}:{chunk_index}")
        chunk_indexes[document_id] = chunk_index + 1

    return ids


def _index_loaded_documents(documents, user_id):
    owned_documents = [
        {
            "text": document["text"],
            "metadata": {
                **document["metadata"],
                "user_id": user_id,
            },
        }
        for document in documents
    ]
    chunks = create_chunks(owned_documents)
    embeddings = create_embeddings(chunks)

    ids = _create_chunk_ids(chunks, user_id)
    texts = [chunk["text"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]

    index_documents(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    return {
        "documents_loaded": len(documents),
        "chunks_created": len(chunks),
        "total_vectors": document_count(user_id),
    }


def ingest_document(file_path, user_id):
    """Load and index one supported document."""
    documents = load_document(file_path)
    result = _index_loaded_documents(documents, user_id)
    result["file_path"] = str(file_path)
    return result


def ingest_documents(user_id="legacy"):
    """Load and index all documents in the configured documents directory."""
    documents = load_documents()
    return _index_loaded_documents(documents, user_id)
