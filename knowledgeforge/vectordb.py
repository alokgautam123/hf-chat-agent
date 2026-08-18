import chromadb

from config import COLLECTION_NAME
from config import VECTOR_DB_DIR

_client = None
_collection = None


def get_collection():
    global _client
    global _collection

    if _client is None:
        _client = chromadb.PersistentClient(path=VECTOR_DB_DIR)

    if _collection is None:
        _collection = _client.get_or_create_collection(
            name=COLLECTION_NAME
        )

    return _collection

def index_documents(ids, documents, embeddings, metadatas):
    collection = get_collection()

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings.tolist(),
        metadatas=metadatas
    )

def search(query_embedding, top_k, user_id):
    collection = get_collection()

    return collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=top_k,
        where={"user_id": user_id},
    )


def delete_by_source(source, user_id):
    collection = get_collection()
    ownership_filter = {
        "$and": [
            {"user_id": user_id},
            {"source": source},
        ]
    }
    matching_entries = collection.get(
        where=ownership_filter,
        include=[],
    )

    collection.delete(
        where=ownership_filter,
    )

    return len(matching_entries["ids"])


def document_count(user_id):
    collection = get_collection()
    matching_entries = collection.get(
        where={"user_id": user_id},
        include=[],
    )
    return len(matching_entries["ids"])
