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

def search(query_embedding, top_k):
    collection = get_collection()

    return collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=top_k
    )


def document_count():
    collection = get_collection()
    return collection.count()
