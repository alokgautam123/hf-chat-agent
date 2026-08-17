from services.ingestion import ingest_documents

print("Loading documents...")
print("Splitting documents...")
print("Generating embeddings...")
print("Indexing into ChromaDB...")
result = ingest_documents()

print()

print("Knowledge base indexed successfully!")
print(f"Loaded {result['documents_loaded']} documents")
print(f"Created {result['chunks_created']} chunks")
print(f"Total vectors: {result['total_vectors']}")
