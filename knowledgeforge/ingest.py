from document_loader import load_documents
from text_splitter import create_chunks
from embeddings import create_embeddings
from vectordb import index_documents, document_count

print("Loading documents...")
documents = load_documents()

print(f"Loaded {len(documents)} documents")

print("Splitting documents...")
chunks = create_chunks(documents)

print(f"Created {len(chunks)} chunks")

print("Generating embeddings...")
embeddings = create_embeddings(chunks)

ids = [f"chunk_{i}" for i in range(len(chunks))]

texts = [chunk["text"] for chunk in chunks]

metadatas = [chunk["metadata"] for chunk in chunks]

print("Indexing into ChromaDB...")
index_documents(
    ids=ids,
    documents=texts,
    embeddings=embeddings,
    metadatas=metadatas,
)

print()

print("Knowledge base indexed successfully!")
print(f"Total vectors: {document_count()}")
