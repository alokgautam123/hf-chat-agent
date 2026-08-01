import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import os

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "rfc791"
LLM_MODEL = "deepseek-ai/DeepSeek-V3"

llm_client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_TOKEN"],
)

# Load embedding model
model = SentenceTransformer(EMBEDDING_MODEL)

# Connect to ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")

try:
    collection = client.get_collection(COLLECTION_NAME)
except Exception:
    print("Collection not found.")
    print("Run ingest.py first.")
    exit(1)



# Ask a question
question = input("Ask a question: ")

# Convert question to embedding
query_embedding = model.encode(question)

# Search
results = collection.query(
    query_embeddings=[query_embedding.tolist()],
    n_results=5
)

contexts = results["documents"][0]

SHOW_CONTEXT = False
if SHOW_CONTEXT:
    print("\nRetrieved Context:\n")
    for i, doc in enumerate(contexts, 1):
        print(f"Chunk {i}")
        print(doc)
        print("-" * 50)

context = "\n\n".join(contexts)

prompt = f"""
You are a networking expert.

Use ONLY the context below to answer the user's question.

If the answer cannot be found in the context,
reply with:

'I couldn't find that information in the provided documents.'

Context:
{context}

Question:
{question}

Answer:
"""

response = llm_client.chat.completions.create(
    model=LLM_MODEL,
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)

print("\nAnswer:\n")
print(response.choices[0].message.content)

