import os

from openai import OpenAI

from config import LLM_MODEL
from config import TOP_K_RESULTS

from embeddings import create_embeddings
from vectordb import search
from pprint import pprint

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_TOKEN"],
)

question = input("Ask a question: ")

query = [
    {
        "text": question,
        "metadata": {}
    }
]

query_embedding = create_embeddings(query)

results = search(
    query_embedding=query_embedding,
    top_k=TOP_K_RESULTS,
)

#print("\nRetrieved Results:\n")
#pprint(results)

documents = results["documents"][0]
metadata = results["metadatas"][0]
distances = results["distances"][0]
chunk_ids = results["ids"][0]

context_parts = []

for doc, meta, distance, chunk_id in zip(
    documents,
    metadata,
    distances,
    chunk_ids
):
    context_parts.append(
        f"""
Source:
{meta.get('source')}

Page:
{meta.get('page', 'N/A')}

Chunk:
{chunk_id}

Content:
{doc}

Similarity:
{distance}
"""
    )

context = "\n\n".join(context_parts)

prompt = f"""
You are an expert assistant.

Answer using ONLY the provided context.

For every answer, include citations in this format:

Sources:
- <filename>
- Page <number>

If information is not present, reply:

I couldn't find that information in the provided documents.

Context:

{context}

Question:

{question}

Answer:
"""

response = client.chat.completions.create(
    model=LLM_MODEL,
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ],
)

print("\nAnswer:\n")
print(response.choices[0].message.content)
