import os

from openai import OpenAI

from config import LLM_MODEL
from config import TOP_K_RESULTS

from embeddings import create_embeddings
from vectordb import search

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

contexts = results["documents"][0]

context = "\n\n".join(contexts)

prompt = f"""
You are an expert assistant.

Use ONLY the context below.

If the answer is not present, reply:

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
