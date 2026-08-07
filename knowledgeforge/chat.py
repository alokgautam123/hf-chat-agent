import os

from openai import OpenAI

from config import LLM_MODEL
from config import TOP_K_RESULTS

from embeddings import create_embeddings
from vectordb import search
from pprint import pprint
from config import MAX_DISTANCE
from prompts import build_qa_prompt
from reranker import rerank


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

filtered_results = []

for doc, meta, distance, chunk_id in zip(
    documents,
    metadata,
    distances,
    chunk_ids,
):
    if distance <= MAX_DISTANCE:
        filtered_results.append(
            (doc, meta, distance, chunk_id)
        )

if not filtered_results:
    print("\nAnswer:\n")
    print("I couldn't find relevant information in the indexed documents.")
    exit()

filtered_results = rerank(
    question,
    filtered_results
)

filtered_results = filtered_results[:TOP_K_RESULTS]

context_parts = []

for doc, meta, distance, chunk_id, score in filtered_results:
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

Distance:
{distance}

Reranker Score:
{score}
"""
    )

context = "\n\n".join(context_parts)

prompt = build_qa_prompt(
    question=question,
    context=context
)

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
