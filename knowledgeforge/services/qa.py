import os
import re

from openai import OpenAI

from config import LLM_MODEL, MAX_DISTANCE, TOP_K_RESULTS
from embeddings import create_embeddings
from prompts import build_qa_prompt
from reranker import rerank
from vectordb import search


NO_RELEVANT_CONTEXT_MESSAGE = (
    "I couldn't find relevant information in the indexed documents."
)
INSUFFICIENT_CONTEXT_MESSAGE = (
    "I couldn't find that information in the provided documents."
)


def _create_citations(answer, results):
    sources_match = re.search(r"(?im)^Sources:\s*$", answer)

    if not sources_match:
        return []

    available_citations = {
        (metadata.get("source"), metadata.get("page"))
        for _, metadata, _, _, _ in results
        if metadata.get("source")
    }
    available_sources = {
        source.casefold(): source
        for source, _ in available_citations
    }

    citations = []
    seen = set()
    pending_source = None

    for line in answer[sources_match.end():].splitlines():
        normalized_line = line.strip().lstrip("-*• ").strip()

        if not normalized_line:
            continue

        matching_sources = [
            source
            for normalized_source, source in available_sources.items()
            if normalized_source in normalized_line.casefold()
        ]

        if matching_sources:
            pending_source = max(matching_sources, key=len)

        page_match = re.search(r"(?i)\bPage\s+(\d+)\b", normalized_line)

        if pending_source and page_match:
            citation_key = (pending_source, int(page_match.group(1)))

            if citation_key in available_citations and citation_key not in seen:
                seen.add(citation_key)
                citations.append(
                    {
                        "source": pending_source,
                        "page": citation_key[1],
                    }
                )

            pending_source = None

    if pending_source:
        citation_key = (pending_source, None)

        if citation_key in available_citations and citation_key not in seen:
            citations.append(
                {
                    "source": pending_source,
                    "page": None,
                }
            )

    return citations


def answer_question(question):
    """Answer a question using the indexed KnowledgeForge documents."""
    query = [
        {
            "text": question,
            "metadata": {},
        }
    ]

    query_embedding = create_embeddings(query)

    results = search(
        query_embedding=query_embedding,
        top_k=TOP_K_RESULTS,
    )

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
        return {
            "answer": NO_RELEVANT_CONTEXT_MESSAGE,
            "answered": False,
            "citations": [],
        }

    filtered_results = rerank(
        question,
        filtered_results,
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
        context=context,
    )

    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=os.environ["HF_TOKEN"],
    )
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    answer = response.choices[0].message.content.strip()

    if answer == "INSUFFICIENT_CONTEXT":
        return {
            "answer": INSUFFICIENT_CONTEXT_MESSAGE,
            "answered": False,
            "citations": [],
        }

    return {
        "answer": answer,
        "answered": True,
        "citations": _create_citations(answer, filtered_results),
    }
