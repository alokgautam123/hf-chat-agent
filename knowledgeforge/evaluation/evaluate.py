from dataset import EVALUATION_DATASET

from config import MAX_DISTANCE
from embeddings import create_embeddings
from reranker import rerank
from vectordb import search


TOP_K = 5


def is_relevant(meta, expected_source, expected_pages):
    return (
        meta.get("source") == expected_source
        and meta.get("page") in expected_pages
    )


def reciprocal_rank(results, expected_source, expected_pages):
    for rank, result in enumerate(results, start=1):
        meta = result[1]

        if is_relevant(meta, expected_source, expected_pages):
            return 1.0 / rank

    return 0.0


def evaluate():
    vector_hits = {1: 0, 3: 0, 5: 0}
    reranked_hits = {1: 0, 3: 0, 5: 0}

    vector_rr = []
    reranked_rr = []

    for test in EVALUATION_DATASET:
        question = test["question"]
        expected_source = test["expected_source"]
        expected_pages = test["expected_pages"]

        query = [
            {
                "text": question,
                "metadata": {},
            }
        ]

        query_embedding = create_embeddings(query)

        results = search(
            query_embedding=query_embedding,
            top_k=TOP_K,
        )

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]
        chunk_ids = results["ids"][0]

        vector_results = []

        for doc, meta, distance, chunk_id in zip(
            documents,
            metadatas,
            distances,
            chunk_ids,
        ):
            if distance <= MAX_DISTANCE:
                vector_results.append(
                    (doc, meta, distance, chunk_id)
                )

        reranked_results = rerank(
            question,
            vector_results,
        )

        for k in [1, 3, 5]:
            if any(
                is_relevant(
                    result[1],
                    expected_source,
                    expected_pages,
                )
                for result in vector_results[:k]
            ):
                vector_hits[k] += 1

            if any(
                is_relevant(
                    result[1],
                    expected_source,
                    expected_pages,
                )
                for result in reranked_results[:k]
            ):
                reranked_hits[k] += 1

        vector_rr.append(
            reciprocal_rank(
                vector_results,
                expected_source,
                expected_pages,
            )
        )

        reranked_rr.append(
            reciprocal_rank(
                reranked_results,
                expected_source,
                expected_pages,
            )
        )

        print(f"\nQuestion: {question}")

        print("Vector pages:",
              [r[1].get("page") for r in vector_results])

        print("Reranked pages:",
              [r[1].get("page") for r in reranked_results])

    total = len(EVALUATION_DATASET)

    print("\n\nRetrieval Evaluation")
    print("--------------------")
    print(f"Questions evaluated: {total}")

    print("\nVector Retrieval")
    for k in [1, 3, 5]:
        print(f"Hit@{k}: {vector_hits[k] / total:.2%}")

    print(f"MRR:   {sum(vector_rr) / total:.3f}")

    print("\nVector + Reranker")
    for k in [1, 3, 5]:
        print(f"Hit@{k}: {reranked_hits[k] / total:.2%}")

    print(f"MRR:   {sum(reranked_rr) / total:.3f}")


if __name__ == "__main__":
    evaluate()
