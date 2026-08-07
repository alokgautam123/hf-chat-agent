from sentence_transformers import CrossEncoder

_model = None


def get_reranker():
    global _model

    if _model is None:
        print("Loading reranker model...")

        _model = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

    return _model


def rerank(question, results):
    model = get_reranker()

    pairs = []

    for doc, meta, distance, chunk_id in results:
        pairs.append((question, doc))

    scores = model.predict(pairs)

    reranked = []

    for result, score in zip(results, scores):
        reranked.append(
            (*result, score)
        )

    reranked.sort(
        key=lambda x: x[4],
        reverse=True
    )

    return reranked
