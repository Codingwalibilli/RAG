from storage.document_store import get_parents

def rerank(query, documents, reranker, k):
    pairs = [
        [query, doc.page_content]
        for doc in documents
    ]

    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(documents, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [doc for doc, score in ranked[:k]]


def expand_parents(children):
    parent_ids = list({
        child.metadata["parent_id"]
        for child in children
    })

    return get_parents(parent_ids)