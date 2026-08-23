from rank_bm25 import BM25Okapi
# Remeber in main: from config import TOP_K_DENSE, TOP_K_SPARSE, RRF_CONSTANT

def dense_search(query, vectorstore, k):
    return vectorstore.similarity_search(query, k=k)

def build_bm25(children):
    tokenized_docs = [
        doc.page_content.lower().split()
        for doc in children
    ]

    bm25 = BM25Okapi(tokenized_docs)

    return bm25

def sparse_search(query, bm25, children, k):
    tokenized_query = query.lower().split()

    scores = bm25.get_scores(tokenized_query)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:k]

    return [children[i] for i in ranked_indices]

def reciprocal_rank_fusion(dense_results, sparse_results, k):
    scores = {}
    docs = {}

    for rank, doc in enumerate(dense_results, start=1):
        chunk_id = doc.metadata["chunk_id"]

        scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank)
        docs[chunk_id] = doc

    for rank, doc in enumerate(sparse_results, start=1):
        chunk_id = doc.metadata["chunk_id"]

        scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank)
        docs[chunk_id] = doc

    ranked_ids = sorted(
        scores,
        key=scores.get,
        reverse=True
    )

    return [docs[chunk_id] for chunk_id in ranked_ids]