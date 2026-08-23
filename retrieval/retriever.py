from config import reranker, embeddings, TOP_K_DENSE, TOP_K_SPARSE, RRF_CONSTANT, TOP_K
from storage.vector_store import get_vectorstore
from storage.document_store import get_children
from hybrid_search import dense_search, build_bm25, sparse_search, reciprocal_rank_fusion
from reranker import rerank, expand_parents

def retrieve(query):
    vectorstore = get_vectorstore(embeddings)

    children = get_children()
    bm25 = build_bm25(children)

    dense = dense_search(
        query=query,
        vectorstore=vectorstore,
        k=TOP_K_DENSE
    )

    sparse = sparse_search(
        query=query,
        bm25=bm25,
        children=children,
        k=TOP_K_SPARSE
    )

    hybrid = reciprocal_rank_fusion(dense_results=dense, sparse_results=sparse, k=RRF_CONSTANT)

    top_results = rerank(
        query=query,
        documents=hybrid,
        reranker=reranker,
        k=TOP_K
    )

    context = expand_parents(top_results)

    return context