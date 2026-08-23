from chunking import chunk_documents
from document_loader import load_documents
from contextualize import contextualize
from storage.document_store import store_documents,store_parents,store_children
from storage.vector_store import store_vectors
from config import llm, embeddings, DATA_DIR

def index():
    docs = load_documents(DATA_DIR)

    store_documents(documents=docs)

    parents, children = chunk_documents(docs=docs)

    contextualized_children = contextualize(
        parents=parents,
        children=children,
        llm=llm
    )

    store_parents(parents=parents)
    store_children(children=children)

    store_vectors(
        documents=contextualized_children,
        embeddings=embeddings
    )