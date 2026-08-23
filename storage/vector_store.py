from langchain_chroma import Chroma
from config import PERSIST_DIRECTORY,COLLECTION_NAME

def get_vectorstore(embeddings):
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIRECTORY
    )

def store_vectors(documents, embeddings):
    vectorstore = get_vectorstore(embeddings)

    ids = [doc.metadata["chunk_id"] for doc in documents]

    vectorstore.add_documents(
        documents=documents,
        ids=ids
    )

    return vectorstore