from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

llm = ChatOllama(
            model="qwen2.5:3b",
            temperature=0
        )
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)
reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L6-v2"
)

DB_PATH = "data/documents.db"
DATA_DIR = "data"
PERSIST_DIRECTORY = "data/chroma"
COLLECTION_NAME = "rag_documents"
TOP_K_SPARSE = 20
TOP_K_DENSE = 20
RRF_CONSTANT = 60
TOP_K = 5