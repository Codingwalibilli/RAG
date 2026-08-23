from langchain_community.document_loaders import (
    TextLoader,
    PyMuPDFLoader,
    UnstructuredMarkdownLoader,
    DirectoryLoader,
)
from langchain_core.documents import Document
from storage.document_store import get_document_sources

from collections import defaultdict

def load_documents(DATA_DIR):
    docs = []

    indexed_sources = set(get_document_sources())

    loaders = {
        "*.txt": (TextLoader, {"encoding": "utf-8"}),
        "*.pdf": (PyMuPDFLoader, {}),
        "*.md": (UnstructuredMarkdownLoader, {"encoding": "utf-8"}),
    }

    for pattern, (loader_cls, kwargs) in loaders.items():
        loader = DirectoryLoader(
            DATA_DIR,
            glob=pattern,
            loader_cls=loader_cls,
            loader_kwargs=kwargs,
            show_progress=True,
        )

        loaded_docs = loader.load()

        loaded_docs = [
            doc for doc in loaded_docs
            if doc.metadata["source"] not in indexed_sources
        ]

        if pattern == "*.pdf":
            pdfs = defaultdict(list)

            for page in loaded_docs:
                pdfs[page.metadata["source"]].append(page)

            for source, pages in pdfs.items():
                full_text = "\n\n".join(
                    page.page_content for page in pages
                )

                docs.append(
                    Document(
                        page_content=full_text,
                        metadata={
                            "source": source,
                            "file_type": "pdf",
                            "num_pages": len(pages),
                        },
                    )
                )

        else:
            file_type = pattern.replace("*.", "")

            for doc in loaded_docs:
                doc.metadata["file_type"] = file_type

            docs.extend(loaded_docs)

    return docs