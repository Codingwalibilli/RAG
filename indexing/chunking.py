from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
import tqdm
from uuid import uuid4

def chunk_documents(docs):
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200
    )

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=40
    )

    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "h1"),
            ("##", "h2"),
            ("###", "h3"),
        ],
        strip_headers=False,
    )

    parents = []
    children = []

    for doc in tqdm.tqdm(docs, desc="Chunking Documents"):
        file_type = doc.metadata.get("file_type")

        if file_type == "md":
            sections = header_splitter.split_text(doc.page_content)

            parent_chunks = []

            for section in tqdm.tqdm(sections, desc="Chunking Markdowns", leave = False):
                section.metadata.update(doc.metadata)

                if len(section.page_content) > 2000 * 1.2:
                    parent_chunks.extend(
                        parent_splitter.split_documents([section])
                    )
                else:
                    parent_chunks.append(section)

        else:
            parent_chunks = parent_splitter.split_documents([doc])

        for parent in tqdm.tqdm(parent_chunks, desc= "Mergeing Chunks", leave = False):
            parent_id = str(uuid4())

            parent.metadata["parent_id"] = parent_id
            parents.append(parent)

            child_chunks = child_splitter.split_documents([parent])

            for i,child in enumerate(child_chunks):
                child.metadata["parent_id"] = parent_id
                child.metadata["chunk_id"] = f"{parent_id}:child:{i}"
                children.append(child)

    return parents, children