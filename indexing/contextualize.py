from langchain_core.documents import Document
import tqdm

def contextualize(children, parents, llm):
    parent_map = {
        parent.metadata["parent_id"]: parent
        for parent in parents
    }

    contextualized = []

    for child in tqdm.tqdm(children, desc="Contextualizing chunks"):
        parent = parent_map[child.metadata["parent_id"]]

        prompt = f"""
        You are preparing a chunk for a retrieval system.

        Given the surrounding parent text and the child chunk,
        write a short 1-2 sentence context explaining where this
        chunk fits in the document.

        Do not summarize the chunk. Only provide enough context
        to make the chunk understandable when retrieved alone.

        PARENT:
        {parent.page_content}

        CHILD:
        {child.page_content}

        CONTEXT:
        """

        response = llm.invoke(prompt)

        context = response.content.strip()

        new_doc = Document(
            page_content=f"{context}\n\n{child.page_content}",
            metadata=child.metadata.copy()
        )

        contextualized.append(new_doc)

    return contextualized