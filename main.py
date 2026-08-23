from indexing.indexer import index
from retrieval.retriever import retrieve
from config import llm

def generate_answer(query, context):
    context_text = "\n\n---\n\n".join(
        doc.page_content for doc in context
    )

    prompt = f"""
        You are a retrieval-augmented assistant.

        Answer the user's question using only the provided context.
        If the answer is not present in the context, say you don't know.

        QUESTION:
        {query}

        CONTEXT:
        {context_text}

        ANSWER:
        """

    response = llm.invoke(prompt)

    return response.content

def main():
    index()

    print("All files indexed")
    print("Type \'exit\' to quit")

    while True:
        query = input("You: ").strip()

        if query.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        if not query:
            continue

        context = retrieve(query)

        if not context:
            print("\nRAG: I couldn't find relevant context.\n")
            continue

        answer = generate_answer(
            query=query,
            context=context
        )

        print(f"\nRAG: {answer}\n")

if __name__ == "__main__":
    main()