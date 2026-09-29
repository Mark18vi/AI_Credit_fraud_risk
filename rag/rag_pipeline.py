from rag.retriever import get_retriever
from rag.prompt import get_rag_prompt

def get_relevant_context(query):
    retriever = get_retriever()
    # In newer LangChain versions, get_relevant_documents is deprecated in favor of invoke
    docs = retriever.invoke(query)
    
    # Combine content of retrieved docs
    context = "\n\n".join([doc.page_content for doc in docs])
    return context, docs

if __name__ == "__main__":
    print("RAG pipeline logic created.")
