import os
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

def get_embeddings_model():
    # Note: API key will be handled via environment variables or .env
    return OpenAIEmbeddings(model="text-embedding-3-small")

def create_vector_store(chunks, embeddings, persist_directory="./chroma_db"):
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    return vector_store

def load_vector_store(embeddings, persist_directory="./chroma_db"):
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )

if __name__ == "__main__":
    print("Embeddings and Vector Store modules created.")
