from ingestion.vector_store import get_embeddings_model, load_vector_store

def get_retriever():
    embeddings = get_embeddings_model()
    vector_store = load_vector_store(embeddings)
    # Using similarity search as retriever
    return vector_store.as_retriever(search_kwargs={"k": 3})

if __name__ == "__main__":
    print("Retriever module created.")
