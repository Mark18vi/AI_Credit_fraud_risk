from ingestion.loader import load_pdfs, chunk_documents
from ingestion.vector_store import get_embeddings_model, create_vector_store
import os

def run_ingestion_pipeline(docs_folder="documents"):
    print("Starting ingestion pipeline...")
    
    # 1. Load PDFs
    docs = load_pdfs(docs_folder)
    print(f"Loaded {len(docs)} pages from PDFs.")
    
    # 2. Chunking
    chunks = chunk_documents(docs)
    print(f"Split into {len(chunks)} chunks.")
    
    # 3. Embeddings & Vector Store
    embeddings = get_embeddings_model()
    vector_store = create_vector_store(chunks, embeddings)
    
    print("Ingestion complete. Vector store created at ./chroma_db")
    return vector_store

if __name__ == "__main__":
    # This would require OPENAI_API_KEY to be set
    try:
        run_ingestion_pipeline()
    except Exception as e:
        print(f"Error: {e}. Ensure OPENAI_API_KEY is set in environment.")
