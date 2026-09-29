import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_pdfs(documents_folder):
    documents = []
    for file in os.listdir(documents_folder):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(documents_folder, file))
            documents.extend(loader.load())
    return documents

def chunk_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

if __name__ == "__main__":
    # Simple test
    print("Loader and Chunker modules created.")
