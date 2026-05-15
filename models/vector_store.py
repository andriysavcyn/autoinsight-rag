from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
from document_processor import load_and_split_pdf, FILE_PATH
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")

def create_vector_db():
    print("1. Get splitted fragments of texts...")
    chunks = load_and_split_pdf(FILE_PATH)

    print("2. Load the model for embeddings...")
    embeddings = HuggingFaceBgeEmbeddings(model_name="all-MiniLM-L6-v2")

    print('3. Build vector database ChromaDB...')
    os.makedirs(CHROMA_PATH, exist_ok=True)
    db = Chroma.from_documents(
        documents=chunks,
        embedding= embeddings,
        persist_directory=CHROMA_PATH
    )

    print(f"Database was successfully saved. Look for the folder '{CHROMA_PATH}' in your project!")
    return db

if __name__ == "__main__":
    create_vector_db()