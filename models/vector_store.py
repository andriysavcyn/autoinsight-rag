import logging
from typing import List
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from document_processor import load_and_split_pdf, FILE_PATH
import os

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")


def create_vector_db(chunks: List[Document], persist_directory: str) -> None:
    # Creates a ChromaDB vector database from the passed chunks and saves it to disk.
    if not chunks:
        logger.warning(
            "There are no text fragments to save to the database. The process has been aborted."
        )
        return

    logger.info("Initializing Embeddings Model (all-MiniLM-L6-v2)...")
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        logger.info(
            f"Creating and saving a vector database to a folder: {persist_directory}..."
        )

        db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_directory,
        )

        logger.info(
            "The vector database has been successfully created and saved! "
        )

    except Exception as e:
        logger.error(f"Error creating vector base: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        logger.info("=== Starting the data processing pipeline ===")

        docs = load_and_split_pdf(FILE_PATH)

        create_vector_db(docs, CHROMA_PATH)

        logger.info("=== The conveyor has successfully completed its work. ===")
    except Exception as e:
        logger.critical(f"The process was stopped due to a critical error: {e}")