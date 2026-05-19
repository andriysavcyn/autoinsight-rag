import os
import logging
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FILE_PATH = os.path.join(BASE_DIR, "data", "audi_rs5_manual.pdf")

def load_and_split_pdf(file_path: str) -> List[Document]:
    # Loads a PDF file and splits it into overlapping chunks.
    logger.info(f"Attempt to download document: {file_path}")

    if not os.path.exists(file_path):
        logger.error(f"File not found at path: {file_path}")
        raise FileNotFoundError(
            f"File '{file_path}' does not exist. Make sure the folder 'data' exists."
        )

    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        if not documents:
            logger.warning(
                f"The file {file_path} was downloaded, but it is empty or the text is not recognized!"
            )
            return []

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )

        chunks = text_splitter.split_documents(documents)
        logger.info(f"Successful! Document split into {len(chunks)} chunks.")

        return chunks

    except Exception as e:
        logger.error(f"Critical error while processing PDF: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        docs = load_and_split_pdf(FILE_PATH)

        if docs and len(docs) > 10:
            print("\n--- Example of chopped text (Chunk #11) ---")
            print(docs[10].page_content)
        elif docs:
            print("\n--- Example of chopped text (Chunk #1) ---")
            print(docs[0].page_content)

    except Exception as e:
        print(f"\nThe process was stopped due to an error: {e}")