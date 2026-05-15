from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

FILE_PATH = "data/audi_rs5_manual.pdf"

def load_and_split_pdf(file_path):
    print(f"Load the document: {file_path}...")

    loader = PyPDFLoader(file_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200
    )

    chunks = text_splitter.split_documents(documents)
    print(f"Ready! The document was split on {len(chunks)} text`s fragments.")
    return chunks

if __name__ == "__main__":
    docs = load_and_split_pdf(FILE_PATH)

    if len(docs) > 10:
        print("\n the example of cutted text")
        print(docs[10].page_content)
