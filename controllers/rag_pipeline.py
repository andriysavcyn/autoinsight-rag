import os
import logging
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_classic.chains.conversational_retrieval.base import (
    ConversationalRetrievalChain,
)
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import (
    CrossEncoderReranker,
)
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")

load_dotenv()

def setup_rag_pipeline() -> ConversationalRetrievalChain:
    # Initializes and returns the Conversational RAG pipeline with Reranking and Memory.
    logger.info("Initializing the RAG pipeline...")

    if not os.getenv("GROQ_API_KEY"):
        logger.critical("GROQ_API_KEY is missing from the .env file!")
        raise ValueError(
            "API key not configured. Please add GROQ_API_KEY to your .env file."
        )

    if not os.path.exists(CHROMA_PATH):
        logger.warning(
            "Vector database not found! Did you forget to run vector_store.py?"
        )

    try:
        logger.info("Connecting to the Chroma vector database...")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        db = Chroma(
            persist_directory=CHROMA_PATH, embedding_function=embeddings
        )

        base_retriever = db.as_retriever(search_kwargs={"k": 10})

        logger.info("Loading the Cross-Encoder Reranker model...")
        cross_encoder_model = HuggingFaceCrossEncoder(
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        compressor = CrossEncoderReranker(model=cross_encoder_model, top_n=3)

        smart_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=base_retriever
        )

        logger.info("Connecting to Llama 3.3 via Groq API...")
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)

        template = """You are a professional mechanic and an AI assistant specifically for the Audi RS5. 
        Use the following pieces of context from the official Audi RS5 car manual to answer the user's question.
        If the context does not contain the answer, just say that you don't know. Do not make up information.

        Context from manual: {context}
        User Question: {question}
        Helpful Answer:"""

        QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

        logger.info("Building the Conversational Retrieval Chain...")
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=smart_retriever,
            combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT},
        )

        logger.info("RAG pipeline successfully initialized! 🎉")
        return qa_chain

    except Exception as e:
        logger.critical(
            f"Critical error during RAG pipeline initialization: {str(e)}"
        )
        raise

if __name__ == "__main__":
    try:
        logger.info("=== Testing RAG Pipeline ===")
        chain = setup_rag_pipeline()

        test_question = "What type of engine oil should I use for this car?"
        logger.info(f"Test Question: {test_question}")

        response = chain.invoke({"question": test_question, "chat_history": []})
        print(f"\n--- AI ANSWER ---\n{response['answer']}\n-----------------")

    except Exception as e:
        logger.error(f"Pipeline test failed: {e}")