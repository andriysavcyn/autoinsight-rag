import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate 

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")

load_dotenv()

def setup_rag_pipeline():
    print("1. Connect to database of car...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

    retriever = db.as_retriever(search_kwargs={"k": 5})

    print("2. Launched LLM (Mixtral 8x7b via Groq)...")
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.1
    )

    print("3. Adjust prompt (Context Injection)...")
    # ОСЬ ТУТ МАГІЯ: Ми жорстко задаємо контекст, що це мануал саме від Audi RS5
    template = """You are a professional mechanic and an AI assistant specifically for the Audi RS5. 
    Use the following pieces of context from the official Audi RS5 car manual to answer the user's question.
    If the context does not contain the answer, just say that you don't know. Do not make up information.

    Context from manual: {context}

    User Question: {question}

    Helpful Answer:"""
    
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    print("4. Combine base and LLM in RAG-chain...")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT} # Підключаємо нашу інструкцію
    )
    return qa_chain

if __name__ == "__main__":
    chain = setup_rag_pipeline()

    question = "What type of engine oil should I use for this car?"
    print(f"\nRequest: {question}")
    
    print("Generate an answer...\n")
    response = chain.invoke(question)
    
    print("=== RESPONSE AUTOINSIGHT RAG ===")
    print(response['result'])