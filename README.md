# 🚘 AutoInsight: Advanced RAG System for Vehicle Diagnostics

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-Integration-121212?style=flat-square)
![PyTorch](https://img.shields.io/badge/PyTorch-CPU_Optimized-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![ChromaDB](https://img.shields.io/badge/Vector_DB-ChromaDB-FF6F00?style=flat-square)
![Llama 3.3](https://img.shields.io/badge/LLM-Llama_3.3_70B-0466C8?style=flat-square)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat-square&logo=docker&logoColor=white)

## 📌 Project Overview
**AutoInsight** is an intelligent, containerized conversational agent designed to provide precise, context-aware diagnostics and maintenance information for the **Audi RS5**. 

To mitigate LLM hallucinations and ensure mechanical accuracy, the system implements an **Advanced Retrieval-Augmented Generation (RAG)** pipeline. It grounds all responses strictly in the official vehicle manual, utilizing a two-stage retrieval process with a Cross-Encoder reranker.

## 🧠 System Architecture

The application is structured following the **Model-View-Controller (MVC)** architectural pattern to ensure separation of concerns, scalability, and maintainability.

### 1. Data Layer (Model & Vector Store)
* **Document Processing:** PDF parsing via `PyPDFLoader` with `RecursiveCharacterTextSplitter` (chunk size: 1000, overlap: 200) to maintain semantic context boundaries.
* **Embeddings:** HuggingFace `all-MiniLM-L6-v2` generates dense vector representations.
* **Storage:** Persisted locally using **ChromaDB** for efficient similarity search.

### 2. Logic Layer (Controller / RAG Pipeline)
* **Initial Retrieval:** Contextual compression retriever performs a broad vector similarity search.
* **Reranking:** A HuggingFace **Cross-Encoder** (`ms-marco-MiniLM-L-6-v2`) re-evaluates and scores the initial retrieved chunks, passing only the highest-relevance context to the LLM.
* **Memory Management:** LangChain handles conversational memory, dynamically injecting chat history for context-aware follow-up queries.
* **Generation:** Powered by **Llama 3.3 (70B)** via the Groq API for rapid, high-fidelity inference.

### 3. Presentation Layer (View)
* **Interface:** A reactive, dark-themed UI built with **Streamlit**.
* **Features:** Session state management, isolated UI rendering logic (`chat_interface.py`), and robust error handling.

## 🛠️ Tech Stack & MLOps
* **Core:** Python 3.11
* **AI/NLP:** LangChain, Sentence-Transformers, PyTorch (CPU-optimized for lightweight inference)
* **Infrastructure:** Docker (Multi-stage dependency resolution)

## 🚀 Quick Start (Docker Deployment)

The application is fully containerized. The `Dockerfile` is highly optimized to pull lightweight CPU versions of PyTorch, preventing bloat and memory issues during the build process.

### 1. Clone the Repository
```bash
git clone [https://github.com/andriysavcyn/autoinsight-rag.git](https://github.com/andriysavcyn/autoinsight-rag.git)
cd autoinsight-rag

### 2. Configure Environment
Create a .env file in the root directory and add your Groq API key:
GROQ_API_KEY=gsk_your_api_key_here
### 3. Configure Environment
# Build the Docker image (utilizes cached layers for faster rebuilds)
docker build -t autoinsight-app .

# Run the container (exposes port 8501 and mounts the environment variables)
docker run -p 8501:8501 --env-file .env autoinsight-app

Access the application at: http://localhost:8501

Local Development (Without Docker)
If you prefer running the application in a local virtual environment:

# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate the Chroma vector database (Ensure the PDF is in the data/ folder)
python document_processor.py

# 4. Launch the application
streamlit run main.py