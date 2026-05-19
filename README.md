# 🚘 AutoInsight: RAG Assistant for Audi RS5

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-Integration-121212?style=flat-square)
![PyTorch](https://img.shields.io/badge/PyTorch-CPU_Only-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![ChromaDB](https://img.shields.io/badge/Vector_DB-ChromaDB-FF6F00?style=flat-square)
![Llama 3.3](https://img.shields.io/badge/LLM-Llama_3.3_70B-0466C8?style=flat-square)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat-square&logo=docker&logoColor=white)

## 📌 What is this?
AutoInsight is a conversational AI assistant built to answer technical, diagnostic, and maintenance questions about the Audi RS5. 

The main goal of this project was to eliminate LLM hallucinations. Instead of relying on the model's baseline training data, the system is strictly grounded in the official vehicle manual using an Advanced RAG architecture with a two-stage retrieval pipeline.

## 🏗️ Architecture & Technical Decisions

I structured the project using the **MVC (Model-View-Controller)** pattern. Keeping the ML logic decoupled from the UI makes it much easier to scale, test, or swap out the frontend later.

### 1. Data Processing (Model)
* **Chunking:** PDF manuals are parsed and split using `RecursiveCharacterTextSplitter`. I set the overlap to 200 characters to ensure we don't lose semantic context between page breaks.
* **Embeddings & Storage:** Text chunks are vectorized using `all-MiniLM-L6-v2` and stored locally in **ChromaDB**.

### 2. The RAG Pipeline (Controller)
Standard vector search often returns loosely related chunks. To fix this, I implemented a two-stage retrieval:
1. **Base Retrieval:** Fetches the top-K relevant documents from ChromaDB.
2. **Cross-Encoder Reranking:** A `ms-marco-MiniLM-L-6-v2` model re-evaluates and scores these chunks. Only the highest-scoring context is passed to the LLM.
3. **Generation:** Llama 3.3 (70B) via the Groq API generates the final response, factoring in the chat history managed by LangChain.

### 3. UI (View)
* Built with **Streamlit** for rapid prototyping. It handles session state and chat history rendering independently from the core pipeline.

## 🚀 How to Run (Docker)

The app is fully containerized. 
*Note: The `Dockerfile` is explicitly configured to pull the CPU-only version of PyTorch. This keeps the image size manageable and prevents memory limit crashes (EOF errors) during the build.*

### 1. Clone & Setup
```bash
git clone [https://github.com/andriysavcyn/autoinsight-rag.git](https://github.com/andriysavcyn/autoinsight-rag.git)
cd autoinsight-rag
```

Create a `.env` file in the root directory with your Groq API key:
```env
GROQ_API_KEY=gsk_your_api_key_here
```

### 2. Build & Start
```bash
docker build -t autoinsight-app .
docker run -p 8501:8501 --env-file .env autoinsight-app
```
Then open `http://localhost:8501` in your browser.

## 💻 Local Development
If you want to run it without Docker:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt

# Make sure your PDF is in data/ and generate the vector DB first:
python document_processor.py

streamlit run main.py
```