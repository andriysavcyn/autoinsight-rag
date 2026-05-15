import streamlit as st
import logging
from controllers.rag_pipeline import setup_rag_pipeline

# 1. Налаштування сторінки (обов'язково ПЕРШИЙ рядок коду)
st.set_page_config(
    page_title="AutoInsight AI", 
    page_icon="🚘", 
    layout="centered",
    initial_sidebar_state="expanded"
)

# 2. Кастомний CSS: ховаємо дефолтний футер Streamlit і робимо свій професійний
custom_css = """
<style>
    /* Ховаємо стандартне меню (три крапки) та футер */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Наш власний закріплений футер */
    .autoinsight-footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: transparent;
        color: #6c757d;
        text-align: center;
        padding: 10px;
        font-size: 13px;
        z-index: 100;
        border-top: 1px solid rgba(255,255,255,0.1);
    }
    /* Заокруглення для фото машини */
    .stImage img {
        border-radius: 12px;
        box-shadow: 0px 4px 20px rgba(0, 0, 0, 0.5);
    }
</style>
<div class="autoinsight-footer">
    Powered by <b>Llama 3.3</b> & <b>LangChain</b> | Developed for AutoInsight
</div>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# 3. БSidebar
with st.sidebar:
    st.title("⚙️ Settings")
    st.markdown("This assistant uses the RAG architecture to search the official **Audi RS5** manual.")
    
    st.divider()
    st.markdown("### 🛠 Technology stack:")
    st.markdown("- **LLM:** Llama 3.3 (70B)\n- **Reranker:** Cross-Encoder\n- **DB:** ChromaDB\n- **UI:** Streamlit")
    st.divider()
    
    # Кнопка очищення пам'яті
    if st.button("🗑️ Clear chat history", use_container_width=True):
        st.session_state.messages = [{"role": "assistant", "content": "Hello! I am your AutoInsight Assistant. What do you want to know about the Audi RS5?"}]
        st.rerun()

# 4. Головний екран: Заголовок
st.title("🚘 AutoInsight: AI Assistant")
st.caption("Ask anything about your vehicle specifications, maintenance, or troubleshooting.")

# === НОВИЙ БЛОК: Графічне відображення машини ===
# Використовуємо якісне фото Audi RS5 з відкритих джерел (Unsplash)
car_image_url = "images/audi_rs5.jpeg"
st.image(
    car_image_url, 
    caption="Audi RS 5 Coupe (4.2L V8 FSI)", 
    use_container_width=True
)
st.divider() # Акуратна лінія під фото
# ===============================================

# 5. Безпечне завантаження бекенду з обробкою подій (Toasts)
@st.cache_resource(show_spinner=False)
def load_ai_backend():
    try:
        pipeline = setup_rag_pipeline()
        return pipeline
    except Exception as e:
        st.error(f"AI initialization error. Please check your console or API settings. Details: {e}")
        return None

with st.spinner("Running neural networks..."):
    chain = load_ai_backend()

if chain is None:
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I am your AutoInsight Assistant. What do you want to know about the Audi RS5?"}]
    st.toast('The system is successfully connected!', icon='✅')

for msg in st.session_state.messages:
    avatar = "👤" if msg["role"] == "user" else "🚘"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

if prompt := st.chat_input("Enter your question here..."):

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    chat_history = []
    user_msg = ""
    for msg in st.session_state.messages[1:-1]: 
        if msg["role"] == "user":
            user_msg = msg["content"]
        elif msg["role"] == "assistant" and user_msg:
            chat_history.append((user_msg, msg["content"]))
            user_msg = ""

    with st.chat_message("assistant", avatar="🚘"):
        with st.spinner("I analyze the manual and generate an answer..."):
            try:
                response = chain.invoke({
                    "question": prompt, 
                    "chat_history": chat_history
                })
                answer = response['answer']
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                error_msg = "Oops! There was a technical error communicating with the AI. Please check your database or API connection."
                st.error(error_msg)
                logging.error(f"Error generating response: {e}")