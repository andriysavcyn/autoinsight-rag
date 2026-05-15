import streamlit as st
from controllers.rag_pipeline import setup_rag_pipeline

st.set_page_config(page_title="AutoInsight RAG", page_icon="🚗", layout="centered")
st.title("AutoInsight: AI Auction Assistant")
st.caption("Ask anything about your vehicle specifications, maintenance, or troubleshooting.")

@st.cache_resource
def load_ai_backend():
    return setup_rag_pipeline()

chain = load_ai_backend()

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I am your AutoInsight Assistant. What do you want to know about the car?"}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 4. Логіка спілкування (поле вводу)
if prompt := st.chat_input("Enter your question here (e.g., How to change a tire?)"):
    
    # Показуємо запит користувача
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Генеруємо та показуємо відповідь AI
    with st.chat_message("assistant"):
        with st.spinner("Searching the manual..."):
            response = chain.invoke(prompt)
            answer = response['result']
            st.markdown(answer)
            
    # Зберігаємо відповідь в історію
    st.session_state.messages.append({"role": "assistant", "content": answer})