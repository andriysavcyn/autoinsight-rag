import streamlit as st
import logging

def render_ui(chain):
    # This function is ONLY responsible for rendering the interface.
    custom_css = """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
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
        .stImage img {
            border-radius: 12px;
            box-shadow: 0px 4px 20px rgba(0, 0, 0, 0.5);
        }
    </style>
    <div class="autoinsight-footer">
        Developed by Andriy Savchyn | Lviv | 2026<br>
        <a href="https://github.com/andriysavcyn" target="_blank">GitHub</a> 
        <a href="https://www.linkedin.com/in/andriy-savchyn" target="_blank">LinkedIn</a>
    </div>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

    with st.sidebar:
        st.title("⚙️ Settings")
        st.markdown("This assistant uses RAG architecture to search through the **Audi RS5** manual.")
        st.divider()
        st.markdown("### 🛠 Technology stack:")
        st.markdown("- **LLM:** Llama 3.3\n- **Reranker:** Cross-Encoder\n- **DB:** ChromaDB\n- **UI:** Streamlit")
        st.divider()
        if st.button("🗑️ Clear conversation history", use_container_width=True):
            st.session_state.messages = [{"role": "assistant", "content": "Hello! I am your AutoInsight Assistant."}]
            st.rerun()

    st.title("🚘 AutoInsight: AI Assistant")
    st.caption("Ask anything about your vehicle specifications, maintenance, or troubleshooting.")

    car_image_url = "https://images.unsplash.com/photo-1614162692292-7ac56d7f7f1e?q=80&w=1000&auto=format&fit=crop"
    st.image(car_image_url, caption="Audi RS 5 Coupe (4.2L V8 FSI)", use_container_width=True)
    st.divider()

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
            with st.spinner("I am analyzing the manual..."):
                try:
                    response = chain.invoke({"question": prompt, "chat_history": chat_history})
                    answer = response['answer']
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error("Oops! A technical error occurred.")
                    logging.error(f"Generation error: {e}")