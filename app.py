import os
import streamlit as st
from rag_gemini_bot import extract_text_from_pdfs, build_vector_store, generate_answer_with_gemini

# ✅ Set your Gemini API Key here
os.environ["GOOGLE_API_KEY"] = "AIzaSyBNEyIz0_wn2h__TxPRTNDHEdJM-KLs0TM"

st.set_page_config(page_title="📚 RAG Chatbot with Gemini", page_icon="🤖")
st.title("📖 RAG Chatbot using Gemini Pro")

# Load and process docs
with st.spinner("Loading documents..."):
    documents = extract_text_from_pdfs("documents")
    index, docs = build_vector_store(documents)

# Chat interface
st.markdown("Ask a question based on the uploaded PDFs:")

user_input = st.text_input("You:", key="input")

if user_input:
    with st.spinner("Generating response..."):
        answer = generate_answer_with_gemini(user_input, index, docs)
        st.markdown("### 🤖 Gemini says:")
        st.write(answer)
