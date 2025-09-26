import streamlit as st
import os
import requests
import tempfile
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings

# -----------------------------
# CONFIG
# -----------------------------
os.environ["GROQ_API_KEY"] = "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"
  # Or set as env variable

# Initialize LLM
llm = ChatGroq(model="llama-3.1-8b-instant")

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.set_page_config(page_title="📄 RAG PDF Chat", page_icon="🤖", layout="centered")
st.title("📄 RAG PDF Chat")
st.write("Upload a **PDF file** or paste a **PDF URL**, then ask questions about it below.")

# User Inputs
pdf_url = st.text_input("🔗 Paste PDF URL (optional):")
uploaded_file = st.file_uploader("📂 Or upload a PDF file", type=["pdf"])
user_query = st.text_input("💬 Ask a question about the PDF:")

# State control
if not pdf_url and not uploaded_file:
    st.info("Please either upload a PDF or paste a PDF URL to start.")
else:
    try:
        # -----------------------------
        # LOAD PDF (URL OR FILE)
        # -----------------------------
        if pdf_url:
            st.info("Fetching PDF from URL...")
            response = requests.get(pdf_url)
            response.raise_for_status()

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
                tmp_pdf.write(response.content)
                pdf_path = tmp_pdf.name

        elif uploaded_file:
            st.info("Processing uploaded PDF...")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
                tmp_pdf.write(uploaded_file.read())
                pdf_path = tmp_pdf.name

        # -----------------------------
        # PROCESS DOCUMENT
        # -----------------------------
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectordb = Chroma.from_documents(chunks, embedding=embeddings)

        retriever = vectordb.as_retriever()
        qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)

        st.success("✅ PDF processed successfully! You can now ask a question above.")

        if user_query:
            with st.spinner("Thinking..."):
                result = qa_chain.run(user_query)
            st.markdown("### 📌 Answer:")
            st.write(result)

    except Exception as e:
        st.error(f"⚠️ Error while processing PDF: {str(e)}")
