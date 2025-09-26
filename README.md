# RAG-Powered-AI-Chatbot-for-Converting-Question-Answering-over-PDFs-and-Online-Documents
# AIM
Allow any user to upload a PDF (or paste a PDF URL) and ask natural-language questions about its contents. The app retrieves relevant passages using semantic search (vector embeddings + Chroma) and generates accurate, context-aware answers via a fast Groq LLM (ChatGroq).
# Tech stack & libraries (what to list)
-> Frontend: Streamlit (UI, file uploader, text inputs).
->LLM provider: Groq via LangChain integration (ChatGroq) — requires GROQ_API_KEY. 
->Document loader: PyPDFLoader from langchain_community. 
->Text splitting: RecursiveCharacterTextSplitter (chunking & overlap). 
->Embeddings: HuggingFaceEmbeddings using a Sentence-Transformer (e.g., all-MiniLM-L6-v2). 
->Vector DB: Chroma (LangChain integration: langchain-chroma) + chromadb backend. 
->Orchestration: LangChain RetrievalQA chain. 

# Architecture & flow (high level)
1, User uploads a PDF or pastes a PDF URL (Streamlit UI). 
2. App saves PDF to a temporary file and loads it using PyPDFLoader. 
3. The text is split into overlapping chunks using RecursiveCharacterTextSplitter (chunk_size & overlap to preserve context). 
4. Each chunk is converted to an embedding (HuggingFace / sentence-transformer). 
5. Embeddings are stored in a Chroma vector store. 
6. For each user query, the vector store retrieves the most relevant chunks, and the LLM (ChatGroq via LangChain) generates an answer using the retrieved context (RetrievalQA). 

# Set your API key
   export GROQ_API_KEY="your_groq_api_key_here"    # Linux / macOS
   setx GROQ_API_KEY "your_groq_api_key_here"     # Windows

#    Run the app
    streamlit run app.py

# How It Works
1. Upload a PDF or Paste a PDF URL in the Streamlit interface.
2. The app:
  * Downloads (or reads) the PDF,
  * Splits it into overlapping chunks for context preservation,
  * Converts chunks into vector embeddings,
  * Stores them in a Chroma vector database.
3. Ask a Question → The retriever fetches the most relevant chunks and passes them to the Groq LLM.
4. LLM generates an answer based on the retrieved context and displays it in the UI.

# Key Features
* PDF Upload + URL Support (flexible input options)
* Semantic Search with HuggingFace Embeddings
* Retrieval-Augmented Generation for accurate answers
* Interactive Chat Interface built with Streamlit
* Easy to Deploy on Streamlit Cloud, Docker, or Heroku

