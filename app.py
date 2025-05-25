# Directory structure:
# nitj_chatbot/
# ├── app.py
# ├── chains.py
# ├── config.py
# ├── data_loader.py
# ├── embeddings.py
# ├── offline_model.py
# ├── utils.py
# ├── vector_store.py
# └── .env

# Below is a complete runnable version of the project broken into modules

# --- config.py ---
import os
from dotenv import load_dotenv

load_dotenv()

USE_OFFLINE = os.getenv("USE_OFFLINE", "False") == "True"
ONLINE_MODEL = os.getenv("ONLINE_MODEL", "gemini-1.5-flash")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/embedding-001")

# --- utils.py ---
def clean_text(text):
    return text.encode("utf-8", "ignore").decode("utf-8", "ignore")

# --- data_loader.py ---
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from utils import clean_text

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text += content + "\n"
    return clean_text(text)

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=500)
    return splitter.split_text(text)

# --- embeddings.py ---
from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from config import EMBEDDING_MODEL

def get_embeddings():
    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

# --- vector_store.py ---
from langchain.vectorstores import FAISS
# from embeddings import get_embeddings

def store_text_chunks(text_chunks):
    embeddings = get_embeddings()
    store = FAISS.from_texts(text_chunks, embedding=embeddings)
    store.save_local("faiss_index")

def load_vector_store():
    embeddings = get_embeddings()
    return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# --- chains.py ---
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import ChatGoogleGenerativeAI
# from config import ONLINE_MODEL

def get_prompt():
    return PromptTemplate(template="""
    You are an intelligent assistant answering questions based strictly on the given context. Use the information provided to answer comprehensively, accurately, and in the appropriate format.

    Guidelines:
    - Use only the given context to answer. If the answer is not found, respond: "Answer is not available in the context."
    - Never fabricate or assume facts not in the context.
    - Treat 'NITJ', 'nitj', 'institute', and 'Dr. B.R. Ambedkar National Institute of Technology' as referring to the same entity.
    - If a question involves steps, procedures, or processes, use clear bullet points.
    - If a numerical answer is requested (e.g. how many clubs), and the number is not directly given, count based on the context.
    - Answer in the tone and format suitable for the question type:
    - For definitions or factual queries: provide concise, formal answers.
    - For lists: use bullet points.
    - For how-to or process questions: step-by-step format.
    - For comparisons: use tables or summaries.
    - Do NOT search externally; use only the provided context.
    - If the context is insufficient, say so clearly.
    - If counting not provided directly then count and give answer.

        Context:
        {context}

        Question:
        {question}

        Answer:
    """, input_variables=["context", "question"])

def get_online_chain():
    model = ChatGoogleGenerativeAI(model=ONLINE_MODEL, temperature=0.3)
    return load_qa_chain(model, chain_type="stuff", prompt=get_prompt())

# --- offline_model.py ---
from langchain.llms import LlamaCpp
from langchain.chains.question_answering import load_qa_chain
# from chains import get_prompt

# NOTE: Download and configure a local GGUF model file before using

def get_offline_chain():
    llm = LlamaCpp(model_path="./models/local_model.gguf", n_ctx=2048)
    return load_qa_chain(llm, chain_type="stuff", prompt=get_prompt())

# --- app.py ---
import streamlit as st
# from config import USE_OFFLINE
# from data_loader import get_pdf_text, get_text_chunks
# from vector_store import store_text_chunks, load_vector_store
# from chains import get_online_chain
# from offline_model import get_offline_chain
# from utils import clean_text

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.set_page_config(page_title="NITJ Chatbot", layout="wide")

st.title("🎓 NITJ Academic Assistant")
user_question = st.text_input("Ask your question about NITJ:")

if user_question:
    st.session_state.chat_history.append(("🧑‍💻", user_question))
    store = load_vector_store()
    docs = store.similarity_search(user_question, k=3)

    if docs:
        chain = get_offline_chain() if USE_OFFLINE else get_online_chain()
        response = chain.invoke({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        answer = response['output_text']
    else:
        answer = "Answer is not available in the context."

    st.session_state.chat_history.append(("🤖", answer))
    st.markdown(answer)

# Chat history
st.markdown("---")
for role, message in reversed(st.session_state.chat_history):
    st.markdown(f"**{role}**: {message}")

# Sidebar
with st.sidebar:
    st.header("Upload Academic PDFs")
    docs = st.file_uploader("Upload Files", accept_multiple_files=True)
    if st.button("Process Documents"):
        with st.spinner("Processing..."):
            raw_text = get_pdf_text(docs)
            chunks = get_text_chunks(raw_text)
            store_text_chunks(chunks)
            st.success("Documents processed!")

    if st.button("Clear Chat"):
        st.session_state.chat_history = []

    st.info("""
        - Toggle model type via .env file (USE_OFFLINE=True/False)
        - Make sure local model is available if using offline mode
    """)
