from dotenv import load_dotenv
import streamlit as st
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# -----------------------------
# Load Environment Variables
# -----------------------------
load_dotenv()

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="AI RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown(
    """
    <style>

    .stApp {
        background-color: #0E1117;
        color: white;
    }

    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: bold;
        margin-bottom: 10px;
    }

    .sub-text {
        text-align: center;
        font-size: 18px;
        color: #A0AEC0;
        margin-bottom: 30px;
    }

    .chat-user {
        background-color: #1E293B;
        padding: 15px;
        border-radius: 12px;
        margin-bottom: 10px;
    }

    .chat-ai {
        background-color: #111827;
        padding: 15px;
        border-radius: 12px;
        margin-bottom: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Title
# -----------------------------
st.markdown(
    "<div class='main-title'>🤖 AI-Powered RAG Chatbot</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='sub-text'>Upload a PDF and chat with your document using AI</div>",
    unsafe_allow_html=True
)

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:

    st.header("📂 Upload PDF")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type="pdf"
    )

    st.markdown("---")

    st.subheader("⚡ Features")

    st.write("✅ PDF Upload")
   
    st.write("✅ AI Responses")
   

# -----------------------------
# Session State
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -----------------------------
# Process PDF
# -----------------------------
if uploaded_file is not None:

    # Save uploaded PDF temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_path = tmp_file.name

    # Load PDF
    loader = PyPDFLoader(temp_path)
    documents = loader.load()

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    texts = text_splitter.split_documents(documents)

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Create FAISS vector store
    vectorstore = FAISS.from_documents(
        texts,
        embeddings
    )

    # Load LLM
    llm = ChatGroq(
        model_name="llama-3.1-8b-instant",
        temperature=0
    )

    st.success("✅ PDF uploaded and processed successfully!")

    # -----------------------------
    # Chat Input
    # -----------------------------
    query = st.chat_input("Ask something about the PDF...")

    if query:

        # Save user message
        st.session_state.messages.append({
            "role": "user",
            "content": query
        })

        # Retrieve relevant chunks
        docs = vectorstore.similarity_search(query, k=3)

        # Combine retrieved context
        context = "\n\n".join([doc.page_content for doc in docs])

        # Prompt
        prompt = f"""
        Answer the question using the context below.

        Context:
        {context}

        Question:
        {query}
        """

        # Generate response
        response = llm.invoke(prompt)

        ai_response = response.content

        # Save AI response
        st.session_state.messages.append({
            "role": "assistant",
            "content": ai_response
        })

    # -----------------------------
    # Display Chat History
    # -----------------------------
    for message in st.session_state.messages:

        if message["role"] == "user":

            st.markdown(
                f"""
                <div class="chat-user">
                <b>🧑 You:</b><br><br>
                {message['content']}
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                f"""
                <div class="chat-ai">
                <b>🤖 AI:</b><br><br>
                {message['content']}
                </div>
                """,
                unsafe_allow_html=True
            )

else:

    st.info("📄 Please upload a PDF to begin chatting.")