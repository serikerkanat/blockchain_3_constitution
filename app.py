import os
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="AI Assistant for Constitution of Kazakhstan",
    page_icon="📜",
    layout="wide",
)

# Initialize session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "model_initialized" not in st.session_state:
    st.session_state.model_initialized = False

def load_documents(files):
    """Load and process uploaded documents"""
    documents = []
    for file in files:
        file_path = os.path.join("documents", file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        
        try:
            if file.name.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            elif file.name.endswith(".docx"):
                loader = Docx2txtLoader(file_path)
            elif file.name.endswith(".txt"):
                loader = TextLoader(file_path)
            else:
                st.error(f"Unsupported file type: {file.name}")
                continue
            
            documents.extend(loader.load())
        except Exception as e:
            st.error(f"Error loading {file.name}: {str(e)}")
            continue
    
    return documents

def process_documents(documents):
    """Process documents and create vector store"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    texts = text_splitter.split_documents(documents)
    
    # Initialize with progress
    progress_text = st.empty()
    progress_text.text("Initializing embeddings...")
    
    embeddings = OllamaEmbeddings(model="phi3:3.8b")
    progress_text.text("Creating vector store...")
    
    vector_store = Chroma.from_documents(
        texts,
        embeddings,
        collection_name="constitution-kz",
        persist_directory="chroma_db",
    )
    
    progress_text.empty()
    return vector_store

def initialize_qa_chain(vector_store):
    """Initialize the QA chain"""
    llm = Ollama(model="phi3:3.8b", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vector_store.as_retriever(),
        return_source_documents=True,
    )
    return qa_chain

def main():
    st.title("🇰🇿 AI Assistant for Constitution of Kazakhstan")
    st.markdown("Ask questions about the Constitution of the Republic of Kazakhstan")
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        
        uploaded_files = st.file_uploader(
            "Upload documents (PDF, DOCX, TXT)",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
        )
        
        if uploaded_files:
            os.makedirs("documents", exist_ok=True)
            with st.spinner("Processing documents..."):
                try:
                    documents = load_documents(uploaded_files)
                    st.session_state.vector_store = process_documents(documents)
                    st.success("Documents processed successfully!")
                except Exception as e:
                    st.error(f"Error processing documents: {str(e)}")
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown(
            "This AI assistant can answer questions about the Constitution "
            "of the Republic of Kazakhstan using Phi-3 3.8B model."
        )
        st.markdown("Model: **phi3:3.8b** (Ollama)")
    
    # Chat interface
    st.subheader("Ask your question")
    
    # Initialize with default constitution
    if not st.session_state.vector_store and not uploaded_files:
        with st.spinner("Loading default constitution (this may take a minute)..."):
            try:
                from langchain_community.document_loaders import WebBaseLoader
                loader = WebBaseLoader("https://www.akorda.kz/en/constitution-of-the-republic-of-kazakhstan-50912")
                documents = loader.load()
                st.session_state.vector_store = process_documents(documents)
                st.success("Default constitution loaded!")
            except Exception as e:
                st.error(f"Failed to load default constitution: {str(e)}")
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # User input
    if prompt := st.chat_input("Ask your question about the constitution"):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        if st.session_state.vector_store:
            with st.spinner("Generating answer..."):
                try:
                    start_time = time.time()
                    qa_chain = initialize_qa_chain(st.session_state.vector_store)
                    result = qa_chain({"query": prompt})
                    response = result["result"]
                    
                    if "source_documents" in result:
                        sources = list({doc.metadata.get("source", "") for doc in result["source_documents"]})
                        if sources:
                            response += "\n\nSources:\n- " + "\n- ".join(sources)
                    
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                    st.info(f"Response generated in {time.time()-start_time:.1f} seconds")
                except Exception as e:
                    response = f"Error generating answer: {str(e)}"
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
        else:
            response = "Please upload documents first to enable question answering."
            st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        with st.chat_message("assistant"):
            st.markdown(response)

if __name__ == "__main__":
    main()