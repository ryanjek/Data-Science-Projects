import streamlit as st
import tempfile
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch

# Set page configuration
st.set_page_config(page_title="PDF Q&A Assistant", layout="wide")

# Determine the best device for model inference
def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

# Cache model to avoid reloading on each rerun
@st.cache_resource
def initialize_llm_model(model_name="google/flan-t5-large"):
    """
    Initialize and cache the language model for text generation.
    
    Args:
        model_name: Name of the HuggingFace model to use
        
    Returns:
        A HuggingFacePipeline object
    """
    device = get_device()
    st.info(f"Loading model on {device}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    
    # Configure the pipeline with appropriate parameters
    hf_pipeline = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=512,  # Maximum response length
        do_sample=False,  # Use greedy decoding for more factual responses
        temperature=0.1,  # Lower temperature for more focused answers
    )
    
    return HuggingFacePipeline(pipeline=hf_pipeline)

# Load and split PDF
@st.cache_data
def load_and_split_pdf(pdf_path):
    """
    Load a PDF file and split it into chunks for processing.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        List of document chunks with content and metadata
    """
    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        # Create a text splitter with appropriate parameters
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,           # Larger chunks for better context
            chunk_overlap=200,         # Sufficient overlap to maintain context between chunks
            separators=["\n\n", "\n", " ", ""],  # Try paragraph breaks first
            length_function=len,
        )
        
        # Split the documents and preserve metadata
        chunks = text_splitter.split_documents(documents)
        return chunks
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return []

# Build vector store
@st.cache_data
def build_vector_store(chunks, embedding_model="sentence-transformers/all-mpnet-base-v2"):
    """
    Create a vector store from document chunks.
    
    Args:
        chunks: List of document chunks
        embedding_model: Model to use for embeddings
        
    Returns:
        FAISS vector store
    """
    try:
        # Initialize the embedding model
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        
        # Create the vector store with documents (preserving metadata)
        vector_store = FAISS.from_documents(chunks, embeddings)
        return vector_store
    except Exception as e:
        st.error(f"Error building vector store: {str(e)}")
        return None

# Perform RAG Q&A
def perform_rag_qa(question, vector_store, llm, k=3):
    """
    Perform retrieval-augmented generation to answer the question.
    
    Args:
        question: User's question
        vector_store: Vector store containing document chunks
        llm: Language model for answer generation
        k: Number of chunks to retrieve
        
    Returns:
        Generated answer and source documents
    """
    # Configure the retriever to get the most relevant chunks
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )
    
    # Build a more informative prompt that guides the model
    retrieved_docs = retriever.get_relevant_documents(question)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    prompt = f"""
    Use only the following context to answer the question. If the answer is not contained in the context, 
    say "I don't have enough information to answer this question." Be concise and accurate.
    
    Context:
    {context}
    
    Question: {question}
    
    Answer:
    """
    
    # Generate the answer using the LLM
    answer = llm.invoke(prompt)
    
    return answer, retrieved_docs

# UI Components
def sidebar():
    """Create and configure the sidebar"""
    with st.sidebar:
        st.title("💬 PDF Assistant")
        st.markdown("---")
        st.markdown(
            """
            ### How to use:
            1. Upload a PDF document
            2. Wait for processing to complete
            3. Ask questions about the content
            
            The assistant will retrieve relevant sections and answer your questions based on the document.
            """
        )
        st.markdown("---")
        st.write("Advanced Settings:")
        k_value = st.slider("Number of chunks to retrieve", min_value=1, max_value=10, value=3)
        
        return k_value

# Main app
def main():
    # Set up the sidebar
    k_value = sidebar()
    
    # Main content
    st.title("📄 PDF Question Answering Assistant")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    
    if uploaded_file:
        # Create a temporary file to store the PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            pdf_path = tmp_file.name
        
        try:
            # Process the PDF
            with st.spinner("Processing your PDF... This may take a moment."):
                chunks = load_and_split_pdf(pdf_path)
                
                if not chunks:
                    st.error("Could not extract text from the PDF. Please try another file.")
                    return
                
                # Create vector store
                vector_store = build_vector_store(chunks)
                
                if not vector_store:
                    st.error("Failed to create vector store. Please try again.")
                    return
                
                st.success(f"✅ PDF processed successfully! {len(chunks)} chunks extracted.")
            
            # Initialize LLM after PDF is processed
            llm = initialize_llm_model()
            
            # Question input
            st.markdown("### Ask a question about your document:")
            question = st.text_input("Your question:", placeholder="What is the main topic of this document?")
            
            if question:
                with st.spinner("Finding answer..."):
                    answer, sources = perform_rag_qa(question, vector_store, llm, k=k_value)
                
                # Display the answer
                st.markdown("### Answer:")
                st.markdown(f"**{answer}**")
                
                # Show source documents with metadata
                with st.expander("View source chunks", expanded=False):
                    for i, doc in enumerate(sources):
                        st.markdown(f"**Chunk {i+1}**")
                        # Format the metadata nicely
                        if hasattr(doc, 'metadata') and doc.metadata:
                            if 'page' in doc.metadata:
                                st.markdown(f"📄 **Page:** {doc.metadata['page']}")
                            if 'source' in doc.metadata:
                                st.markdown(f"📂 **Source:** {os.path.basename(doc.metadata['source'])}")
                        # Display the content
                        st.markdown(f"```\n{doc.page_content}\n```")
                        st.markdown("---")
        
        finally:
            # Clean up the temporary file
            if os.path.exists(pdf_path):
                os.unlink(pdf_path)

if __name__ == "__main__":
    main()
