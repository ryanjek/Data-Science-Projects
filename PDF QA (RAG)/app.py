import streamlit as st
import tempfile 
import os
from llm import initialize_llm_model, initialize_llm_model_openai
from utils import load_and_split_pdfs, build_vector_store
from rag import perform_rag_qa

# Set page config
st.set_page_config(page_title="PDF Q&A Assistant", layout="wide")

def sidebar():
    with st.sidebar:
        st.title("PDF Q&A Assistant")
        st.markdown("---")
        st.markdown(
            """
            ### How to use:
            1. Upload relevant PDF documents
            2. Wait for processing to complete
            3. Ask questions about the content
            """
        )
        st.markdown("---")
        # User can select either OpenAI gpt-3.5-turbo or Google FLAN-T5
        model_choice = st.radio("Choose LLM provider:", ["OpenAI", "HuggingFace"])
        # User can select top K chunk
        k_value = st.slider("Number of chunks to retrieve", min_value=1, max_value=10, value=3)
        return model_choice, k_value

def main():
    model_choice, k_value = sidebar()

    # Check if user is using the same model. If model changes then reload the LLM instance.
    if st.session_state.get("model_choice") != model_choice:
        st.session_state.model_choice = model_choice
        if "llm" in st.session_state:
            del st.session_state.llm  

    st.title("PDF Question Answering Assistant")
    # Upload PDFs
    uploaded_files = st.file_uploader("Upload one or more PDFs", type="pdf", accept_multiple_files=True)

    # Check if PDFs have been uploaded before
    if uploaded_files:
        file_names = [f.name for f in uploaded_files]
        file_changed = file_names != st.session_state.get("uploaded_filenames")

        if file_changed:
            # Clear all cached session state related to the previous files
            st.session_state.clear()
            st.session_state.uploaded_filenames = file_names

            # Save the uploaded files temporarily into the disk so that they can be passed to PyPDF parser later.
            # Will not persist once Streamlit is closed or restarted
            file_paths = []
            for file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(file.getvalue())
                    file_paths.append(tmp.name)

            st.session_state.pdf_paths = file_paths

        elif "pdf_paths" not in st.session_state:
            # Fall back edge case
            file_paths = []
            for file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(file.getvalue())
                    file_paths.append(tmp.name)
            st.session_state.pdf_paths = file_paths

        # Extract text from the PDFs and split them into overlapping chunks for the RAG
        if "chunks" not in st.session_state:
            with st.spinner("Processing your PDFs..."):
                chunks = load_and_split_pdfs(st.session_state.pdf_paths)
                if not chunks:
                    st.error("Could not extract text from the uploaded PDFs.")
                    return
                st.session_state.chunks = chunks

        # Use mpnet-base-v2 for pretrained sentence embedding to embed each chunk and store in FAISS index
        if "vector_store" not in st.session_state:
            with st.spinner("Building vector store..."):
                vector_store = build_vector_store(st.session_state.chunks)
                if not vector_store:
                    st.error("Failed to create vector store.")
                    return
                st.session_state.vector_store = vector_store

        st.success(f"PDFs processed! {len(st.session_state.chunks)} chunks ready.")

        if "llm" not in st.session_state:
            with st.spinner("Loading model..."):
                try:
                    if model_choice == "OpenAI":
                        st.session_state.llm = initialize_llm_model_openai()
                    else:
                        st.session_state.llm = initialize_llm_model()
                except Exception as e:
                    st.error(f"Error loading model: {e}")
                    return

        # Text box for user to input question
        question = st.text_input("Your question:", placeholder="e.g., What is the main topic?")

        if question:
            with st.spinner("Generating answer..."):
                # Retrieve top k relevant chunks from vectorDB
                # Format a prompt with it
                # Send to LLM to generate answer
                answer, sources = perform_rag_qa(
                    question, 
                    st.session_state.vector_store, 
                    st.session_state.llm, 
                    k=k_value
                )

            # Generated answer
            st.markdown("### Answer:")
            st.markdown(answer)

            # To show actual context for explainability.
            with st.expander("View source chunks"):
                for i, doc in enumerate(sources):
                    st.markdown(f"**Chunk {i+1}:**")
                    if doc.metadata:
                        if 'page' in doc.metadata:
                            st.markdown(f"Page: {doc.metadata['page']}")
                        if 'source' in doc.metadata:
                            st.markdown(f"Source: {os.path.basename(doc.metadata['source'])}")
                    st.code(doc.page_content)
                    st.markdown("---")

# Runs only if the script is run directly
if __name__ == "__main__":
    main()
